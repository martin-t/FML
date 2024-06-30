//! A simple x86-64 assembler.
//!
//! Takes an instruction described using the `Instr` enum (there's no assembly parser),
//! converts it into an `Encoding` and then serializes it into bytes.
//! Can also deserialize bytes back into an `Encoding` for easier debugging.
//!
//! `Encoding` serves as an intermediate representation which can be inspected
//! to find the meaning of individual bytes.
//!
//! # Reference
//!
//! The Intel x86 manual can be downloaded here:
//! https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html
//!
//! Most useful parts of the Intel manual - Volume 2:
//! - 2.1 INSTRUCTION FORMAT FOR PROTECTED MODE, REAL-ADDRESS MODE, AND VIRTUAL-8086 MODE
//!     - A diagram of the format and tables describing the ModR/M and SIB bytes.
//! - 2.2 IA-32E MODE
//!     - Diagrams explaining the REX prefix and tables with special cases in 64 bit mode.
//! - 3.1.1.1 Opcode Column in the Instruction Summary Table (Instructions without VEX Prefix)
//!     - Explains how to read the pages describing individual instructions.
//!
//! # Online assemblers
//!
//! It might be interesting to compare the output with real assemblers.
//! Some are available online, they support a much larger set of instructions
//! but don't provide an inspetable intermediate representation like our `Encoding`.
//!
//! - https://asm.x32.dev/
//!     - instant feedback
//!     - still works without internet connection
//!     - can silently ignore instructions
//!         - e.g. when using the wrong syntax like `dword ptr` vs `dword`
//!     - sometimes gets stuck (and even block other tabs)
//!     - treats numbers as hex, no way to specify decimal
//!     - recommended usage: switch syntax to 64bit (NASM), output to C array
//! - https://defuse.ca/online-x86-assembler.htm
//! - https://disasm.pro/
//!     - instant feedback
//!     - requires network
//!     - will silently stop updating if indirect addressing doesn't use "ptr"

use std::fmt::{self, Display, Formatter};

use fnv::FnvHashMap;
use smallvec::SmallVec;

use crate::jit::asm_repr::*;

impl Instr {
    pub fn encode(self) -> Encoding {
        // Useful reading:
        // - Implicit uses and special-ness for each register:
        //   https://stackoverflow.com/a/51347294/2468852
        // - Table 2-5. Special Cases of REX Encodings

        match self {
            Instr::Nop => Encoding::just_opcode(0x90),
            Instr::AddRR(dst, src) => {
                // 01 /r            ADD r/m32, r32
                // REX.W + 01 /r    ADD r/m64, r64
                Self::encode_reg_reg(0x01, dst, src)
            }
            Instr::AddMR(dst, src) => {
                // 01 /r            ADD r/m32, r32
                // REX.W + 01 /r    ADD r/m64, r64
                Self::encode_mem_reg(0x01, dst, src)
            }
            Instr::AddRM(dst, src) => {
                // 03 /r            ADD r32, r/m32
                // REX.W + 03 /r    ADD r64, r/m64
                Self::encode_mem_reg(0x03, src, dst)
            }
            Instr::AddRI(dst, imm) => {
                // 81 /0 id             ADD r/m32, imm32
                // REX.W + 81 /0 id     ADD r/m64, imm32
                Self::encode_reg_imm(0x81, 0, dst, imm)
            }
            Instr::AddMI(dst, imm) => {
                // 81 /0 id             ADD r/m32, imm32
                // REX.W + 81 /0 id     ADD r/m64, imm32
                Self::encode_mem_imm(0x81, 0, dst, imm)
            }
            Instr::AndRR(dst, src) => {
                // 21 /r            AND r/m32, r32
                // REX.W + 21 /r    AND r/m64, r64
                Self::encode_reg_reg(0x21, dst, src)
            }
            Instr::AndMR(dst, src) => {
                // 21 /r            AND r/m32, r32
                // REX.W + 21 /r    AND r/m64, r64
                Self::encode_mem_reg(0x21, dst, src)
            }
            Instr::AndRM(dst, src) => {
                // 23 /r            AND r32, r/m32
                // REX.W + 23 /r    AND r64, r/m64
                Self::encode_mem_reg(0x23, src, dst)
            }
            Instr::AndRI(dst, imm) => {
                // 81 /4 id             AND r/m32, imm32
                // REX.W + 81 /4 id     AND r/m64, imm32
                Self::encode_reg_imm(0x81, 4, dst, imm)
            }
            Instr::AndMI(dst, imm) => {
                // 81 /4 id             AND r/m32, imm32
                // REX.W + 81 /4 id     AND r/m64, imm32
                Self::encode_mem_imm(0x81, 4, dst, imm)
            }
            Instr::_CallRel(rel) => {
                // E8 cd            CALL rel32

                // The size of this instruction is 5 bytes
                // and the offset is relative to the next instruction
                // so we subtract 5.
                let rel = rel
                    .checked_sub(5)
                    .expect("CallRel attempted to subtract with underflow");
                Self::encode_call32(0xE8, rel)
            }
            Instr::CallRel32(rel) => Self::encode_call32(0xE8, rel),
            Instr::CallAbsR(abs) => {
                // FF /2            CALL r/m64
                let (abs_b, abs_bbb) = abs.encode();
                let rex = Rex {
                    w: 0,
                    r: 0,
                    x: 0,
                    b: abs_b,
                };
                let opcode = 0xFF;
                let modrm = ModRm {
                    mod_: MODRM_MOD_DIRECT,
                    reg: 2,
                    rm: abs_bbb,
                };
                Encoding {
                    rex: if rex.is_used() { Some(rex) } else { None },
                    opcode,
                    modrm: Some(modrm),
                    ..Default::default()
                }
            }
            Instr::CallAbsM(abs) => {
                // FF /2            CALL r/m64
                let EncodedMem {
                    rex_x,
                    rex_b,
                    mod_,
                    rm,
                    sib,
                    disp,
                } = abs.encode();
                let rex = Rex {
                    w: 0,
                    r: 0,
                    x: rex_x,
                    b: rex_b,
                };
                let opcode = 0xFF;
                let modrm = ModRm { mod_, reg: 2, rm };
                Encoding {
                    rex: if rex.is_used() { Some(rex) } else { None },
                    opcode,
                    modrm: Some(modrm),
                    sib,
                    disp,
                    ..Default::default()
                }
            }
            Instr::CmpRR(dst, src) => Self::encode_reg_reg(0x39, dst, src),
            Instr::CmpMR(dst, src) => Self::encode_mem_reg(0x39, dst, src),
            Instr::CmpRM(dst, src) => Self::encode_mem_reg(0x3B, src, dst),
            Instr::CmpRI(dst, imm) => Self::encode_reg_imm(0x81, 7, dst, imm),
            Instr::CmpMI(dst, imm) => Self::encode_mem_imm(0x81, 7, dst, imm),
            Instr::Cdq => {
                // 99                CDQ
                Encoding {
                    opcode: 0x99,
                    ..Default::default()
                }
            }
            Instr::Cqo => {
                // 48 99             CQO
                let rex = Rex { w: 1, r: 0, x: 0, b: 0 };
                Encoding {
                    rex: Some(rex),
                    opcode: 0x99,
                    ..Default::default()
                }
            }
            Instr::Hlt => Encoding::just_opcode(0xF4),
            Instr::IdivR(op) => {
                // F7 /7            IDIV r/m32
                // REX.W + F7 /7    IDIV r/m64
                Self::encode_reg(0xF7, 7, op)
            }
            Instr::IdivM(op) => {
                // F7 /7            IDIV r/m32
                // REX.W + F7 /7    IDIV r/m64
                Self::encode_mem(0xF7, 7, op)
            }
            Instr::ImulR(op) => {
                // F7 /5            IMUL r/m32
                // REX.W + F7 /5    IMUL r/m64
                Self::encode_reg(0xF7, 5, op)
            }
            Instr::ImulM(op) => {
                // F7 /5            IMUL r/m32
                // REX.W + F7 /5    IMUL r/m64
                Self::encode_mem(0xF7, 5, op)
            }
            Instr::ImulRR(dst, src) => {
                // 0F AF /r             IMUL r32, r/m32
                // REX.W + 0F AF /r     IMUL r64, r/m64
                Encoding {
                    opcode2: Some(0xAF),
                    ..Self::encode_reg_reg(0x0F, src, dst)
                }
            }
            Instr::ImulRM(dst, src) => {
                // 0F AF /r             IMUL r32, r/m32
                // REX.W + 0F AF /r     IMUL r64, r/m64
                Encoding {
                    opcode2: Some(0xAF),
                    ..Self::encode_mem_reg(0x0F, src, dst)
                }
            }
            Instr::Int3 => Encoding::just_opcode(0xCC),
            Instr::Int1 => Encoding::just_opcode(0xF1),
            Instr::_Je(rel) => Self::encode_jump(0x74, 0x0F, Some(0x84), rel),
            Instr::_Jg(rel) => Self::encode_jump(0x7F, 0x0F, Some(0x8F), rel),
            Instr::_Jge(rel) => Self::encode_jump(0x7D, 0x0F, Some(0x8D), rel),
            Instr::_Jl(rel) => Self::encode_jump(0x7C, 0x0F, Some(0x8C), rel),
            Instr::_Jle(rel) => Self::encode_jump(0x7E, 0x0F, Some(0x8E), rel),
            Instr::_Jne(rel) => Self::encode_jump(0x75, 0x0F, Some(0x85), rel),
            Instr::Je32(rel) => Self::encode_jump32(0x0F, Some(0x84), rel),
            Instr::Jg32(rel) => Self::encode_jump32(0x0F, Some(0x8F), rel),
            Instr::Jge32(rel) => Self::encode_jump32(0x0F, Some(0x8D), rel),
            Instr::Jl32(rel) => Self::encode_jump32(0x0F, Some(0x8C), rel),
            Instr::Jle32(rel) => Self::encode_jump32(0x0F, Some(0x8E), rel),
            Instr::Jne32(rel) => Self::encode_jump32(0x0F, Some(0x85), rel),
            Instr::_JmpRel(rel) => {
                // EB cb        JMP rel8
                // E9 cd        JMP rel32

                Self::encode_jump(0xEB, 0xE9, None, rel)
            }
            Instr::JmpRel32(rel) => Self::encode_jump32(0xE9, None, rel),
            Instr::MovRR(dst, src) => {
                // 89 /r            MOV r/m32, r32
                // REX.W + 89 /r    MOV r/m64, r64
                Self::encode_reg_reg(0x89, dst, src)
            }
            Instr::MovMR(dst, src) => {
                // 89 /r            MOV r/m32, r32
                // REX.W + 89 /r    MOV r/m64, r64
                Self::encode_mem_reg(0x89, dst, src)
            }
            Instr::MovRM(dst, src) => {
                // 8B /r            MOV r32, r/m32
                // REX.W + 8B /r    MOV r64, r/m64
                Self::encode_mem_reg(0x8B, src, dst)
            }
            Instr::MovRI(dst, imm64) => {
                // B8+ rd id            MOV r32, imm32
                // REX.W + B8+ rd io    MOV r64, imm64
                // REX.W + C7 /0 id     MOV r/m64, imm32

                let (dst_b, dst_bbb) = dst.encode();
                let rex = Rex {
                    w: dst.is_64bit().into(),
                    r: 0,
                    x: 0,
                    b: dst_b,
                };

                let opcode;
                let modrm;
                let imm;

                // Existing assemblers seem to try to use fewer bytes when possible
                // and use the B8 opcode with a 32 bit immediate when possible,
                // even when moving to a 64 bit register. We do the same
                // to make generating tests easier.
                //
                // Examples:
                // mov eax, 1                       B8 01 00 00 00
                // mov rax, 1                       48 C7 C0 01 00 00 00
                // mov rax, 9223372036854775807     48 B8 FF FF FF FF FF FF FF 7F

                let imm32: Option<i32> = imm64.try_into().ok();
                if let Some(imm32) = imm32 {
                    if dst.is_64bit() {
                        opcode = 0xC7;
                        modrm = Some(ModRm {
                            mod_: MODRM_MOD_DIRECT,
                            reg: 0,
                            rm: dst_bbb,
                        });
                    } else {
                        opcode = 0xB8 + dst_bbb;
                        modrm = None;
                    };
                    imm = Num::Num32(imm32);
                } else {
                    opcode = 0xB8 + dst_bbb;
                    modrm = None;
                    imm = Num::Num64(imm64);
                }

                Encoding {
                    rex: if rex.is_used() { Some(rex) } else { None },
                    opcode,
                    modrm,
                    imm: Some(imm),
                    ..Default::default()
                }
            }
            Instr::MovMI(dst, imm) => {
                // C7 /0 id             MOV r/m32, imm32
                // REX.W + C7 /0 id     MOV r/m64, imm32
                Self::encode_mem_imm(0xC7, 0, dst, imm)
            }
            Instr::OrRR(dst, src) => Self::encode_reg_reg(0x09, dst, src),
            Instr::OrMR(dst, src) => Self::encode_mem_reg(0x09, dst, src),
            Instr::OrRM(dst, src) => Self::encode_mem_reg(0x0B, src, dst),
            Instr::OrRI(dst, imm) => Self::encode_reg_imm(0x81, 1, dst, imm),
            Instr::OrMI(dst, imm) => Self::encode_mem_imm(0x81, 1, dst, imm),
            Instr::PopR(op) => {
                // 58+ rd            POP r64
                Self::encode_pop_push_reg(0x58, op)
            }
            Instr::PopM(op) => {
                // 8F /0             POP r/m64
                Self::encode_mem(0x8F, 0, op)
            }
            Instr::PushR(op) => {
                // 50+ rd            PUSH r64
                Self::encode_pop_push_reg(0x50, op)
            }
            Instr::PushM(op)=>{
                // FF /6             PUSH r/m64
                Self::encode_mem(0xFF, 6, op)
            }
            Instr::Ret => {
                // In x64 we should only need ret near.
                Encoding::just_opcode(0xC3)
            }
            Instr::SubRR(dst, src) => Self::encode_reg_reg(0x29, dst, src),
            Instr::SubMR(dst, src) => Self::encode_mem_reg(0x29, dst, src),
            Instr::SubRM(dst, src) => Self::encode_mem_reg(0x2B, src, dst),
            Instr::SubRI(dst, imm) => Self::encode_reg_imm(0x81, 5, dst, imm),
            Instr::SubMI(dst, imm) => Self::encode_mem_imm(0x81, 5, dst, imm),
            Instr::TestRR(dst, src) => Self::encode_reg_reg(0x85, dst, src),
            Instr::TestMR(dst, src) => Self::encode_mem_reg(0x85, dst, src),
            Instr::TestRI(dst, imm) => Self::encode_reg_imm(0xF7, 0, dst, imm),
            Instr::TestMI(dst, imm) => Self::encode_mem_imm(0xF7, 0, dst, imm),
            Instr::Ud2 => Encoding {
                opcode: 0x0F,
                opcode2: Some(0x0B),
                ..Default::default()
            },

            Instr::Label(_)
            | Instr::CallLabel(_)
            | Instr::JeLabel(_)
            | Instr::JgLabel(_)
            | Instr::JgeLabel(_)
            | Instr::JlLabel(_)
            | Instr::JleLabel(_)
            | Instr::JneLabel(_)
            | Instr::JmpLabel(_) => {
                unreachable!("Encountered {self} - virtual instructions must be removed before encoding")
            }
        }
    }

    fn encode_reg(opcode: u8, extension: u8, dst: Reg) -> Encoding {
        assert!(extension <= 7);

        let (dst_b, dst_bbb) = dst.encode();
        let rex = Rex {
            w: dst.is_64bit().into(),
            r: 0,
            x: 0,
            b: dst_b,
        };

        let modrm = ModRm {
            mod_: MODRM_MOD_DIRECT,
            reg: extension,
            rm: dst_bbb,
        };
        Encoding {
            rex: if rex.is_used() { Some(rex) } else { None },
            opcode,
            modrm: Some(modrm),
            ..Default::default()
        }
    }

    fn encode_mem(opcode: u8, extension: u8, dst: Mem) -> Encoding {
        let EncodedMem {
            rex_x,
            rex_b,
            mod_,
            rm,
            sib,
            disp,
        } = dst.encode();
        let rex = Rex {
            w: 0,
            r: 0,
            x: rex_x,
            b: rex_b,
        };

        let modrm = ModRm {
            mod_,
            reg: extension,
            rm,
        };
        Encoding {
            rex: if rex.is_used() { Some(rex) } else { None },
            opcode,
            modrm: Some(modrm),
            sib,
            disp,

            ..Default::default()
        }
    }

    /// Note there are often 2 ways to encode some instructions.
    /// E.g. AddRR can be 0x01 or 0x03 with opeerands swapped.
    /// We do what NASM does to make generating tests easier.
    fn encode_reg_reg(opcode: u8, rm: Reg, reg: Reg) -> Encoding {
        assert_eq!(rm.is_64bit(), reg.is_64bit());

        let (rm_b, rm_bbb) = rm.encode();
        let (reg_r, reg_rrr) = reg.encode();
        let rex = Rex {
            w: rm.is_64bit().into(),
            r: reg_r,
            x: 0,
            b: rm_b,
        };

        let modrm = ModRm {
            mod_: MODRM_MOD_DIRECT,
            reg: reg_rrr,
            rm: rm_bbb,
        };

        Encoding {
            rex: if rex.is_used() { Some(rex) } else { None },
            opcode,
            modrm: Some(modrm),
            ..Default::default()
        }
    }

    fn encode_mem_reg(opcode: u8, mem: Mem, reg: Reg) -> Encoding {
        let EncodedMem {
            rex_x,
            rex_b,
            mod_,
            rm,
            sib,
            disp,
        } = mem.encode();

        let (reg_r, reg_rrr) = reg.encode();
        let rex = Rex {
            w: reg.is_64bit().into(),
            r: reg_r,
            x: rex_x,
            b: rex_b,
        };

        let modrm = ModRm { mod_, reg: reg_rrr, rm };

        Encoding {
            rex: if rex.is_used() { Some(rex) } else { None },
            opcode,
            modrm: Some(modrm),
            sib,
            disp,
            ..Default::default()
        }
    }

    fn encode_reg_imm(opcode: u8, extension: u8, dst: Reg, imm: i32) -> Encoding {
        Encoding {
            imm: Some(Num::Num32(imm)),
            ..Self::encode_reg(opcode, extension, dst)
        }
    }

    fn encode_mem_imm(opcode: u8, extension: u8, dst: Mem, imm: i32) -> Encoding {
        Encoding {
            imm: Some(Num::Num32(imm)),
            ..Self::encode_mem(opcode, extension, dst)
        }
    }

    fn encode_call32(opcode: u8, rel: i32) -> Encoding {
        Encoding {
            opcode,
            imm: Some(Num::Num32(rel)),
            ..Default::default()
        }
    }

    fn encode_jump(opcode_short: u8, opcode_near: u8, opcode2_near: Option<u8>, rel: i32) -> Encoding {
        // This is similar to CallRel but it has 2 variants.
        // Short jump is 2 bytes, near jump is 5 bytes.
        let rel_short = rel
            .checked_sub(2)
            .expect("encode_jump short attempted to subtract with underflow");
        let rel8: Option<i8> = rel_short.try_into().ok();
        if let Some(rel8) = rel8 {
            Self::encode_jump8(opcode_short, rel8)
        } else {
            let near_instr_len = if opcode2_near.is_some() { 6 } else { 5 };
            let rel_near = rel
                .checked_sub(near_instr_len)
                .expect("encode_jump near attempted to subtract with underflow");
            Self::encode_jump32(opcode_near, opcode2_near, rel_near)
        }
    }

    fn encode_jump8(opcode: u8, rel: i8) -> Encoding {
        Encoding {
            opcode,
            imm: Some(Num::Num8(rel)),
            ..Default::default()
        }
    }

    fn encode_jump32(opcode: u8, opcode2: Option<u8>, rel: i32) -> Encoding {
        Encoding {
            opcode,
            opcode2,
            imm: Some(Num::Num32(rel)),
            ..Default::default()
        }
    }

    fn encode_pop_push_reg(opcode: u8, op: Reg) -> Encoding {
        assert!(op.is_64bit());

        let (op_b, op_bbb) = op.encode();
        let rex = Rex {
            w: 0,
            r: 0,
            x: 0,
            b: op_b,
        };
        Encoding {
            rex: if rex.is_used() { Some(rex) } else { None },
            opcode: opcode + op_bbb,
            ..Default::default()
        }
    }
}

impl Mem {
    fn encode(self) -> EncodedMem {
        // Make sure there is at least one register
        // because using only displacement is a special case in 64-bit mode
        // which enables RIP-relative addressing.
        // LATER(martin-t) According to 2.2.1.6 RIP-Relative Addressing,
        // the solution is to always use a SIB byte.
        // However, NASM behaves weirdly when generating test-cases
        // and since I am unlikely to need this addressing mode,
        // I am just not gonna support it.
        // E.g. `add eax, [1]` behaves differently when encoding for
        // 32 vs 64 bits and it decodes into rip relative in 64 bit mode.
        assert!(self.base.is_some() || self.index.is_some());

        // Only support 64 bit registers for now.
        // LATER(martin-t) What is needed to support 32 bit regs?
        //  - Add Address-size override prefix (0x67)
        //  - The code was written with the assumption that only 64 bit registers
        //    will be used, make sure 32 bit regs are handled everywhere correctly.
        if let Some(base) = self.base {
            assert!(base.is_64bit());
        }
        if let Some(index) = self.index {
            assert!(index.is_64bit());

            // See Table 2-5. Special Cases of REX Encodings:
            // ESP/RSP as index can't be encoded at all.
            // R12D/R12 works normally.
            assert_ne!(index, Reg::Rsp, "RSP can't be used as index");
        }

        // Volume 2, Chapter 2.1.5, Table 2-2. 32-Bit Addressing Forms with the ModR/M Byte
        // page 530
        // These might be overriden later because of some special cases
        // depending on the addressing mode and registers used.
        let mut mod_;
        let mut disp;
        let disp_8bit: Option<i8> = self.disp.try_into().ok();
        if self.disp == 0 {
            mod_ = MODRM_MOD_NO_DISP_OR_NO_BASE;
            disp = None;
        } else if let Some(disp_8bit) = disp_8bit {
            mod_ = MODRM_MOD_DISP8;
            disp = Some(Num::Num8(disp_8bit));
        } else {
            mod_ = MODRM_MOD_DISP32;
            disp = Some(Num::Num32(self.disp));
        };

        let rex_x;
        let rex_b;
        let rm;
        let sib;
        if let Some(index) = self.index {
            // Index is used, we have to use a SIB byte.

            let (index_x, index_xxx) = index.encode();

            rex_x = index_x;
            rm = MODRM_RM_USE_SIB;

            if let Some(base) = self.base {
                // Base and index

                let (base_b, base_bbb) = base.encode();

                if mod_ == MODRM_MOD_NO_DISP_OR_NO_BASE && base_bbb == SIB_BASE_SPECIAL {
                    // Special case - see note beelow the SIB table.
                    // Mod == 0b00 and base == 0b101 would mean no base, just index and disp32.
                    // To avoid it, we use a dummy disp8.
                    mod_ = MODRM_MOD_DISP8;
                    disp = Some(Num::Num8(0));
                }

                rex_b = base_b;
                sib = Some(Sib {
                    scale: self.scale,
                    index: index_xxx,
                    base: base_bbb,
                });
            } else {
                // No base, just index - special case in the SIB table.

                // See note under Table 2-3. 32-Bit Addressing Forms with the SIB Byte
                mod_ = MODRM_MOD_NO_DISP_OR_NO_BASE;
                disp = disp.to_32_bits();

                rex_b = 0;
                sib = Some(Sib {
                    scale: self.scale,
                    index: index_xxx,
                    base: SIB_BASE_SPECIAL,
                });
            }
        } else {
            // Index is not used - no SIB.
            rex_x = 0;
            //sib = None;

            if let Some(base) = self.base {
                // Just base, no index

                let (base_b, base_bbb) = base.encode();

                if mod_ == MODRM_MOD_NO_DISP_OR_NO_BASE && base_bbb == SIB_BASE_SPECIAL {
                    // Special case - see note beelow the SIB table.
                    // Mod == 0b00 and base == 0b101 would mean no base, just index and disp32.
                    // To avoid it, we use a dummy disp8.
                    mod_ = MODRM_MOD_DISP8;
                    disp = Some(Num::Num8(0));
                }

                rex_b = base_b;
                rm = base_bbb;
                if base_bbb == Reg::Rsp.encode().1 {
                    sib = Some(Sib {
                        scale: 1,
                        index: SIB_INDEX_NONE,
                        base: base_bbb,
                    });
                } else {
                    sib = None;
                }
            } else {
                // No base, no index - special case in the ModR/M table:
                // mod == 0b00 and r/m == 0b101 means disp32.
                // However in 64-bit mode, it means RIP-relative addressing,
                // which we don't support - see note above.

                unimplemented!();
            }
        }

        EncodedMem {
            rex_x,
            rex_b,
            mod_,
            rm,
            sib,
            disp,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EncodedMem {
    /// 1 bit
    rex_x: u8,
    /// 1 bit
    rex_b: u8,
    /// 2 bits
    mod_: u8,
    /// 3 bits
    rm: u8,
    sib: Option<Sib>,
    disp: Option<Num>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Encoding {
    /// https://wiki.osdev.org/X86-64_Instruction_Encoding#Legacy_Prefixes
    ///
    /// There should be at most 4
    /// but AFAICT the manual doesn't forbid using more.
    pub legacy_prefixes: SmallVec<[u8; 4]>,
    pub rex: Option<Rex>,
    pub opcode: u8,
    /// Second byte of the opcode, if any.
    pub opcode2: Option<u8>,
    pub modrm: Option<ModRm>,
    pub sib: Option<Sib>,
    pub disp: Option<Num>,
    pub imm: Option<Num>,
}

impl Encoding {
    fn just_opcode(opcode: u8) -> Self {
        Encoding {
            opcode,
            ..Default::default()
        }
    }

    fn prefixes_opcode(legacy_prefixes: SmallVec<[u8; 4]>, rex: Option<Rex>, opcode: u8) -> Self {
        Encoding {
            legacy_prefixes,
            rex,
            opcode,
            ..Default::default()
        }
    }

    // LATER(martin-t) Use Write?
    pub fn serialize(&self, buf: &mut Vec<u8>) {
        let old_len = buf.len();

        if let Some(rex) = self.rex {
            buf.push(rex.serialize());
        }
        buf.push(self.opcode);
        if let Some(opcode2) = self.opcode2 {
            buf.push(opcode2);
        }
        if let Some(modrm) = self.modrm {
            buf.push(modrm.serialize());
        }
        if let Some(sib) = self.sib {
            buf.push(sib.serialize());
        }
        if let Some(disp) = self.disp {
            disp.serialize(buf);
        }
        if let Some(imm) = self.imm {
            imm.serialize(buf);
        }

        // Sanity check - x86 instructions are at most 15 bytes long
        assert!(buf.len() - old_len <= 15);
    }

    #[allow(dead_code)]
    pub fn serialize_to_vec(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.serialize(&mut buf);
        buf
    }

    /// Returns the encoded instruction and the number of bytes read.
    pub fn deserialize(bytes: &[u8]) -> (Encoding, usize) {
        assert!(!bytes.is_empty());
        let mut i = 0;

        // Legacy prefixes
        let mut legacy_prefixes = SmallVec::new();
        while let 0xf0 | 0xf2 | 0xf3 | 0x2e | 0x36 | 0x3e | 0x26 | 0x64 | 0x65 | 0x66 | 0x67 = bytes[i] {
            legacy_prefixes.push(bytes[i]);
            i += 1;
        }

        // REX prefix
        let mut rex = None;
        if bytes[i] & REX_BASE_MASK == REX_BASE {
            rex = Some(Rex::deserialize(bytes[i]));
            i += 1;
        }

        // Opcode
        let opcode = bytes[i];
        i += 1;

        // LATER(martin-t) Decode lowest 3 bits of some instrucions as register.
        let mut encoding = Encoding::prefixes_opcode(legacy_prefixes, rex, opcode);
        match opcode {
            0x01 | 0x03 => {
                // AddRR, AddMR, AddRM
                Self::deserialize_modrm(bytes, &mut i, &mut encoding);
            }
            0x09 | 0x0B => {
                // OrRR, OrMR, OrRM
                Self::deserialize_modrm(bytes, &mut i, &mut encoding);
            }
            0x0F => {
                // 2-byte opcode
                let opcode2 = bytes[i];
                i += 1;
                encoding.opcode2 = Some(opcode2);

                match opcode2 {
                    0x0B => {
                        // Ud2
                    }
                    0x84 | 0x8F | 0x8D | 0x8C | 0x8E | 0x85 => {
                        // Jcc - 32 bit offset
                        Self::deserialize_num(bytes, &mut i, &mut encoding, 4)
                    }
                    0xAF => {
                        // ImulRR, ImulRM
                        Self::deserialize_modrm(bytes, &mut i, &mut encoding);
                    }
                    _ => panic!("opcode {opcode:02x}, unknown opcode2 {opcode2:02x}"),
                }
            }
            0x21 | 0x23 => {
                // AndRR, AndMR, AndRM
                Self::deserialize_modrm(bytes, &mut i, &mut encoding);
            }
            0x29 | 0x2B => {
                // SubRR, SubMR, SubRM
                Self::deserialize_modrm(bytes, &mut i, &mut encoding);
            }
            0x39 | 0x3B => {
                // CmpRR, CmpMR, CmpRM
                Self::deserialize_modrm(bytes, &mut i, &mut encoding);
            }
            0x50..=0x57 => {
                // PushR
            }
            0x58..=0x5F => {
                // PopR
            }
            0x74 | 0x7F | 0x7D | 0x7C | 0x7E | 0x75 => {
                // Jcc - 8 bit offset
                Self::deserialize_num(bytes, &mut i, &mut encoding, 1)
            }
            0x81 => {
                // AddRI, AddMI
                // AndRI, AndMI
                // CmpRI, CmpMI
                // OrRI, OrMI
                // SubRI, SubMI
                Self::deserialize_modrm(bytes, &mut i, &mut encoding);
                Self::deserialize_num(bytes, &mut i, &mut encoding, 4);
            }
            0x83 => {
                // AddRI, AddMI
                // AndRI, AndMI
                // CmpRI, CmpMI
                // OrRI, OrMI
                // SubRI, SubMI
                Self::deserialize_modrm(bytes, &mut i, &mut encoding);
                Self::deserialize_num(bytes, &mut i, &mut encoding, 1);
            }
            0x85 => {
                // TestRR, TestMR
                Self::deserialize_modrm(bytes, &mut i, &mut encoding);
            }
            0x89 | 0x8b => {
                // MovRR, MovMR, MovRM
                Self::deserialize_modrm(bytes, &mut i, &mut encoding);
            }
            0x8f => {
                Self::deserialize_modrm(bytes, &mut i, &mut encoding);
                let extension = encoding.modrm.unwrap().reg;
                match extension{
                    0 => {
                        // PopM
                    }
                    _ => panic!("opcode {opcode:02x}, unknown extension {extension}"),
                }
            }
            0x90 => {
                // Nop
            }
            0x99 => {
                // Cdq, Cqo
            }
            0xB8..=0xBF => {
                // MovRI
                let mut size = 4;
                if let Some(rex) = rex {
                    if rex.w == 1 {
                        size = 8;
                    }
                }
                Self::deserialize_num(bytes, &mut i, &mut encoding, size);
            }
            0xC3 => {
                // Ret
            }
            0xC7 => {
                // MovRI
                // LATER(martin-t) Modrm.reg is opcode extension (0 for MOV)
                Self::deserialize_modrm(bytes, &mut i, &mut encoding);
                Self::deserialize_num(bytes, &mut i, &mut encoding, 4);
            }
            0xCC => {
                // Int3
            }
            0xE8 => {
                // CallRel
                Self::deserialize_num(bytes, &mut i, &mut encoding, 4);
            }
            0xE9 => {
                // JmpRel 32 bit offset
                Self::deserialize_num(bytes, &mut i, &mut encoding, 4);
            }
            0xEB => {
                // JmpRel 8 bit offset
                Self::deserialize_num(bytes, &mut i, &mut encoding, 1);
            }
            0xF1 => {
                // Int1
            }
            0xF4 => {
                // Hlt
            }
            0xF7 => {
                Self::deserialize_modrm(bytes, &mut i, &mut encoding);
                let extension = encoding.modrm.unwrap().reg;
                match extension {
                    0 => {
                        // TestRI, TestMI
                        Self::deserialize_num(bytes, &mut i, &mut encoding, 4);
                    }
                    5 => {
                        // ImulR, ImulM
                    }
                    7 => {
                        // IdivR, IdivM
                    }
                    _ => panic!("opcode {opcode:02x}, unknown extension {extension}"),
                }
            }
            0xFF => {
                Self::deserialize_modrm(bytes, &mut i, &mut encoding);
                let extension = encoding.modrm.unwrap().reg;
                match extension{
                    2 => {
                        // CallAbs
                    }
                    6 => {
                        // PushM
                    }
                    _ => panic!("opcode {opcode:02x}, unknown extension {extension}"),
                }
            }
            _ => panic!("unknown opcode {:02x}", opcode),
        }

        (encoding, i)
    }

    fn deserialize_modrm(bytes: &[u8], index: &mut usize, encoding: &mut Encoding) {
        // For debugging - prevent a panic when reading past the end of bytes.
        // let mut bytes = bytes.to_vec();
        // bytes.extend_from_slice(&[0, 0, 0, 0]);
        // let bytes = &bytes;

        let mut i = *index;

        let modrm = ModRm::deserialize(bytes[i]);
        i += 1;
        encoding.modrm = Some(modrm);

        if modrm.mod_ != MODRM_MOD_DIRECT && modrm.rm == MODRM_RM_USE_SIB {
            encoding.sib = Some(Sib::deserialize(bytes[i]));
            i += 1;
        }

        match modrm.mod_ {
            MODRM_MOD_NO_DISP_OR_NO_BASE => {
                if (modrm.rm == MODRM_RM_USE_SIB && encoding.sib.unwrap().base == SIB_BASE_SPECIAL)
                    || modrm.rm == MODRM_RM_DISP32
                {
                    let size = 4;
                    encoding.disp = Some(Num::deserialize(&bytes[i..i + size]));
                    i += size;
                }
            }
            MODRM_MOD_DISP8 => {
                let size = 1;
                encoding.disp = Some(Num::deserialize(&bytes[i..i + size]));
                i += size;
            }
            MODRM_MOD_DISP32 => {
                let size = 4;
                encoding.disp = Some(Num::deserialize(&bytes[i..i + size]));
                i += size;
            }
            MODRM_MOD_DIRECT => {}
            _ => unreachable!(),
        }

        *index = i;
    }

    fn deserialize_num(bytes: &[u8], index: &mut usize, encoding: &mut Encoding, size: usize) {
        encoding.imm = Some(Num::deserialize(&bytes[*index..*index + size]));
        *index += size;
    }

    pub fn deserialize_and_print(bytes: &[u8]) -> (Encoding, usize) {
        let (encoding, consumed) = Self::deserialize(bytes);
        print_asm(&bytes[..consumed]);
        println!("{encoding}");
        (encoding, consumed)
    }

    #[allow(dead_code)] // LATER(martin-t) Might be nice to expose this in the CLI
    pub fn deserialize_and_print_all(mut bytes: &[u8]) {
        while !bytes.is_empty() {
            let (_, consumed) = Self::deserialize_and_print(bytes);
            bytes = &bytes[consumed..];
        }
    }
}

impl Default for Encoding {
    fn default() -> Self {
        Encoding {
            legacy_prefixes: SmallVec::new(),
            rex: None,
            opcode: 0x90,
            opcode2: None,
            modrm: None,
            sib: None,
            disp: None,
            imm: None,
        }
    }
}

#[allow(clippy::print_in_format_impl)] // Intentional so we notice it
impl Display for Encoding {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if !self.legacy_prefixes.is_empty() {
            writeln!(f, "legacy prefixes:")?;
            for &prefix in &self.legacy_prefixes {
                write!(f, "    0x{:02x}: ", prefix)?;
                match prefix {
                    0xf0 => writeln!(f, "LOCK")?,
                    0xf2 => writeln!(f, "REPNE/REPNZ")?,
                    0xf3 => writeln!(f, "REP/REPE/REPZ")?,
                    0x2e => writeln!(f, "CS segment override / branch not taken")?,
                    0x36 => writeln!(f, "SS segment override")?,
                    0x3e => writeln!(f, "DS segment override / branch taken")?,
                    0x26 => writeln!(f, "ES segment override")?,
                    0x64 => writeln!(f, "FS segment override")?,
                    0x65 => writeln!(f, "GS segment override")?,
                    0x66 => writeln!(f, "operand-size override")?,
                    0x67 => writeln!(f, "address-size override")?,
                    _ => unreachable!(),
                };
            }
        }
        if let Some(rex) = &self.rex {
            writeln!(f, "{rex}")?;
        }
        write!(f, "opcode 0x{:02x}: ", self.opcode)?;
        // Print relevant rows from the Instruction column
        match self.opcode {
            0x01 => writeln!(f, "add r/m32, r32 | r/m64, r64")?,
            0x03 => writeln!(f, "add r32, r/m32 | r64, r/m64")?,
            0x09 => writeln!(f, "or r/m32, r32 | r/m64, r64")?,
            0x0f => match self.opcode2.unwrap() {
                0x0b => writeln!(f, "ud2")?,
                0x84 => writeln!(f, "je rel32")?,
                0x8F => writeln!(f, "jg rel32")?,
                0x8D => writeln!(f, "jge rel32")?,
                0x8C => writeln!(f, "jl rel32")?,
                0x8E => writeln!(f, "jle rel32")?,
                0x85 => writeln!(f, "jne rel32")?,
                0xaf => writeln!(f, "imul r32, r/m32 | r64, r/m64")?,
                opcode2 => eprintln!("OPCODE {:02x}, opcode2 {:02x}: unknown/todo", self.opcode, opcode2),
            },
            0x0b => writeln!(f, "or r32, r/m32 | r64, r/m64")?,
            0x29 => writeln!(f, "sub r/m32, r32 | r/m64, r64")?,
            0x2b => writeln!(f, "sub r32, r/m32 | r64, r/m64")?,
            0x39 => writeln!(f, "cmp r/m32, r32 | r/m64, r64")?,
            0x3b => writeln!(f, "cmp r32, r/m32 | r64, r/m64")?,
            0x21 => writeln!(f, "and r/m32, r32 | r/m64, r64")?,
            0x23 => writeln!(f, "and r32, r/m32 | r64, r/m64")?,
            0x50..=0x57 => writeln!(f, "push r64")?,
            0x58..=0x5F => writeln!(f, "pop r64")?,
            0x74 => writeln!(f, "je rel8")?,
            0x7F => writeln!(f, "jg rel8")?,
            0x7D => writeln!(f, "jge rel8")?,
            0x7C => writeln!(f, "jl rel8")?,
            0x7E => writeln!(f, "jle rel8")?,
            0x75 => writeln!(f, "jne rel8")?,
            0x81 => match self.modrm.unwrap().reg {
                0 => writeln!(f, "add r/m32, imm32 | r/m64, imm32")?,
                1 => writeln!(f, "or r/m32, imm32 | r/m64, imm32")?,
                4 => writeln!(f, "and r/m32, imm32 | r/m64, imm32")?,
                5 => writeln!(f, "sub r/m32, imm32 | r/m64, imm32")?,
                7 => writeln!(f, "cmp r/m32, imm32 | r/m64, imm32")?,
                ext => eprintln!("OPCODE {:02x}, EXTENSION {ext}: unknown/todo", self.opcode),
            },
            0x85 => writeln!(f, "test r/m32, r32 | r/m64, r64")?,
            0x89 => writeln!(f, "mov r/m32, r32 | r/m64, r64")?,
            0x8b => writeln!(f, "mov r32, r/m32 | r64, r/m64")?,
            0x90 => writeln!(f, "nop")?,
            0x99 => writeln!(f, "cdq | cqo")?,
            0xb8..=0xbf => writeln!(f, "mov r32, imm32 | r64, imm32")?,
            0xc3 => writeln!(f, "ret")?,
            0xc7 => writeln!(f, "mov r/m32, imm32 | r/m64, imm32")?,
            0xcc => writeln!(f, "int3")?,
            0xe8 => writeln!(f, "call rel32")?,
            0xe9 => writeln!(f, "jmp rel32")?,
            0xeb => writeln!(f, "jmp rel8")?,
            0xc1 => writeln!(f, "int1")?,
            0xf4 => writeln!(f, "hlt")?,
            0xf7 => match self.modrm.unwrap().reg {
                0 => writeln!(f, "test r/32, imm32 | r/m64, imm32")?,
                5 => writeln!(f, "imul r/m32 | r/m64")?,
                7 => writeln!(f, "idiv r/m32 | r/m64")?,
                ext => eprintln!("OPCODE {:02x}, EXTENSION {ext}: unknown/todo", self.opcode),
            },
            0xff => writeln!(f, "call r/m64")?,
            _ => eprintln!("OPCODE 0x{:02x}: unknown/todo", self.opcode),
        }
        if let Some(modrm) = &self.modrm {
            writeln!(f, "{modrm}")?;
        }
        if let Some(sib) = &self.sib {
            writeln!(f, "{sib}")?;
        }
        if let Some(disp) = &self.disp {
            writeln!(f, "disp = {disp}")?;
        }
        if let Some(imm) = &self.imm {
            writeln!(f, "imm = {imm}")?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Compiled {
    pub code: Vec<u8>,
    pub label_offsets: FnvHashMap<usize, usize>,
}

pub fn compile(instrs: &[Instr]) -> Compiled {
    use Instr::*;

    // LATER(martin-t) Optimize jumps to use 2 byte encoding where possible.
    // Ideas:
    //  - do backwards jumps first
    //  - calculate min and max length in bytes, only jumps between them
    //    are undecided, the rest has to be 2 or 6 bytes

    let mut code = Vec::new();
    let mut label_offsets = FnvHashMap::default();
    // Only 32 bit replacements here, 8 bit require different handling
    let mut replace32 = Vec::new();

    // println!("Compiling instrs:");
    // for &instr in instrs {
    //     println!("{:x?}", instr);
    // }

    for &instr in instrs {
        match instr {
            Label(label) => {
                let prev = label_offsets.insert(label, code.len());
                assert_eq!(prev, None);
            }
            CallLabel(label) => {
                CallRel32(0).encode().serialize(&mut code);
                replace32.push((code.len(), label));
            }
            JeLabel(label) => {
                Je32(0).encode().serialize(&mut code);
                replace32.push((code.len(), label));
            }
            JgLabel(label) => {
                Jg32(0).encode().serialize(&mut code);
                replace32.push((code.len(), label));
            }
            JgeLabel(label) => {
                Jge32(0).encode().serialize(&mut code);
                replace32.push((code.len(), label));
            }
            JlLabel(label) => {
                Jl32(0).encode().serialize(&mut code);
                replace32.push((code.len(), label));
            }
            JleLabel(label) => {
                Jle32(0).encode().serialize(&mut code);
                replace32.push((code.len(), label));
            }
            JneLabel(label) => {
                Jne32(0).encode().serialize(&mut code);
                replace32.push((code.len(), label));
            }
            JmpLabel(label) => {
                JmpRel32(0).encode().serialize(&mut code);
                replace32.push((code.len(), label));
            }
            _ => instr.encode().serialize(&mut code),
        }
    }
    //print_hex(&code);

    for (offset, label) in replace32 {
        let label_offset = label_offsets[&label];

        let rel = (label_offset as i64 - offset as i64) as i32;
        let imm = rel.to_le_bytes();

        let dest = &mut code[offset - 4..offset];
        assert_eq!(dest, &[0, 0, 0, 0]);
        dest.copy_from_slice(&imm);
    }
    //print_hex(&code);

    Compiled { code, label_offsets }
}

#[allow(dead_code)]
pub fn print_asm(bytes: &[u8]) {
    println!("Asm as bytes: {}", fmt_bytes(bytes));
    println!("Asm as hex:   {}", fmt_hex(bytes));
}

#[allow(dead_code)]
pub fn eprint_asm(bytes: &[u8]) {
    eprintln!("Asm as bytes: {}", fmt_bytes(bytes));
    eprintln!("Asm as hex:   {}", fmt_hex(bytes));
}

pub fn fmt_bytes(bytes: &[u8]) -> String {
    itertools::free::join(bytes.iter().map(|byte| format!("0x{:02x}", byte)), ", ")
}

pub fn fmt_hex(bytes: &[u8]) -> String {
    itertools::free::join(bytes.iter().map(|byte| format!("{:02x}", byte)), " ")
}
