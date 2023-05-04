//! Data structures for representing x86-64 assembly.

use std::{
    fmt::{self, Display, Formatter, LowerHex},
    iter,
};

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Instr {
    Nop,

    AddRR(Reg, Reg),
    AddMR(Mem, Reg),
    AddRM(Reg, Mem),
    AddRI(Reg, i32),
    AddMI(Mem, i32),

    AndRR(Reg, Reg),
    AndMR(Mem, Reg),
    AndRM(Reg, Mem),
    AndRI(Reg, i32),
    AndMI(Mem, i32),

    /// Offset is relative to the start of this instruction.
    _CallRel(i32),
    /// Offset is relative to the start of the next instruction.
    CallRel32(i32),
    CallAbsR(Reg),
    CallAbsM(Mem),

    CmpRR(Reg, Reg),
    CmpMR(Mem, Reg),
    CmpRM(Reg, Mem),
    CmpRI(Reg, i32),
    CmpMI(Mem, i32),

    Cdq,
    Cqo,

    Hlt,

    IdivR(Reg),
    IdivM(Mem),

    ImulR(Reg),
    ImulM(Mem),
    ImulRR(Reg, Reg),
    ImulRM(Reg, Mem),

    Int3,
    Int1,

    // Jcc
    // We should only need these 6 because we only use signed integers.
    // Also these are all relative, there's no absolute variant.

    // Offset is relative to the start of this instruction.
    _Je(i32),
    _Jg(i32),
    _Jge(i32),
    _Jl(i32),
    _Jle(i32),
    _Jne(i32),

    // Offset is relative to the start of the next instruction.
    Je32(i32),
    Jg32(i32),
    Jge32(i32),
    Jl32(i32),
    Jle32(i32),
    Jne32(i32),

    /// Offset is relative to the start of this instruction.
    _JmpRel(i32),
    /// Offset is relative to the start of the next instruction.
    JmpRel32(i32),

    MovRR(Reg, Reg),
    MovMR(Mem, Reg),
    MovRM(Reg, Mem),
    MovRI(Reg, i64),
    MovMI(Mem, i32),

    OrRR(Reg, Reg),
    OrMR(Mem, Reg),
    OrRM(Reg, Mem),
    OrRI(Reg, i32),
    OrMI(Mem, i32),

    Pop(Reg),
    Push(Reg),

    Ret,

    SubRR(Reg, Reg),
    SubMR(Mem, Reg),
    SubRM(Reg, Mem),
    SubRI(Reg, i32),
    SubMI(Mem, i32),

    TestRR(Reg, Reg),
    TestMR(Mem, Reg),
    // There is no TestRM in x86
    TestRI(Reg, i32),
    TestMI(Mem, i32),

    Ud2,

    // Virtual instructions
    Label(usize),
    CallLabel(usize),
    JeLabel(usize),
    JgLabel(usize),
    JgeLabel(usize),
    JlLabel(usize),
    JleLabel(usize),
    JneLabel(usize),
    JmpLabel(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reg {
    Eax,
    Ecx,
    Edx,
    Ebx,
    Esp,
    Ebp,
    Esi,
    Edi,
    R8d,
    R9d,
    R10d,
    R11d,
    R12d,
    R13d,
    R14d,
    R15d,
    Rax,
    Rcx,
    Rdx,
    Rbx,
    Rsp,
    Rbp,
    Rsi,
    Rdi,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
}

/// https://blog.yossarian.net/2020/06/13/How-x86_64-addresses-memory
///
/// LATER(martin-t) Add a way to specify size (dword/qword).
///     We currently have no way to use Instr::*MI with 64 bit memory locations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Mem {
    pub base: Option<Reg>,
    pub index: Option<Reg>,
    pub scale: u8,
    // LATER(martin-t) This can be 64 bits in some special cases (using A* registers?)
    /// Displacement (offset)
    pub disp: i32,
}

/// https://wiki.osdev.org/X86-64_Instruction_Encoding#REX_prefix
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rex {
    /// When 1, a 64-bit operand size is used.
    pub w: u8,
    /// This 1-bit value is an extension to the MODRM.reg field.
    pub r: u8,
    /// This 1-bit value is an extension to the SIB.index field.
    pub x: u8,
    /// This 1-bit value is an extension to the MODRM.rm field,
    /// SIB.base field or Opcode reg field.
    pub b: u8,
}

/// modRM - page 525 and 604
///  - mod - 2 bits
///  - reg - 3 bits
///  - rm - 3 bits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModRm {
    /// Used along with `rm` to encode the addressing mode.
    pub mod_: u8,
    /// Register number or opcode extension.
    pub reg: u8,
    /// Addressing mode together with `mod_` or register number.
    pub rm: u8,
}

/// SIB - page 525
/// - scale - 2 bits
/// - index - 3 bits
/// - base - 3 bits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sib {
    /// Possible values: 1, 2, 4, 8
    pub scale: u8,
    /// Register multiplied by `scale`.
    pub index: u8,
    /// Base register.
    pub base: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Num {
    Num8(i8),
    Num32(i32),
    Num64(i64),
}

/// 0x40
pub const REX_BASE: u8 = 0b0100_0000;
pub const REX_BASE_MASK: u8 = 0b1111_0000;

/// See note under the SIB table
pub const MODRM_MOD_NO_DISP_OR_NO_BASE: u8 = 0b00;
pub const MODRM_MOD_DISP8: u8 = 0b01;
pub const MODRM_MOD_DISP32: u8 = 0b10;
pub const MODRM_MOD_DIRECT: u8 = 0b11;

/// This is `[--][--]` in the ModR/M table.
/// If mod != 0b11 then r/m 0b100 means a SIB byte is used.
///
/// Same value as ESP/RSP.
pub const MODRM_RM_USE_SIB: u8 = 0b100;

/// This is `disp32` in the ModR/M table
/// If mod == 0b00 then there is no index or base and disp is 32 bits.
///
/// Same value as EBP/RBP.
pub const MODRM_RM_DISP32: u8 = 0b101;

/// This is `none` in the SIB table.
/// The index register is not present.
///
/// Same value as RSP/R12.
///
/// There seems to be no way to use these registers as index,
/// NASM on asm.x32.dev silently ignores instructions which do.
pub const SIB_INDEX_NONE: u8 = 0b100;

/// This is `[*]` in the SIB table.
/// If mod == 0b00 then there is no base register and disp is 32 bits.
///
/// Same value as RBP/R13.
pub const SIB_BASE_SPECIAL: u8 = 0b101;

impl Reg {
    pub fn encode(self) -> (u8, u8) {
        match self {
            Reg::Eax => (0, 0b000),
            Reg::Ecx => (0, 0b001),
            Reg::Edx => (0, 0b010),
            Reg::Ebx => (0, 0b011),
            Reg::Esp => (0, 0b100),
            Reg::Ebp => (0, 0b101),
            Reg::Esi => (0, 0b110),
            Reg::Edi => (0, 0b111),
            Reg::R8d => (1, 0b000),
            Reg::R9d => (1, 0b001),
            Reg::R10d => (1, 0b010),
            Reg::R11d => (1, 0b011),
            Reg::R12d => (1, 0b100),
            Reg::R13d => (1, 0b101),
            Reg::R14d => (1, 0b110),
            Reg::R15d => (1, 0b111),
            Reg::Rax => (0, 0b000),
            Reg::Rcx => (0, 0b001),
            Reg::Rdx => (0, 0b010),
            Reg::Rbx => (0, 0b011),
            Reg::Rsp => (0, 0b100),
            Reg::Rbp => (0, 0b101),
            Reg::Rsi => (0, 0b110),
            Reg::Rdi => (0, 0b111),
            Reg::R8 => (1, 0b000),
            Reg::R9 => (1, 0b001),
            Reg::R10 => (1, 0b010),
            Reg::R11 => (1, 0b011),
            Reg::R12 => (1, 0b100),
            Reg::R13 => (1, 0b101),
            Reg::R14 => (1, 0b110),
            Reg::R15 => (1, 0b111),
        }
    }

    pub fn decode(wide: bool, bit: u8, bits: u8) -> Self {
        if !wide {
            assert!(bit == 0);
            match (bit, bits) {
                (0, 0b000) => Reg::Eax,
                (0, 0b001) => Reg::Ecx,
                (0, 0b010) => Reg::Edx,
                (0, 0b011) => Reg::Ebx,
                (0, 0b100) => Reg::Esp,
                (0, 0b101) => Reg::Ebp,
                (0, 0b110) => Reg::Esi,
                (0, 0b111) => Reg::Edi,
                (1, 0b000) => Reg::R8d,
                (1, 0b001) => Reg::R9d,
                (1, 0b010) => Reg::R10d,
                (1, 0b011) => Reg::R11d,
                (1, 0b100) => Reg::R12d,
                (1, 0b101) => Reg::R13d,
                (1, 0b110) => Reg::R14d,
                (1, 0b111) => Reg::R15d,
                _ => panic!("Invalid bits"),
            }
        } else {
            match (bit, bits) {
                (0, 0b000) => Reg::Rax,
                (0, 0b001) => Reg::Rcx,
                (0, 0b010) => Reg::Rdx,
                (0, 0b011) => Reg::Rbx,
                (0, 0b100) => Reg::Rsp,
                (0, 0b101) => Reg::Rbp,
                (0, 0b110) => Reg::Rsi,
                (0, 0b111) => Reg::Rdi,
                (1, 0b000) => Reg::R8,
                (1, 0b001) => Reg::R9,
                (1, 0b010) => Reg::R10,
                (1, 0b011) => Reg::R11,
                (1, 0b100) => Reg::R12,
                (1, 0b101) => Reg::R13,
                (1, 0b110) => Reg::R14,
                (1, 0b111) => Reg::R15,
                _ => panic!("Invalid bits"),
            }
        }
    }

    #[allow(dead_code)]
    pub fn is_32bit(self) -> bool {
        match self {
            Reg::Eax => true,
            Reg::Ecx => true,
            Reg::Edx => true,
            Reg::Ebx => true,
            Reg::Esp => true,
            Reg::Ebp => true,
            Reg::Esi => true,
            Reg::Edi => true,
            Reg::R8d => true,
            Reg::R9d => true,
            Reg::R10d => true,
            Reg::R11d => true,
            Reg::R12d => true,
            Reg::R13d => true,
            Reg::R14d => true,
            Reg::R15d => true,
            Reg::Rax => false,
            Reg::Rcx => false,
            Reg::Rdx => false,
            Reg::Rbx => false,
            Reg::Rsp => false,
            Reg::Rbp => false,
            Reg::Rsi => false,
            Reg::Rdi => false,
            Reg::R8 => false,
            Reg::R9 => false,
            Reg::R10 => false,
            Reg::R11 => false,
            Reg::R12 => false,
            Reg::R13 => false,
            Reg::R14 => false,
            Reg::R15 => false,
        }
    }

    pub fn is_64bit(self) -> bool {
        !self.is_32bit()
    }

    /// Page 530 - Table 2-2. 32-Bit Addressing Forms with the ModR/M Byte
    /// Page 531 - Table 2-3. 32-Bit Addressing Forms with the SIB Byte
    ///
    /// Note that we don't know here whether 32 or 64 bit registers are used
    /// so we always print 32 bit register names.
    pub fn to_meaning(self) -> String {
        if self.encode().1 == 0b100 {
            format!("{} / [--][--]", self)
        } else if self.encode().1 == 0b101 {
            format!("{} / disp32 / [*]", self)
        } else {
            format!("{}", self)
        }
    }

    #[allow(dead_code)]
    pub fn variants() -> impl Iterator<Item = Reg> {
        [
            Reg::Eax,
            Reg::Ecx,
            Reg::Edx,
            Reg::Ebx,
            Reg::Esp,
            Reg::Ebp,
            Reg::Esi,
            Reg::Edi,
            Reg::R8d,
            Reg::R9d,
            Reg::R10d,
            Reg::R11d,
            Reg::R12d,
            Reg::R13d,
            Reg::R14d,
            Reg::R15d,
            Reg::Rax,
            Reg::Rcx,
            Reg::Rdx,
            Reg::Rbx,
            Reg::Rsp,
            Reg::Rbp,
            Reg::Rsi,
            Reg::Rdi,
            Reg::R8,
            Reg::R9,
            Reg::R10,
            Reg::R11,
            Reg::R12,
            Reg::R13,
            Reg::R14,
            Reg::R15,
        ]
        .iter()
        .copied()
    }

    #[allow(dead_code)]
    pub fn none_and_variants() -> impl Iterator<Item = Option<Reg>> {
        iter::once(None).chain(Reg::variants().map(Some))
    }
}

impl Mem {
    #[allow(dead_code)]
    pub fn base(reg: Reg) -> Mem {
        Mem {
            base: Some(reg),
            ..Default::default()
        }
    }

    #[allow(dead_code)]
    pub fn base_offset(reg: Reg, disp: i32) -> Mem {
        Mem {
            base: Some(reg),
            disp,
            ..Default::default()
        }
    }

    #[allow(dead_code)]
    pub fn index(reg: Reg, scale: u8) -> Mem {
        Mem {
            index: Some(reg),
            scale,
            ..Default::default()
        }
    }

    #[allow(dead_code)]
    pub fn index_offset(reg: Reg, scale: u8, disp: i32) -> Mem {
        Mem {
            index: Some(reg),
            scale,
            disp,
            ..Default::default()
        }
    }

    #[allow(dead_code)]
    pub fn base_index(base: Reg, index: Reg, scale: u8) -> Mem {
        Mem {
            base: Some(base),
            index: Some(index),
            scale,
            ..Default::default()
        }
    }

    #[allow(dead_code)]
    pub fn base_index_offset(base: Reg, index: Reg, scale: u8, disp: i32) -> Mem {
        Mem {
            base: Some(base),
            index: Some(index),
            scale,
            disp,
        }
    }
}

impl Default for Mem {
    fn default() -> Self {
        Self {
            base: None,
            index: None,
            scale: 1,
            disp: 0,
        }
    }
}

impl Rex {
    pub fn serialize(self) -> u8 {
        let w = self.w;
        let r = self.r;
        let x = self.x;
        let b = self.b;
        REX_BASE | (w << 3) | (r << 2) | (x << 1) | b
    }

    pub fn deserialize(byte: u8) -> Self {
        assert_eq!(byte & REX_BASE_MASK, REX_BASE);
        let w = (byte >> 3) & 1;
        let r = (byte >> 2) & 1;
        let x = (byte >> 1) & 1;
        let b = byte & 1;
        Self { w, r, x, b }
    }

    pub fn is_used(self) -> bool {
        self.w == 1 || self.r == 1 || self.x == 1 || self.b == 1
    }
}

impl ModRm {
    pub fn serialize(self) -> u8 {
        assert!(self.mod_ <= 0b11);
        assert!(self.reg <= 0b111);
        assert!(self.rm <= 0b111);
        (self.mod_ << 6) | (self.reg << 3) | self.rm
    }

    pub fn deserialize(byte: u8) -> Self {
        let mod_ = (byte & 0b1100_0000) >> 6;
        let reg = (byte & 0b0011_1000) >> 3;
        let rm = byte & 0b0000_0111;
        Self { mod_, reg, rm }
    }
}

impl Sib {
    pub fn serialize(self) -> u8 {
        let scale_bits = self.scale_bits();
        assert!(self.index <= 0b111);
        assert!(self.base <= 0b111);
        (scale_bits << 6) | (self.index << 3) | self.base
    }

    pub fn deserialize(byte: u8) -> Self {
        let scale_bits = (byte & 0b1100_0000) >> 6;
        let scale = Self::bits_to_scale(scale_bits);
        let index = (byte & 0b0011_1000) >> 3;
        let base = byte & 0b0000_0111;
        Self { scale, index, base }
    }

    pub fn scale_bits(&self) -> u8 {
        match self.scale {
            1 => 0b00,
            2 => 0b01,
            4 => 0b10,
            8 => 0b11,
            _ => panic!("Invalid scale"),
        }
    }

    fn bits_to_scale(bits: u8) -> u8 {
        match bits {
            0b00 => 1,
            0b01 => 2,
            0b10 => 4,
            0b11 => 8,
            _ => unreachable!(),
        }
    }
}

impl Num {
    pub fn serialize(self, buf: &mut Vec<u8>) {
        match self {
            Self::Num8(disp) => buf.extend_from_slice(&disp.to_le_bytes()),
            Self::Num32(disp) => buf.extend_from_slice(&disp.to_le_bytes()),
            Self::Num64(disp) => buf.extend_from_slice(&disp.to_le_bytes()),
        }
    }

    pub fn deserialize(bytes: &[u8]) -> Self {
        match bytes.len() {
            1 => Self::Num8(bytes[0] as i8),
            4 => Self::Num32(i32::from_le_bytes(bytes.try_into().unwrap())),
            8 => Self::Num64(i64::from_le_bytes(bytes.try_into().unwrap())),
            _ => unreachable!(),
        }
    }
}

/// Hack because I can't do `impl Option<Num> {}`.
pub trait NumberExt {
    fn to_32_bits(self) -> Self;
}

impl NumberExt for Option<Num> {
    fn to_32_bits(self) -> Self {
        match self {
            None => Some(Num::Num32(0)),
            Some(Num::Num8(num)) => Some(Num::Num32(num.into())),
            Some(Num::Num32(_)) => self,
            Some(Num::Num64(_)) => panic!("Not meant to be used on 64 bits"),
        }
    }
}

/// Format instructions so that https://disasm.pro/ can parse them
/// except some instructions which are only formatted for humans. LATER
///
/// Print numbers in decimal.
/// LATER(martin-t) Always specify size (dword/qword) and use "ptr".
impl Display for Instr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Instr::Nop => write!(f, "nop"),
            Instr::AddRR(dst, src) => write!(f, "add {dst}, {src}"),
            Instr::AddMR(dst, src) => write!(f, "add {dst}, {src}"),
            Instr::AddRM(dst, src) => write!(f, "add {dst}, {src}"),
            Instr::AddRI(dst, imm) => write!(f, "add {dst}, {imm}"),
            Instr::AddMI(dst, imm) => write!(f, "add {dst}, {imm}"),
            Instr::AndRR(dst, src) => write!(f, "and {dst}, {src}"),
            Instr::AndMR(dst, src) => write!(f, "and {dst}, {src}"),
            Instr::AndRM(dst, src) => write!(f, "and {dst}, {src}"),
            Instr::AndRI(dst, imm) => write!(f, "and {dst}, {imm}"),
            Instr::AndMI(dst, imm) => write!(f, "and {dst}, {imm}"),
            Instr::_CallRel(rel) => write!(f, "call {rel}"),
            Instr::CallRel32(rel) => write!(f, "call32 {rel}"),
            Instr::CallAbsR(abs) => write!(f, "call {abs}"),
            Instr::CallAbsM(abs) => write!(f, "call {abs}"),
            Instr::CmpRR(dst, src) => write!(f, "cmp {dst}, {src}"),
            Instr::CmpMR(dst, src) => write!(f, "cmp {dst}, {src}"),
            Instr::CmpRM(dst, src) => write!(f, "cmp {dst}, {src}"),
            Instr::CmpRI(dst, imm) => write!(f, "cmp {dst}, {imm}"),
            Instr::CmpMI(dst, imm) => write!(f, "cmp {dst}, {imm}"),
            Instr::Cdq => write!(f, "cdq"),
            Instr::Cqo => write!(f, "cqo"),
            Instr::Hlt => write!(f, "hlt"),
            Instr::IdivR(op) => write!(f, "idiv {op}"),
            Instr::IdivM(op) => write!(f, "idiv dword ptr {op}"),
            Instr::ImulR(op) => write!(f, "imul {op}"),
            Instr::ImulM(op) => write!(f, "imul dword ptr {op}"),
            Instr::ImulRR(dst, src) => write!(f, "imul {dst}, {src}"),
            Instr::ImulRM(dst, src) => write!(f, "imul {dst}, {src}"),
            Instr::Int3 => write!(f, "int3"),
            Instr::Int1 => write!(f, "int1"),
            Instr::_Je(rel) => write!(f, "je {rel}"),
            Instr::_Jg(rel) => write!(f, "jg {rel}"),
            Instr::_Jge(rel) => write!(f, "jge {rel}"),
            Instr::_Jl(rel) => write!(f, "jl {rel}"),
            Instr::_Jle(rel) => write!(f, "jle {rel}"),
            Instr::_Jne(rel) => write!(f, "jne {rel}"),
            Instr::Je32(rel) => write!(f, "je32 {rel}"),
            Instr::Jg32(rel) => write!(f, "jg32 {rel}"),
            Instr::Jge32(rel) => write!(f, "jge32 {rel}"),
            Instr::Jl32(rel) => write!(f, "jl32 {rel}"),
            Instr::Jle32(rel) => write!(f, "jle32 {rel}"),
            Instr::Jne32(rel) => write!(f, "jne32 {rel}"),
            Instr::_JmpRel(rel) => write!(f, "jmp {rel}"),
            Instr::JmpRel32(rel) => write!(f, "jmp32 {rel}"),
            Instr::MovRR(dst, src) => write!(f, "mov {dst}, {src}"),
            Instr::MovMR(dst, src) => write!(f, "mov {dst}, {src}"),
            Instr::MovRM(dst, src) => write!(f, "mov {dst}, {src}"),
            Instr::MovRI(dst, imm) => write!(f, "mov {dst}, {imm}"),
            Instr::MovMI(dst, imm) => write!(f, "mov dword ptr {dst}, {imm}"),
            Instr::OrRR(dst, src) => write!(f, "or {dst}, {src}"),
            Instr::OrMR(dst, src) => write!(f, "or {dst}, {src}"),
            Instr::OrRM(dst, src) => write!(f, "or {dst}, {src}"),
            Instr::OrRI(dst, imm) => write!(f, "or {dst}, {imm}"),
            Instr::OrMI(dst, imm) => write!(f, "or {dst}, {imm}"),
            Instr::Pop(op) => write!(f, "pop {op}"),
            Instr::Push(op) => write!(f, "push {op}"),
            Instr::Ret => write!(f, "ret"),
            Instr::SubRR(dst, src) => write!(f, "sub {dst}, {src}"),
            Instr::SubMR(dst, src) => write!(f, "sub {dst}, {src}"),
            Instr::SubRM(dst, src) => write!(f, "sub {dst}, {src}"),
            Instr::SubRI(dst, imm) => write!(f, "sub {dst}, {imm}"),
            Instr::SubMI(dst, imm) => write!(f, "sub {dst}, {imm}"),
            Instr::TestRR(dst, src) => write!(f, "test {dst}, {src}"),
            Instr::TestMR(dst, src) => write!(f, "test {dst}, {src}"),
            Instr::TestRI(dst, imm) => write!(f, "test {dst}, {imm}"),
            Instr::TestMI(dst, imm) => write!(f, "test {dst}, {imm}"),
            Instr::Ud2 => write!(f, "ud2"),

            Instr::Label(label) => write!(f, "Label {label}:"),
            Instr::CallLabel(label) => write!(f, "CallLabel {label}"),
            Instr::JeLabel(label) => write!(f, "JeLabel {label}"),
            Instr::JgLabel(label) => write!(f, "JgLabel {label}"),
            Instr::JgeLabel(label) => write!(f, "JgeLabel {label}"),
            Instr::JlLabel(label) => write!(f, "JlLabel {label}"),
            Instr::JleLabel(label) => write!(f, "JleLabel {label}"),
            Instr::JneLabel(label) => write!(f, "JneLabel {label}"),
            Instr::JmpLabel(label) => write!(f, "JmpLabel {label}"),
        }
    }
}

/// Format instructions so that NASM on https://asm.x32.dev/ can parse them.
///
/// Print numbers in hex because it interprets everything as hex.
/// Mem only needs to specify size (dword/qword)
/// if it can't be inferred from register size
/// and must not use "ptr".
///
/// Yes, this is abusing the difference between Display and LowerHex a little.
impl LowerHex for Instr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            Instr::Nop => write!(f, "nop"),
            Instr::AddRR(dst, src) => write!(f, "add {dst}, {src}"),
            Instr::AddMR(dst, src) => write!(f, "add {dst:x}, {src}"),
            Instr::AddRM(dst, src) => write!(f, "add {dst}, {src:x}"),
            Instr::AddRI(dst, imm) => write!(f, "add {dst}, {}", Hex(imm)),
            Instr::AddMI(dst, imm) => write!(f, "add dword {dst:x}, {}", Hex(imm)),
            Instr::AndRR(dst, src) => write!(f, "and {dst}, {src}"),
            Instr::AndMR(dst, src) => write!(f, "and {dst:x}, {src}"),
            Instr::AndRM(dst, src) => write!(f, "and {dst}, {src:x}"),
            Instr::AndRI(dst, imm) => write!(f, "and {dst}, {}", Hex(imm)),
            Instr::AndMI(dst, imm) => write!(f, "and dword {dst:x}, {}", Hex(imm)),
            Instr::_CallRel(rel) => write!(f, "call {}", Hex(rel)),
            Instr::CallRel32(rel) => write!(f, "call32 {}", Hex(rel)),
            Instr::CallAbsR(abs) => write!(f, "call {abs}"),
            Instr::CallAbsM(abs) => write!(f, "call {abs:x}"),
            Instr::CmpRR(dst, src) => write!(f, "cmp {dst}, {src}"),
            Instr::CmpMR(dst, src) => write!(f, "cmp {dst:x}, {src}"),
            Instr::CmpRM(dst, src) => write!(f, "cmp {dst}, {src:x}"),
            Instr::CmpRI(dst, imm) => write!(f, "cmp {dst}, {}", Hex(imm)),
            Instr::CmpMI(dst, imm) => write!(f, "cmp dword {dst:x}, {}", Hex(imm)),
            Instr::Cdq => write!(f, "cdq"),
            Instr::Cqo => write!(f, "cqo"),
            Instr::Hlt => write!(f, "hlt"),
            Instr::IdivR(op) => write!(f, "idiv {op}"),
            Instr::IdivM(op) => write!(f, "idiv dword {op:x}"),
            Instr::ImulR(op) => write!(f, "imul {op}"),
            Instr::ImulM(op) => write!(f, "imul dword {op:x}"),
            Instr::ImulRR(dst, src) => write!(f, "imul {dst}, {src}"),
            Instr::ImulRM(dst, src) => write!(f, "imul {dst}, {src:x}"),
            Instr::Int3 => write!(f, "int3"),
            Instr::Int1 => write!(f, "int1"),
            Instr::_Je(rel) => write!(f, "je {}", Hex(rel)),
            Instr::_Jg(rel) => write!(f, "jg {}", Hex(rel)),
            Instr::_Jge(rel) => write!(f, "jge {}", Hex(rel)),
            Instr::_Jl(rel) => write!(f, "jl {}", Hex(rel)),
            Instr::_Jle(rel) => write!(f, "jle {}", Hex(rel)),
            Instr::_Jne(rel) => write!(f, "jne {}", Hex(rel)),
            Instr::Je32(rel) => write!(f, "je32 {}", Hex(rel)),
            Instr::Jg32(rel) => write!(f, "jg32 {}", Hex(rel)),
            Instr::Jge32(rel) => write!(f, "jge32 {}", Hex(rel)),
            Instr::Jl32(rel) => write!(f, "jl32 {}", Hex(rel)),
            Instr::Jle32(rel) => write!(f, "jle32 {}", Hex(rel)),
            Instr::Jne32(rel) => write!(f, "jne32 {}", Hex(rel)),
            Instr::_JmpRel(rel) => write!(f, "jmp {}", Hex(rel)),
            Instr::JmpRel32(rel) => write!(f, "jmp32 {rel}"),
            Instr::MovRR(dst, src) => write!(f, "mov {dst}, {src}"),
            Instr::MovMR(dst, src) => write!(f, "mov {dst:x}, {src}"),
            Instr::MovRM(dst, src) => write!(f, "mov {dst}, {src:x}"),
            Instr::MovRI(dst, imm) => write!(f, "mov {dst}, {}", Hex64(imm)),
            Instr::MovMI(dst, imm) => write!(f, "mov dword {dst:x}, {}", Hex(imm)),
            Instr::OrRR(dst, src) => write!(f, "or {dst}, {src}"),
            Instr::OrMR(dst, src) => write!(f, "or {dst:x}, {src}"),
            Instr::OrRM(dst, src) => write!(f, "or {dst}, {src:x}"),
            Instr::OrRI(dst, imm) => write!(f, "or {dst}, {}", Hex(imm)),
            Instr::OrMI(dst, imm) => write!(f, "or dword {dst:x}, {}", Hex(imm)),
            Instr::Pop(op) => write!(f, "pop {op}"),
            Instr::Push(op) => write!(f, "push {op}"),
            Instr::Ret => write!(f, "ret"),
            Instr::SubRR(dst, src) => write!(f, "sub {dst}, {src}"),
            Instr::SubMR(dst, src) => write!(f, "sub {dst:x}, {src}"),
            Instr::SubRM(dst, src) => write!(f, "sub {dst}, {src:x}"),
            Instr::SubRI(dst, imm) => write!(f, "sub {dst}, {}", Hex(imm)),
            Instr::SubMI(dst, imm) => write!(f, "sub dword {dst:x}, {}", Hex(imm)),
            Instr::TestRR(dst, src) => write!(f, "test {dst}, {src}"),
            Instr::TestMR(dst, src) => write!(f, "test {dst:x}, {src}"),
            Instr::TestRI(dst, imm) => write!(f, "test {dst}, {}", Hex(imm)),
            Instr::TestMI(dst, imm) => write!(f, "test dword {dst:x}, {}", Hex(imm)),
            Instr::Ud2 => write!(f, "ud2"),

            Instr::Label(label) => write!(f, "Label {label:#x}:"),
            Instr::CallLabel(label) => write!(f, "CallLabel {label:#x}"),
            Instr::JeLabel(label) => write!(f, "JeLabel {label:#x}"),
            Instr::JgLabel(label) => write!(f, "JgLabel {label:#x}"),
            Instr::JgeLabel(label) => write!(f, "JgeLabel {label:#x}"),
            Instr::JlLabel(label) => write!(f, "JlLabel {label:#x}"),
            Instr::JleLabel(label) => write!(f, "JleLabel {label:#x}"),
            Instr::JneLabel(label) => write!(f, "JneLabel {label:#x}"),
            Instr::JmpLabel(label) => write!(f, "JmpLabel {label:#x}"),
        }
    }
}

impl Display for Reg {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Reg::Eax => write!(f, "eax"),
            Reg::Ecx => write!(f, "ecx"),
            Reg::Edx => write!(f, "edx"),
            Reg::Ebx => write!(f, "ebx"),
            Reg::Esp => write!(f, "esp"),
            Reg::Ebp => write!(f, "ebp"),
            Reg::Esi => write!(f, "esi"),
            Reg::Edi => write!(f, "edi"),
            Reg::R8d => write!(f, "r8d"),
            Reg::R9d => write!(f, "r9d"),
            Reg::R10d => write!(f, "r10d"),
            Reg::R11d => write!(f, "r11d"),
            Reg::R12d => write!(f, "r12d"),
            Reg::R13d => write!(f, "r13d"),
            Reg::R14d => write!(f, "r14d"),
            Reg::R15d => write!(f, "r15d"),
            Reg::Rax => write!(f, "rax"),
            Reg::Rcx => write!(f, "rcx"),
            Reg::Rdx => write!(f, "rdx"),
            Reg::Rbx => write!(f, "rbx"),
            Reg::Rsp => write!(f, "rsp"),
            Reg::Rbp => write!(f, "rbp"),
            Reg::Rsi => write!(f, "rsi"),
            Reg::Rdi => write!(f, "rdi"),
            Reg::R8 => write!(f, "r8"),
            Reg::R9 => write!(f, "r9"),
            Reg::R10 => write!(f, "r10"),
            Reg::R11 => write!(f, "r11"),
            Reg::R12 => write!(f, "r12"),
            Reg::R13 => write!(f, "r13"),
            Reg::R14 => write!(f, "r14"),
            Reg::R15 => write!(f, "r15"),
        }
    }
}

impl Display for Mem {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fmt_mem(self, f, false)
    }
}

impl LowerHex for Mem {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fmt_mem(self, f, true)
    }
}

fn fmt_mem(mem: &Mem, f: &mut Formatter<'_>, hex: bool) -> fmt::Result {
    write!(f, "[")?;
    let mut first = true;
    if let Some(base) = mem.base {
        write!(f, "{}", base)?;
        first = false;
    }
    if let Some(index) = mem.index {
        if !first {
            write!(f, " + ")?;
        }
        write!(f, "{}", index)?;
        // Print scale even if 1 to make it clear that the register is index and not base.
        // This also forces NASM to treat it as index.
        write!(f, " * {}", mem.scale)?;
        first = false;
    }
    if mem.disp != 0 {
        if !first {
            write!(f, " + ")?;
        }
        if hex {
            write!(f, "0x{:x}", mem.disp)?;
        } else {
            write!(f, "{}", mem.disp)?;
        }
    } else if first {
        // We have to print something there.
        write!(f, "0")?;
    }
    write!(f, "]")?;
    Ok(())
}

impl Display for Rex {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let byte = self.serialize();
        write!(
            f,
            "REX 0x{byte:02x} = 0b{byte:08b}: W={} R={} X={} B={}",
            self.w, self.r, self.x, self.b
        )
    }
}

impl Display for ModRm {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let byte = self.serialize();
        let register_reg = Reg::decode(false, 0, self.reg).to_meaning();
        let register_rm = Reg::decode(false, 0, self.rm).to_meaning();
        write!(
            f,
            "modRM 0x{byte:02x} = 0b{byte:08b}: mod = {:02b}, reg = {:03b} / {}, rm = {:03b} / {}",
            self.mod_, self.reg, register_reg, self.rm, register_rm
        )
    }
}

impl Display for Sib {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let byte = self.serialize();
        let register_index = Reg::decode(false, 0, self.index).to_meaning();
        let register_base = Reg::decode(false, 0, self.base).to_meaning();
        write!(
            f,
            "SIB 0x{byte:02x} = 0b{byte:08b}: scale = {:02b} = *{}, index = {:03b} / {}, base = {:03b} / {}",
            self.scale_bits(),
            self.scale,
            self.index,
            register_index,
            self.base,
            register_base,
        )
    }
}

impl Display for Num {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Num8(num) => write!(f, "0x{num:02x} (8 bits) = {num}"),
            Self::Num32(num) => write!(f, "0x{num:08x} (32 bits) = {num}"),
            Self::Num64(num) => write!(f, "0x{num:016x} (64 bits) = {num}"),
        }
    }
}

/// Hack for printing negative hex numbers with a minus sign
/// so that https://asm.x32.dev/ can parse them correctly.
///
/// We don't want to print integers using {:x} because
/// it's less readable for humans.
/// It also makes NASM pick a larger instruction unnecessarily
/// (and in a way tat makes writing predictable test cases harder).
/// E.g. jump uses i32 as offset. That means `jmp -2`
/// would become `jmp 0xfffffffe` and NASM would use a 5 byte JMP.
/// With Hex it becomes `jmp -0x2` and NASM uses the 2 byte form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Hex(i32);

impl Display for Hex {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.0 >= 0 {
            write!(f, "{:#x}", self.0)
        } else {
            // Cast to avoid overflow when self.0 == i32::MIN
            write!(f, "-{:#x}", -(self.0 as i64))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Hex64(i64);

impl Display for Hex64 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.0 >= 0 {
            write!(f, "{:#x}", self.0)
        } else {
            // Cast to avoid overflow when self.0 == i64::MIN
            write!(f, "-{:#x}", -(self.0 as i128))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex() {
        assert_eq!(format!("{}", Hex(0)), "0x0");
        assert_eq!(format!("{}", Hex(1)), "0x1");
        assert_eq!(format!("{}", Hex(-1)), "-0x1");
        assert_eq!(format!("{}", Hex(127)), "0x7f");
        assert_eq!(format!("{}", Hex(-127)), "-0x7f");
        assert_eq!(format!("{}", Hex(128)), "0x80");
        assert_eq!(format!("{}", Hex(-128)), "-0x80");
        assert_eq!(format!("{}", Hex(255)), "0xff");
        assert_eq!(format!("{}", Hex(-255)), "-0xff");
        assert_eq!(format!("{}", Hex(256)), "0x100");
        assert_eq!(format!("{}", Hex(-256)), "-0x100");
        assert_eq!(format!("{}", Hex(i32::MAX)), "0x7fffffff");
        assert_eq!(format!("{}", Hex(i32::MIN)), "-0x80000000");
    }

    #[test]
    fn test_hex64() {
        assert_eq!(format!("{}", Hex64(0)), "0x0");
        assert_eq!(format!("{}", Hex64(1)), "0x1");
        assert_eq!(format!("{}", Hex64(-1)), "-0x1");
        assert_eq!(format!("{}", Hex64(127)), "0x7f");
        assert_eq!(format!("{}", Hex64(-127)), "-0x7f");
        assert_eq!(format!("{}", Hex64(128)), "0x80");
        assert_eq!(format!("{}", Hex64(-128)), "-0x80");
        assert_eq!(format!("{}", Hex64(255)), "0xff");
        assert_eq!(format!("{}", Hex64(-255)), "-0xff");
        assert_eq!(format!("{}", Hex64(256)), "0x100");
        assert_eq!(format!("{}", Hex64(-256)), "-0x100");
        assert_eq!(format!("{}", Hex64(i64::MAX)), "0x7fffffffffffffff");
        assert_eq!(format!("{}", Hex64(i64::MIN)), "-0x8000000000000000");
    }
}
