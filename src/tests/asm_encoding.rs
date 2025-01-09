//! Tests for encoding and serialization of instructions.
//!
//! Most of the expected output was generated using https://asm.x32.dev/.

use crate::jit::asm_encoding::*;
use crate::jit::asm_repr::*;

use Reg::*;

/// A "randomly" chosen register for testing without any special properties.
/// Shouldn't be *AX because some instructions (ADD) have shorter forms for it.
/// Shouldn't be *BP/*SP/R12/R13 because they have special meanings.
const TEST_REG: Reg = Reg::Ebx;
/// A "randomly" chosen memory addressing mode without any special properties.
const TEST_MEM: Mem = Mem {
    base: Some(Rdx),
    index: Some(Rcx),
    scale: 2,
    disp: 42,
};
/// Interesting i32 values
const IMMS32: [i32; 13] = [
    0,
    1,
    -1,
    i8::MAX as i32,     // 127,
    i8::MIN as i32,     // -128,
    i8::MAX as i32 + 1, // 128,
    i8::MIN as i32 - 1, // -129,
    i16::MAX as i32,
    i16::MIN as i32,
    i16::MAX as i32 + 1,
    i16::MIN as i32 - 1,
    i32::MAX,
    i32::MIN,
];
/// Interesting i32 values that can't fit in 8 bits.
///
/// 1 byte numbers have shorter special cases in some instructions
/// and normal assemblers prefer them so we skip those when generating some tests.
const IMMS32_LITE: [i32; 8] = [
    i8::MAX as i32 + 1, // 128,
    i8::MIN as i32 - 1, // -129,
    i16::MAX as i32,
    i16::MIN as i32,
    i16::MAX as i32 + 1,
    i16::MIN as i32 - 1,
    i32::MAX,
    i32::MIN,
];
/// Interesting i64 values
const IMMS64: [i64; 17] = [
    0,
    1,
    -1,
    i8::MAX as i64,     // 127,
    i8::MIN as i64,     // -128,
    i8::MAX as i64 + 1, // 128,
    i8::MIN as i64 - 1, // -129,
    i16::MAX as i64,
    i16::MIN as i64,
    i16::MAX as i64 + 1,
    i16::MIN as i64 - 1,
    i32::MAX as i64,
    i32::MIN as i64,
    i32::MAX as i64 + 1,
    i32::MIN as i64 - 1,
    i64::MAX,
    i64::MIN,
];

/// An example how the intermediate Encoding representation
/// can be used to help understand the individual parts of the binary encoding.
///
/// This works both when starting from an Instr and from already encoded machine code.
///
/// Run using: cargo test -- --nocapture example_print_encoding
#[test]
fn example_print_encoding() {
    let instr = Instr::AddRM(
        Reg::Eax,
        Mem {
            base: Some(Reg::Rbx),
            index: Some(Reg::Rcx),
            scale: 4,
            disp: 100,
        },
    );

    let encoded = instr.encode();
    println!("{}", encoded);

    let hex = encoded.serialize_to_vec();
    print_asm(&hex);
    println!();

    let decoded = Encoding::deserialize(&hex);
    println!("{}", decoded.0);

    assert_eq!(encoded, decoded.0);
}

#[test]
fn test_simple() {
    let instrs = vec![
        Instr::Nop,
        Instr::Cdq,
        Instr::Cqo,
        Instr::Hlt,
        Instr::Int3,
        Instr::Int1,
        Instr::Ret,
        Instr::Ud2,
    ];
    let expecteds: &[&[u8]] = &[
        &[0x90],       // nop
        &[0x99],       // cdq
        &[0x48, 0x99], // cqo
        &[0xf4],       // hlt
        &[0xcc],       // int3
        &[0xf1],       // int1
        &[0xc3],       // ret
        &[0x0f, 0x0b], // ud2
    ];
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_add_examples() {
    let instrs = vec![
        // https://www.systutorials.com/beginners-guide-x86-64-instruction-encoding/
        // add r8, [rdi + 0xa]
        Instr::AddRM(Reg::R8, Mem::base_offset(Reg::Rdi, 0xa)),
        // https://stackoverflow.com/questions/28664856/how-to-interpret-x86-opcode-map
        // The example would require address-size override prefix
        // so we use 64 bit regs here:
        // add edx, [rbx + rcx*4 + 0x15]
        Instr::AddRM(Reg::Edx, Mem::base_index_offset(Reg::Rbx, Reg::Rcx, 4, 0x15)),
        // Based on https://stackoverflow.com/questions/52522544/rbp-not-allowed-as-sib-base
        // add eax, [rbp+rcx]
        Instr::AddRM(Reg::Eax, Mem::base_index(Reg::Rbp, Reg::Rcx, 1)),
    ];

    let expecteds: &[&[u8]] = &[
        &[0x4c, 0x03, 0x47, 0x0a], // add r8, [rdi + 0xa]
        &[0x03, 0x54, 0x8b, 0x15], // add edx, [rbx + rcx * 4 + 0x15]
        &[0x03, 0x44, 0x0d, 0x00], // add eax, [rbp+rcx]
    ];

    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_add_reg_reg() {
    let mut instrs = Vec::new();
    for (dst, src) in combinations_reg_reg() {
        instrs.push(Instr::AddRR(dst, src));
    }
    let expecteds: &[&[u8]] = &include!("data/add_reg_reg.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_add_mem_reg() {
    let mut instrs = Vec::new();
    for (mem, reg) in combinations_mem_reg() {
        instrs.push(Instr::AddMR(mem, reg));
    }
    let expecteds: &[&[u8]] = &include!("data/add_mem_reg.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_add_reg_mem() {
    let mut instrs = Vec::new();
    for (mem, reg) in combinations_mem_reg() {
        instrs.push(Instr::AddRM(reg, mem));
    }
    let expecteds: &[&[u8]] = &include!("data/add_reg_mem.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_add_reg_imm() {
    let mut instrs = Vec::new();
    for (reg, imm) in combinations_reg_imm() {
        instrs.push(Instr::AddRI(reg, imm));
    }
    let expecteds: &[&[u8]] = &include!("data/add_reg_imm.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_add_mem_imm() {
    let mut instrs = Vec::new();
    for (mem, imm) in combinations_mem_imm() {
        instrs.push(Instr::AddMI(mem, imm));
    }
    let expecteds: &[&[u8]] = &include!("data/add_mem_imm.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_and_reg_reg() {
    let mut instrs = Vec::new();
    for (dst, src) in combinations_reg_reg() {
        instrs.push(Instr::AndRR(dst, src));
    }
    let expecteds: &[&[u8]] = &include!("data/and_reg_reg.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_and_mem_reg() {
    let mut instrs = Vec::new();
    for (mem, reg) in combinations_mem_reg() {
        instrs.push(Instr::AndMR(mem, reg));
    }
    let expecteds: &[&[u8]] = &include!("data/and_mem_reg.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_and_reg_mem() {
    let mut instrs = Vec::new();
    for (mem, reg) in combinations_mem_reg() {
        instrs.push(Instr::AndRM(reg, mem));
    }
    let expecteds: &[&[u8]] = &include!("data/and_reg_mem.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_and_reg_imm() {
    let mut instrs = Vec::new();
    for (reg, imm) in combinations_reg_imm() {
        instrs.push(Instr::AndRI(reg, imm));
    }
    let expecteds: &[&[u8]] = &include!("data/and_reg_imm.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_call_rel() {
    let mut instrs = Vec::new();
    for rel in [i32::MIN + 5, -12345, -5, 0, 5, 12345, i32::MAX] {
        instrs.push(Instr::_CallRel(rel));
    }
    let expecteds: &[&[u8]] = &include!("data/call_rel.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_call_abs_reg() {
    let mut instrs = Vec::new();
    for reg in Reg::variants() {
        if reg.is_32bit() {
            continue;
        }
        instrs.push(Instr::CallAbsR(reg));
    }
    let expecteds: &[&[u8]] = &include!("data/call_abs_reg.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_call_abs_mem() {
    let mut instrs = Vec::new();
    for mem in combinations_mem() {
        instrs.push(Instr::CallAbsM(mem));
    }
    let expecteds: &[&[u8]] = &include!("data/call_abs_mem.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_idiv_imul_1op() {
    let mut instrs = Vec::new();
    for reg in Reg::variants() {
        instrs.push(Instr::IdivR(reg));
        instrs.push(Instr::ImulR(reg));
    }
    for mem in combinations_mem() {
        instrs.push(Instr::IdivM(mem));
        instrs.push(Instr::ImulM(mem));
    }
    let expecteds: &[&[u8]] = &include!("data/idiv_imul_1op.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_imul_2ops() {
    let mut instrs = Vec::new();
    for (dst, src) in combinations_reg_reg() {
        instrs.push(Instr::ImulRR(dst, src));
    }
    for (mem, reg) in combinations_mem_reg() {
        instrs.push(Instr::ImulRM(reg, mem));
    }
    let expecteds: &[&[u8]] = &include!("data/idiv_imul_2ops.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_jcc() {
    let mut instrs = Vec::new();
    for rel in [
        // The 32 bit version has 6 bytes
        i32::MIN + 6,
        -12345,
        -127,
        -126,
        -2,
        0,
        2,
        129,
        130,
        12345,
        i32::MAX,
    ] {
        instrs.push(Instr::_Je(rel));
        instrs.push(Instr::_Jg(rel));
        instrs.push(Instr::_Jge(rel));
        instrs.push(Instr::_Jl(rel));
        instrs.push(Instr::_Jle(rel));
        instrs.push(Instr::_Jne(rel));
    }
    let expecteds: &[&[u8]] = &include!("data/jcc.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_jmp_rel() {
    let mut instrs = Vec::new();
    for rel in [
        // The 32 bit version has 5 bytes
        i32::MIN + 5,
        -12345,
        -127,
        -126,
        -2,
        0,
        2,
        129,
        130,
        12345,
        i32::MAX,
    ] {
        instrs.push(Instr::_JmpRel(rel));
    }
    let expecteds: &[&[u8]] = &include!("data/jmp_rel.in");
    assert_encoding_and_serialization(&instrs, expecteds);

    // The exact boundary between rel8 and rel32 (comment/uncomment ;nop):
    // a:
    // mov r15, 0x8000000000000000
    // mov r15, 0x8000000000000000
    // mov r15, 0x8000000000000000
    // mov r15, 0x8000000000000000
    // mov r15, 0x8000000000000000
    // mov r15, 0x8000000000000000
    // mov r15, 0x8000000000000000
    // mov r15, 0x8000000000000000
    // mov r15, 0x8000000000000000
    // mov r15, 0x8000000000000000
    //
    // mov r15, 0x8000000000000000
    // mov r15, 0xeeeeeeeeeeeeeeee
    //
    // nop
    // nop
    // nop
    // nop
    // nop
    // nop
    // ;nop
    //
    // b:
    // jmp a
    //
    // c:
    // nop
}

#[test]
fn test_mov_reg_reg() {
    let mut instrs = Vec::new();
    for (dst, src) in combinations_reg_reg() {
        instrs.push(Instr::MovRR(dst, src));
    }
    let expecteds: &[&[u8]] = &include!("data/mov_reg_reg.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_mov_mem_reg() {
    let mut instrs = Vec::new();
    for (mem, reg) in combinations_mem_reg() {
        instrs.push(Instr::MovMR(mem, reg));
    }
    let expecteds: &[&[u8]] = &include!("data/mov_mem_reg.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_mov_reg_mem() {
    let mut instrs = Vec::new();
    for (mem, reg) in combinations_mem_reg() {
        instrs.push(Instr::MovRM(reg, mem));
    }
    let expecteds: &[&[u8]] = &include!("data/mov_reg_mem.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_mov_reg_imm() {
    let mut instrs = Vec::new();
    for dst in Reg::variants() {
        for imm in IMMS64 {
            let imm32: Option<i32> = imm.try_into().ok();
            if dst.is_32bit() && imm32.is_none() {
                continue;
            }
            instrs.push(Instr::MovRI(dst, imm));
        }
    }
    let expecteds: &[&[u8]] = &include!("data/mov_reg_imm.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_mov_mem_imm() {
    let mut instrs = Vec::new();
    for dst in combinations_mem() {
        instrs.push(Instr::MovMI(dst, 42));
    }
    for imm in IMMS32 {
        instrs.push(Instr::MovMI(TEST_MEM, imm));
    }
    let expecteds: &[&[u8]] = &include!("data/mov_mem_imm.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_pop_push_reg() {
    let mut instrs = Vec::new();
    for reg in Reg::variants() {
        if reg.is_32bit() {
            continue;
        }
        instrs.push(Instr::PopR(reg));
        instrs.push(Instr::PushR(reg));
    }
    let expecteds: &[&[u8]] = &include!("data/pop_push_reg.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_pop_push_mem() {
    let mut instrs = Vec::new();
    for mem in combinations_mem() {
        instrs.push(Instr::PopM(mem));
        instrs.push(Instr::PushM(mem));
    }
    let expecteds: &[&[u8]] = &include!("data/pop_push_mem.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_addlike() {
    // These instructions are similar to ADD and AND
    // so the tests are less detailed.

    let instrs = [
        Instr::CmpRR(Esp, Ebp),
        Instr::CmpRR(Rsp, Rbp),
        Instr::CmpMR(TEST_MEM, Esp),
        Instr::CmpMR(TEST_MEM, Rsp),
        Instr::CmpRM(Esp, TEST_MEM),
        Instr::CmpRM(Rsp, TEST_MEM),
        Instr::CmpRI(Esp, 420),
        Instr::CmpRI(Rsp, 420),
        Instr::CmpMI(TEST_MEM, 420),
        Instr::CmpMI(TEST_MEM, 420),
        //
        Instr::OrRR(Esp, Ebp),
        Instr::OrRR(Rsp, Rbp),
        Instr::OrMR(TEST_MEM, Esp),
        Instr::OrMR(TEST_MEM, Rsp),
        Instr::OrRM(Esp, TEST_MEM),
        Instr::OrRM(Rsp, TEST_MEM),
        Instr::OrRI(Esp, 420),
        Instr::OrRI(Rsp, 420),
        Instr::OrMI(TEST_MEM, 420),
        Instr::OrMI(TEST_MEM, 420),
        //
        Instr::SubRR(Esp, Ebp),
        Instr::SubRR(Rsp, Rbp),
        Instr::SubMR(TEST_MEM, Esp),
        Instr::SubMR(TEST_MEM, Rsp),
        Instr::SubRM(Esp, TEST_MEM),
        Instr::SubRM(Rsp, TEST_MEM),
        Instr::SubRI(Esp, 420),
        Instr::SubRI(Rsp, 420),
        Instr::SubMI(TEST_MEM, 420),
        Instr::SubMI(TEST_MEM, 420),
        //
        Instr::TestRR(Esp, Ebp),
        Instr::TestRR(Rsp, Rbp),
        Instr::TestMR(TEST_MEM, Esp),
        Instr::TestMR(TEST_MEM, Rsp),
        Instr::TestRI(Esp, 420),
        Instr::TestRI(Rsp, 420),
        Instr::TestMI(TEST_MEM, 420),
        Instr::TestMI(TEST_MEM, 420),
    ];

    let expecteds: &[&[u8]] = &include!("data/addlike.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_addressing_single_reg() {
    let mem_regs = [
        Rax, Rcx, Rdx, Rbx, Rsp, Rbp, Rsi, Rdi, R8, R9, R10, R11, R12, R13, R14, R15,
    ];

    let mut instrs = Vec::new();
    for &base in &mem_regs {
        instrs.push(Instr::MovRM(TEST_REG, Mem::base(base)));
    }
    for &index in &mem_regs {
        if index == Rsp {
            continue;
        }
        instrs.push(Instr::MovRM(TEST_REG, Mem::index(index, 1)));
    }

    let expecteds: &[&[u8]] = &include!("data/addressing_single_reg.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

#[test]
fn test_addressing_combinations() {
    // All 64-bit regs except RSP
    let mem_regs = [Rax, Rcx, Rdx, Rbx, Rbp, Rsi, Rdi, R8, R9, R10, R11, R12, R13, R14, R15];

    let mut instrs = Vec::new();
    for &reg in &mem_regs {
        for mem in [
            Mem::base(reg),
            Mem::base_offset(reg, 42),
            Mem::index(reg, 1),
            Mem::index_offset(reg, 1, 42),
            Mem::base_index(reg, reg, 1),
            Mem::base_index_offset(reg, reg, 1, 42),
        ] {
            instrs.push(Instr::MovRM(TEST_REG, mem));
        }
    }

    // Same as above, except cases where RSP is the index
    instrs.push(Instr::MovRM(TEST_REG, Mem::base(Rsp)));
    instrs.push(Instr::MovRM(TEST_REG, Mem::base_offset(Rsp, 42)));

    let expecteds: &[&[u8]] = &include!("data/addressing_combinations.in");
    assert_encoding_and_serialization(&instrs, expecteds);
}

fn combinations_reg_reg() -> Vec<(Reg, Reg)> {
    let mut ret = Vec::new();

    for dst in Reg::variants() {
        for src in Reg::variants() {
            if dst.is_64bit() != src.is_64bit() {
                continue;
            }

            ret.push((dst, src));
        }
    }

    ret
}

fn combinations_mem_reg() -> Vec<(Mem, Reg)> {
    let mut ret = Vec::new();

    let mems_mix = combinations_mem_mix();
    for dst in [Eax, Edi, Rax, Rdi, R8, R15] {
        for &mem in &mems_mix {
            ret.push((mem, dst));
        }
    }

    for dst in Reg::variants() {
        ret.push((TEST_MEM, dst));
    }
    for mem in combinations_mem_each() {
        ret.push((mem, TEST_REG));
    }

    ret
}

fn combinations_reg_imm() -> Vec<(Reg, i32)> {
    let mut ret = Vec::new();
    for reg in Reg::variants() {
        if reg.encode().1 == Eax.encode().1 {
            // Reg::*AX have shorter special cases
            // and normal assemblers prefer them
            // so we skip them when generating tests.
            continue;
        }
        for imm in IMMS32_LITE {
            ret.push((reg, imm));
        }
    }
    ret
}

fn combinations_mem_imm() -> Vec<(Mem, i32)> {
    let mut ret = Vec::new();
    for mem in combinations_mem() {
        ret.push((mem, 420));
    }
    for imm in IMMS32_LITE {
        ret.push((TEST_MEM, imm));
    }
    ret
}

fn combinations_mem() -> Vec<Mem> {
    let mut mems = combinations_mem_mix();
    mems.append(&mut combinations_mem_each());
    mems
}

/// Test a couple combinations of everything.
/// A few registers, a few scales, a few immediates.
fn combinations_mem_mix() -> Vec<Mem> {
    let mut ret = Vec::new();

    let mem_regs = [None, Some(Rax), Some(Rdi), Some(R8), Some(R15)];
    for &base in &mem_regs {
        for &index in &mem_regs {
            if base.is_none() && index.is_none() {
                continue;
            }
            for scale in [1, 8] {
                for disp in [0, -128, i32::MIN] {
                    let mem = Mem {
                        base,
                        index,
                        scale,
                        disp,
                    };
                    ret.push(mem);
                }
            }
        }
    }

    ret
}

/// Test more possible values in each position but not all combinations:
/// - all registers except those we explicitly don't support
/// - all 4 scales
/// - more displacements
fn combinations_mem_each() -> Vec<Mem> {
    let mut ret = Vec::new();

    let mem_regs = [
        None,
        Some(Rax),
        Some(Rcx),
        Some(Rdx),
        Some(Rbx),
        // No RSP
        Some(Rbp),
        Some(Rsi),
        Some(Rdi),
        Some(R8),
        Some(R9),
        Some(R10),
        Some(R11),
        // No R12
        Some(R13),
        Some(R14),
        Some(R15),
    ];
    for base in mem_regs {
        let mem = Mem { base, ..TEST_MEM };
        ret.push(mem);
    }
    for index in mem_regs {
        let mem = Mem { index, ..TEST_MEM };
        ret.push(mem);
    }
    for scale in [1, 2, 4, 8] {
        let mem = Mem { scale, ..TEST_MEM };
        ret.push(mem);
    }
    for disp in [0, 1, -1, 127, -128, 128, -129, i32::MAX, i32::MIN] {
        let mem = Mem { disp, ..TEST_MEM };
        ret.push(mem);
    }

    ret
}

fn assert_encoding_and_serialization(instrs: &[Instr], expecteds: &[&[u8]]) {
    // The include files for expecteds
    // are generated by pasting the instructions into
    // https://asm.x32.dev/ and suitably editing the result
    // to become a valid rust expression.
    for instr in instrs {
        // Depending on which assembler we're using to generate tests,
        // we might want to print in hex because
        // e.g. https://asm.x32.dev/ treats everything as hex anyway.
        println!("{:x}", instr);
    }
    println!("{} instructions total", instrs.len());

    assert_eq!(instrs.len(), expecteds.len());
    for (instr, &expected) in instrs.iter().zip(expecteds) {
        println!("--------\n");
        println!("{instr}\n");

        println!("Expected:");
        print_asm(expected);
        let (expected_encoding, expected_consumed) = Encoding::deserialize(expected);
        println!("{expected_encoding}");

        println!("Got:");
        let got_encoding1 = instr.encode();
        let got = got_encoding1.serialize_to_vec();
        print_asm(&got);
        let (got_encoding2, got_consumed) = Encoding::deserialize(&got);
        println!("{got_encoding2}");

        assert_eq!(expected_consumed, expected.len());
        assert_eq!(got_consumed, got.len());
        assert_eq!(
            got, expected,
            "in hex\n      got: {got:02x?}\n expected: {expected:02x?}"
        );
        assert_eq!(got_encoding1, expected_encoding);
        assert_eq!(got_encoding2, expected_encoding);
    }
}

#[test]
#[should_panic]
fn test_rsp_index() {
    Instr::AddRM(
        Eax,
        Mem {
            base: None,
            index: Some(Rsp),
            scale: 1,
            disp: 42,
        },
    )
    .encode();
}

#[test]
#[should_panic]
fn test_rip_relative() {
    Instr::AddRM(
        Eax,
        Mem {
            base: None,
            index: None,
            scale: 1,
            disp: 42,
        },
    )
    .encode();
}

#[test]
#[should_panic]
fn test_call_32bit() {
    Instr::CallAbsR(Eax).encode();
}

#[test]
#[should_panic]
fn test_pop_32bit() {
    Instr::PopR(Eax).encode();
}

#[test]
#[should_panic]
fn test_push_32bit() {
    Instr::PushR(Eax).encode();
}
