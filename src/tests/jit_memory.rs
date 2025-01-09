//! A bunch of simple functions written in assembly.
//!
//! This serves multiple purposes
//! - as tests for JitMemory
//! - to test out assumptions about assembly on multiple platforms
//! - as examples of simple assembly functions

use std::arch::asm;

use crate::{
    jit::{
        asm_encoding::{compile, fmt_hex},
        asm_repr::{Instr, Reg},
        memory::JitMemory,
    },
    jit_fn,
};

use Instr::*;
use Reg::*;

#[test]
fn test_fn_void() {
    let instrs = [Ret];
    let code = compile(&instrs).code;
    let jit = JitMemory::new(&code);
    let f = jit_fn!(jit, fn());
    f();
    f();
}

#[test]
fn test_fn_int() {
    let instrs = [MovRI(Rax, 1337), Ret];
    let code = compile(&instrs).code;
    let jit = JitMemory::new(&code);
    let f = jit_fn!(jit, fn() -> i32);
    let ret = f();
    assert_eq!(ret, 1337);
    let ret = f();
    assert_eq!(ret, 1337);
}

#[test]
fn return_args_1_6() {
    // 6 functions that return arguments 1-6.

    let code1 = [
        0x48, 0x89, 0xf8, // mov rax, rdi
        0xc3, // ret
    ];
    let code2 = [
        0x48, 0x89, 0xf0, // mov rax, rsi
        0xc3, // ret
    ];
    let code3 = [
        0x48, 0x89, 0xd0, // mov rax, rdx
        0xc3, // ret
    ];
    let code4 = [
        0x48, 0x89, 0xc8, // mov rax, rcx
        0xc3, // ret
    ];
    let code5 = [
        0x4c, 0x89, 0xc0, // mov rax, r8
        0xc3, // ret
    ];
    let code6 = [
        0x4c, 0x89, 0xc8, // mov rax, r9
        0xc3, // ret
    ];
    let codes = [code1, code2, code3, code4, code5, code6];

    for (i, code) in codes.iter().enumerate() {
        println!("{}", fmt_hex(code));
        let jit = JitMemory::new(code);
        let func = jit_fn!(jit, fn(i32, i32, i32, i32, i32, i32) -> i32);
        let ret = func(1001, 1002, 1003, 1004, 1005, 1006);
        assert_eq!(ret, i as i32 + 1001);
    }
}

#[test]
fn add4() {
    // Take 4 numbers as args and add them together.

    let code = [
        0x49, 0x89, 0xf9, // mov r9, rdi
        0x49, 0x01, 0xf1, // add r9, rsi
        0x49, 0x01, 0xd1, // add r9, rdx
        0x49, 0x01, 0xc9, // add r9, rcx
        0x4c, 0x89, 0xc8, // mov rax, r9
        0xc3, // ret
    ];

    println!("{}", fmt_hex(&code));

    let jit = JitMemory::new(&code);
    let add4 = jit_fn!(jit, fn(i32, i32, i32, i32) -> i32);

    let ret = add4(1, 2, 3, 4);
    assert_eq!(ret, 1 + 2 + 3 + 4);

    let vals = [-2, -1, 0, 1, 3, 5, 6, 7];
    for a in vals {
        for b in vals {
            let ret = add4(a, a, b, b);
            assert_eq!(ret, a + a + b + b);
        }
    }
}

/// Interesting values for 4-arg functions.
const LIST4: [(i32, i32, i32, i32); 35] = [
    (1, 2, 3, 4),
    //
    (0, 0, 0, 0),
    (0, 0, 0, 2),
    (0, 0, 2, 0),
    (0, 2, 0, 0),
    (2, 0, 0, 0),
    (0, 0, 2, 2),
    (0, 2, 0, 2),
    (0, 2, 2, 0),
    (2, 0, 0, 2),
    (2, 0, 2, 0),
    (2, 2, 0, 0),
    (0, 2, 2, 2),
    (2, 0, 2, 2),
    (2, 2, 0, 2),
    (2, 2, 2, 0),
    (2, 2, 2, 2),
    //
    (3, 2, 1, 0),
    //
    (1, 1, 1, 1),
    (1, 1, 1, 2),
    (1, 1, 2, 1),
    (1, 2, 1, 1),
    (2, 1, 1, 1),
    (1, 1, 2, 2),
    (1, 2, 1, 2),
    (1, 2, 2, 1),
    (2, 1, 1, 2),
    (2, 1, 2, 1),
    (2, 2, 1, 1),
    (1, 2, 2, 2),
    (2, 1, 2, 2),
    (2, 2, 1, 2),
    (2, 2, 2, 1),
    (2, 2, 2, 2),
    //
    (4, 3, 2, 1),
];

#[test]
fn add_squares_static_asm() {
    // Take 4 numbers and add their squares.

    fn add_squares(a: i32, b: i32, c: i32, d: i32) -> i32 {
        let ret;
        unsafe {
            asm!(
                // Imul has multiple variants.
                // With one operand, it writes into rax and rdx.
                // With two operands, it writes into the first operand.
                // So we use the 2 operand version here to avoid overwriting rdx.
                "push rbx",
                "xor rbx, rbx",
                "mov rax, rdi",
                "imul rax, rax",
                "add rbx, rax",
                "mov rax, rsi",
                "imul rax, rax",
                "add rbx, rax",
                "mov rax, rdx",
                "imul rax, rax",
                "add rbx, rax",
                "mov rax, rcx",
                "imul rax, rax",
                "add rbx, rax",
                "mov rax, rbx",
                "pop rbx",
                in("rdi") a,
                in("rsi") b,
                in("rdx") c,
                in("rcx") d,
                out("rax") ret,
            )
        }
        ret
    }

    for (a, b, c, d) in LIST4 {
        print!("a={}, b={}, c={}, d={}", a, b, c, d);
        let expected = a * a + b * b + c * c + d * d;
        let ret = add_squares(a, b, c, d);
        print!(" => {}", ret);
        if ret == expected {
            print!(" OK");
        } else {
            print!(" should be {}", expected);
        }
        println!();
        assert_eq!(ret, expected);
    }
}

#[test]
fn add_squares_jit_asm() {
    // Take 4 numbers and add their squares.
    // Same as add_squares_static_asm but using our encoder.

    let code = [
        0x53, // push rbx
        0x48, 0x31, 0xdb, // xor rbx, rbx
        0x48, 0x89, 0xf8, // mov rax, rdi
        0x48, 0x0f, 0xaf, 0xc0, // imul rax, rax
        0x48, 0x01, 0xc3, // add rbx, rax
        0x48, 0x89, 0xf0, // mov rax, rsi
        0x48, 0x0f, 0xaf, 0xc0, // imul rax, rax
        0x48, 0x01, 0xc3, // add rbx, rax
        0x48, 0x89, 0xd0, // mov rax, rdx
        0x48, 0x0f, 0xaf, 0xc0, // imul rax, rax
        0x48, 0x01, 0xc3, // add rbx, rax
        0x48, 0x89, 0xc8, // mov rax, rcx
        0x48, 0x0f, 0xaf, 0xc0, // imul rax, rax
        0x48, 0x01, 0xc3, // add rbx, rax
        0x48, 0x89, 0xd8, // mov rax, rbx
        0x5b, // pop rbx
        0xc3, // ret
    ];

    let jit = JitMemory::new(&code);
    let add_squares = jit_fn!(jit, fn(i32, i32, i32, i32) -> i32);

    for (a, b, c, d) in LIST4 {
        print!("a={}, b={}, c={}, d={}", a, b, c, d);
        let expected = a * a + b * b + c * c + d * d;
        let ret = add_squares(a, b, c, d);
        print!(" => {}", ret);
        if ret == expected {
            print!(" OK");
        } else {
            print!(" should be {}", expected);
        }
        println!();
        assert_eq!(ret, expected);
    }
}

#[test]
fn call_square_sysv64() {
    // Call a function defined in Rust

    #[rustfmt::skip]
    let mut code = [
        0x48, 0xb8, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, // mov rax, 0x6666666666666666
        0xff, 0xd0, // call rax
        0xc3, // ret
    ];

    let addr = square_sysv64 as *const ();
    println!("addr {:?}", addr);
    let addr = (addr as usize as u64).to_le_bytes();
    println!("{}", fmt_hex(&code));
    let fn_begin = 2;
    let fn_end = fn_begin + 8;
    // Sanity check we're replacing the right bytes
    assert_eq!(code[fn_begin..fn_end], 0x6666666666666666_u64.to_le_bytes());
    code[fn_begin..fn_end].copy_from_slice(&addr);
    println!("{}", fmt_hex(&code));

    let jit = JitMemory::new(&code);
    let asm_square = jit_fn!(jit, fn(i32) -> i32);

    let ret = asm_square(7);
    assert_eq!(ret, 49);
}

#[test]
fn add_squares_square_sysv64() {
    // Take 4 numbers and add their squares.
    // Same as add_squares_static_asm but call a function to square them.

    #[rustfmt::skip]
    let mut code = [
        0x53, // push rbx
        0x41, 0x54, // push r12
        0x41, 0x55, // push r13
        0x41, 0x56, // push r14
        0x41, 0x57, // push r15
        // RSP must be aligned here.
        // Might have to sub 8 if adding/removing push instructions
        //0x48, 0x83, 0xec, 0x08, // sub rsp, 8

        // Also change monkeypatch offset if changing this
        0x49, 0xbc, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, // mov r12, 0x6666666666666666

        // Save args into nonvalatile registers
        0x49, 0x89, 0xf5, // mov r13, rsi
        0x49, 0x89, 0xd6, // mov r14, rdx
        0x49, 0x89, 0xcf, // mov r15, rcx

        // Zero out the sum
        0x48, 0x31, 0xdb, // xor rbx, rbx

        // First argument is in rdi - no need for a mov
        0x41, 0xff, 0xd4, // call r12
        0x48, 0x01, 0xc3, // add rbx, rax

        0x4c, 0x89, 0xef, // mov rdi, r13
        0x41, 0xff, 0xd4, // call r12
        0x48, 0x01, 0xc3, // add rbx, rax

        0x4c, 0x89, 0xf7, // mov rdi, r14
        0x41, 0xff, 0xd4, // call r12
        0x48, 0x01, 0xc3, // add rbx, rax

        0x4c, 0x89, 0xff, // mov rdi, r15
        0x41, 0xff, 0xd4, // call r12
        0x48, 0x01, 0xc3, // add rbx, rax

        0x48, 0x89, 0xd8, // mov rax, rbx

        0x41, 0x5f, // pop r15
        0x41, 0x5e, // pop r14
        0x41, 0x5d, // pop r13
        0x41, 0x5c, // pop r12
        0x5b, // pop rbx

        0xc3, // ret
    ];

    let addr = square_sysv64 as *const ();
    println!("addr {:?}", addr);
    let addr = (addr as usize as u64).to_le_bytes();
    println!("{}", fmt_hex(&code));
    let fn_begin = 11;
    let fn_end = fn_begin + 8;
    // Sanity check we're replacing the right bytes
    assert_eq!(code[fn_begin..fn_end], 0x6666666666666666_u64.to_le_bytes());
    code[fn_begin..fn_end].copy_from_slice(&addr);
    println!("{}", fmt_hex(&code));

    let jit = JitMemory::new(&code);
    let add_squares = jit_fn!(jit, fn(i32, i32, i32, i32) -> i32);

    for (a, b, c, d) in LIST4 {
        print!("a={}, b={}, c={}, d={}", a, b, c, d);
        let expected = a * a + b * b + c * c + d * d;
        let ret = add_squares(a, b, c, d);
        print!(" => {}", ret);
        if ret == expected {
            print!(" OK");
        } else {
            print!(" should be {}", expected);
        }
        println!();
        assert_eq!(ret, expected);
    }

    let vals = [-2, -1, 0, 1, 3, 5, 6, 7];
    for a in vals {
        for b in vals {
            let ret = add_squares(a, a, b, b);
            assert_eq!(ret, 2 * a * a + 2 * b * b, "Failed for inputs a={a}, b={b}");
        }
    }
}

extern "sysv64" fn square_sysv64(a: i32) -> i32 {
    a * a
}
