use std::mem;

use crate::{
    fn_addr,
    jit::{asm_encoding::*, asm_repr::*, memory::JitMemory, VariableAddr},
};

use Instr::*;
use Reg::*;

const LABEL_RECURSIVE: usize = usize::MAX;

#[test]
fn test_factorial_rel() {
    factorial_handwritten(Instr::_Jg(9)); // 2 bytes - added to the offset
}

#[test]
fn test_factorial_rel8() {
    factorial_handwritten(Instr::Jg32(7)); // 6 bytes - but not added to the offset
}

#[test]
fn test_factorial_label() {
    factorial_handwritten(Instr::JgLabel(LABEL_RECURSIVE));
}

fn factorial_handwritten(jg: Instr) {
    // Recursive factorial manually written in assembly.

    let label_factorial = 0;

    let instrs = vec![
        // factorial(i32) -> i32:
        Label(label_factorial),
        // Prologue
        Push(Rbp),
        MovRR(Rbp, Rsp),
        // Base case
        CmpRI(Edi, 1),
        jg,
        MovRI(Eax, 1), // 5 bytes
        Pop(Rbp),      // 1 byte
        Ret,           // 1 byte
        // Recursive case
        Label(LABEL_RECURSIVE),
        Push(Rdi),
        Push(Rdi), // Align stack
        SubRI(Edi, 1),
        CallLabel(label_factorial),
        Pop(Rdi),
        Pop(Rdi),
        ImulR(Edi),
        // Epilogue
        Pop(Rbp),
        Ret,
    ];

    test_factorial(&instrs);
}

#[test]
fn factorial_godbolt() {
    // Recursive factorial generated by godbolt.org,
    // then transcribed into a Vec of Instrs.

    /*
    Input code:
    pub fn factorial(x: i32) -> i32 {
        if x <= 1 {
            return 1;
        }
        x * factorial(x - 1)
    }

    Flags:
    -Cdebug-assertions=n -Copt-level=0

    Generated assembly:
        push    rax
        mov     dword ptr [rsp], edi
        cmp     edi, 1
        jle     .LBB0_2
        mov     edi, dword ptr [rsp]
        sub     edi, 1
        call    qword ptr [rip + example::factorial@GOTPCREL]
        mov     ecx, eax
        mov     eax, dword ptr [rsp]
        imul    eax, ecx
        mov     dword ptr [rsp + 4], eax
        jmp     .LBB0_3
    .LBB0_2:
        mov     dword ptr [rsp + 4], 1
    .LBB0_3:
        mov     eax, dword ptr [rsp + 4]
        pop     rcx
        ret
    */

    let label_factorial = 0;
    let label_one = 1;
    let label_return = 2;

    let instrs = vec![
        // factorial(i32) -> i32:
        Label(label_factorial),
        Push(Rax), // Align stack
        MovMR(Mem::base(Rsp), Edi),
        CmpRI(Edi, 1),
        JleLabel(label_one),
        MovRM(Edi, Mem::base(Rsp)),
        SubRI(Edi, 1),
        CallLabel(label_factorial),
        MovRR(Ecx, Eax),
        MovRM(Eax, Mem::base(Rsp)),
        ImulRR(Eax, Ecx),
        MovMR(Mem::base_offset(Rsp, 4), Eax),
        JmpLabel(label_return),
        Label(label_one),
        MovMI(Mem::base_offset(Rsp, 4), 1),
        Label(label_return),
        MovRM(Eax, Mem::base_offset(Rsp, 4)),
        Pop(Rcx), // Undo align stack (don't care about result)
        Ret,
    ];

    test_factorial(&instrs);
}

#[test]
fn test_factorial_godbolt_no_rsp_r11() {
    factorial_godbolt_no_rsp(R11); // R11 is volatile
}

#[test]
fn test_factorial_godbolt_no_rsp_r12() {
    factorial_godbolt_no_rsp(R12); // R12 is nonvolatile
}

fn factorial_godbolt_no_rsp(reg: Reg) {
    // Same as factorial_godbolt but modified to minimize using Rsp.
    // This means another register needs to be used and later restored
    // without being overwritten by the recursive calls.
    // Additionally if the register used is nonvolatile,
    // we also have to
    // The outer function passes this register's original value back into Rust
    // so we can check it was restored properly.
    // Of course, if it's not restored the whole program can crash before
    // we even get to the assert.

    let label_factorial = 0;
    let label_one = 1;
    let label_return = 2;

    // Random initial values that are not the same.
    let mut before: usize = 10;
    let mut after: usize = 11;

    let before_addr = before.var_addr_mut();
    let after_addr = after.var_addr_mut();

    let instrs = vec![
        // factorial_outer(i32) -> i32:
        SubRI(Rsp, 8),
        MovRI(R8, before_addr),
        MovRI(R9, after_addr),
        MovMR(Mem::base(R8), reg),
        CallLabel(label_factorial),
        MovMR(Mem::base(R9), reg),
        AddRI(Rsp, 8),
        Ret,
        // factorial(i32) -> i32:
        Label(label_factorial),
        Push(reg),
        SubRI(Rsp, 16),
        MovRR(reg, Rsp),
        MovMR(Mem::base(reg), Edi),
        CmpRI(Edi, 1),
        JleLabel(label_one),
        MovRM(Edi, Mem::base(reg)),
        SubRI(Edi, 1),
        CallLabel(label_factorial),
        MovRR(Ecx, Eax),
        MovRM(Eax, Mem::base(reg)),
        ImulRR(Eax, Ecx),
        MovMR(Mem::base_offset(reg, 4), Eax),
        JmpLabel(label_return),
        Label(label_one),
        MovMI(Mem::base_offset(reg, 4), 1),
        Label(label_return),
        MovRM(Eax, Mem::base_offset(reg, 4)),
        AddRI(Rsp, 16),
        Pop(reg),
        Ret,
    ];

    test_factorial(&instrs);

    println!("Before: {:#x}", before);
    println!("After: {:#x}", after);
    assert_eq!(before, after);
}

fn test_factorial(instrs: &[Instr]) {
    let code = compile(instrs).code;
    let jit = JitMemory::new(&code);
    let f: extern "sysv64" fn(i32) -> i32 = unsafe { mem::transmute(jit.code) };

    let expected = [
        1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600,
    ];
    let mut got = Vec::new();
    for i in 0..expected.len() {
        let res = f(i as i32);
        got.push(res);
        println!("f({i}) = {res}");
    }
    for i in 0..expected.len() {
        assert_eq!(got[i], expected[i]);
    }
}

#[test]
fn call_functions() {
    // A more elaborate example of calling functions - Rust to jit

    let mut heap = Vec::<u8>::new();

    // We have to cast a reference to the whole Vec to a usize,
    // not use `as_mut_ptr` because we wanna pass the whole Vec through jitted code,
    // not just its data.
    // Also the vector hasn't even allocated yet so `as_mut_ptr` would return
    // something useless, for example 1.
    let heap_addr = heap.var_addr_mut();
    let heap_addr2 = &mut heap as *mut _ as i64;
    assert_eq!(heap_addr, heap_addr2);
    println!("heap_addr {:#x}", heap_addr);

    let allocate_addr = fn_addr!(allocate1);
    let take_9_args_addr = fn_addr!(take_9_args);
    println!("allocate_addr {:#x}", allocate_addr);
    println!("take_9_args_addr {:#x}", take_9_args_addr);

    let label_start = 0;
    let label_alloc = 1;
    let label_entry = 2;
    let label_reverse = 3;

    // LATER(martin-t) would be nice to have PushI
    let instrs = [
        // fn start():
        // Allocate space for 2 i32s.
        Label(label_start),
        Push(Rax),
        _CallRel(12), // 5 bytes
        _CallRel(7),  // 5 bytes
        Pop(Rcx),     // 1 byte
        Ret,          // 1 byte
        // --------------------
        // alloc(i32):
        // Allocate space for an i32.
        Label(label_alloc),
        Push(Rax),
        MovRI(R10, allocate_addr),
        MovRI(Rdi, heap_addr),
        MovRI(Rsi, 4),
        CallAbsR(R10),
        Pop(Rcx),
        Ret,
        // --------------------
        // entry:
        Label(label_entry),
        Push(R14), // R14 is callee-saved
        // Allocate space for another i32
        // (if start was called previously, 3 slots will be allocated in total).
        CallLabel(label_alloc),
        Push(R15), // R15 is callee-saved
        MovRR(R15, Rax),
        // Save 1 to the first slot.
        MovMI(Mem::base(R15), 1),
        // Call take_9_args with numbers 1 to 9.
        MovRI(Rdi, 1),
        MovRI(Rsi, 2),
        MovRI(Rdx, 3),
        MovRI(Rcx, 4),
        MovRI(R8, 5),
        MovRI(R9, 6),
        MovRI(Rax, 9),
        Push(Rax),
        MovRI(Rax, 8),
        Push(Rax),
        MovRI(Rax, 7),
        Push(Rax),
        MovRI(R14, take_9_args_addr),
        CallAbsR(R14),
        AddRI(Rsp, 24),
        // Save the result to the second slot.
        MovMR(Mem::base_offset(R15, 4), Rax),
        // Call reverse with numbers 1 to 9.
        // Args 1-6 in registers
        MovRI(Rdi, 1),
        MovRI(Rsi, 2),
        MovRI(Rdx, 3),
        MovRI(Rcx, 4),
        MovRI(R8, 5),
        MovRI(R9, 6),
        // Args 7-9 on the stack in reverse order
        MovRI(Rax, 9),
        Push(Rax),
        MovRI(Rax, 8),
        Push(Rax),
        MovRI(Rax, 7),
        Push(Rax),
        CallLabel(label_reverse),
        AddRI(Rsp, 24),
        // Save the result to the third slot.
        MovMR(Mem::base_offset(R15, 8), Rax),
        Pop(R15),
        Pop(R14),
        Ret,
        // --------------------
        // reverse:
        Label(label_reverse),
        Push(Rbp),
        MovRR(Rbp, Rsp),
        // Call f with reverse's args reversed
        // Push args 7-9 first so we can use the registers.
        Push(Rax), // Dummy so stack ends up aligned
        Push(Rdi),
        Push(Rsi),
        Push(Rdx),
        // Args 1-3 from the stack
        MovRM(Rdi, Mem::base_offset(Rbp, 32)),
        MovRM(Rsi, Mem::base_offset(Rbp, 24)),
        MovRM(Rdx, Mem::base_offset(Rbp, 16)),
        // Args 4-6: just swap 4 and 6
        MovRR(Rax, Rcx),
        MovRR(Rcx, R9),
        MovRR(R9, Rax),
        MovRI(R11, take_9_args_addr),
        CallAbsR(R11),
        AddRI(Rsp, 32),
        Pop(Rbp),
        Ret,
    ];

    let compiled = compile(&instrs);
    let jit = JitMemory::new(&compiled.code);
    let start: extern "sysv64" fn() = unsafe { mem::transmute(jit.code) };
    let entry_offset = compiled.label_offsets[&label_entry];
    let entry: extern "sysv64" fn() = unsafe { mem::transmute(jit.code.add(entry_offset)) };

    // Just a quick experiment to see if it's possible
    // to use a relative call from jitted code back into Rust.
    // It's not, at least on my machine, the difference is too large
    // so we have to save the address to a register and do an absolute call.
    // Disassembling functions like this one reveals that
    // calling from Rust into jitted code also uses this method.
    println!("jit.code: {:#x}", jit.code as usize);
    println!("take_9_args: {:#x}", take_9_args as usize);
    let diff = (jit.code as usize).abs_diff(take_9_args as usize);
    println!("diff: {diff:#x} = {diff}");
    let rel32 = u32::try_from(diff);
    println!("fits into 32 bits: {}", rel32.is_ok());

    println!("\nstart():");
    start();
    print!("heap: ");
    print_hex(&heap);
    assert_eq!(heap.len(), 8);
    assert_eq!(heap, [0, 0, 0, 0, 0, 0, 0, 0]);

    println!("\nentry():");
    entry();
    print!("heap: ");
    print_hex(&heap);
    println!("heap: {:#x?}", heap);
    assert_eq!(heap.len(), 12);

    let read_i32 = |i: usize| -> i32 {
        let range = i * 4..(i + 1) * 4;
        i32::from_le_bytes(heap[range].try_into().unwrap())
    };
    let slot1 = read_i32(0);
    let slot2 = read_i32(1);
    let slot3 = read_i32(2);
    println!("slot1: {slot1} = {slot1:#x}");
    println!("slot2: {slot2} = {slot2:#x}");
    println!("slot3: {slot3} = {slot3:#x}");

    let expected2 = take_9_args(0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9);
    let expected3 = take_9_args(0x9, 0x8, 0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1);
    println!("expected2: {expected2} = {expected2:#x}");
    println!("expected3: {expected3} = {expected3:#x}");

    assert_eq!(slot1, 1);
    assert_eq!(slot2, expected2);
    assert_eq!(slot3, expected3);
}

#[repr(C)]
struct Allocated(usize, usize);

/// Note that this can reallocate the heap's data
/// so make sure you're using the most recently returned address.
extern "sysv64" fn allocate1(heap: &mut Vec<u8>, size: usize) -> Allocated {
    let offset = heap.len();
    heap.resize(heap.len() + size, 0);
    let base = heap.as_mut_ptr() as usize;

    println!("allocate - size: {size}, base: {base:#x}, offset: {offset}");

    Allocated(base, base + offset)
}

pub extern "sysv64" fn take_9_args(
    a1: i32,
    a2: i32,
    a3: i32,
    a4: i32,
    a5: i32,
    a6: i32,
    a7: i32,
    a8: i32,
    a9: i32,
) -> i32 {
    println!("take_9_args: {a1:#x} {a2:#x} {a3:#x} {a4:#x} {a5:#x} {a6:#x} {a7:#x} {a8:#x} {a9:#x}");
    a1 * 11 + a2 * 12 + a3 * 13 + a4 * 14 + a5 * 15 + a6 * 16 + a7 * 17 + a8 * 18 + a9 * 19
}
