#![cfg_attr(target_os = "windows", allow(dead_code))] // LATER(martin-t) Remove

pub mod asm_encoding;
pub mod asm_repr;

#[cfg(not(target_os = "windows"))]
mod memory_unix;
#[cfg(not(target_os = "windows"))]
pub mod memory {
    pub use super::memory_unix::*;
}

use std::{fmt::Write, path::PathBuf};

use anyhow::Result;

use crate::bytecode::{
    program::Program,
    state::{Output, State},
};

pub trait ToAddr {
    /// Intentionally named "const" to avoid using it accidentally
    /// instead of "to_addr_mut".
    fn take_addr_const(&self) -> i64;
    fn take_addr_mut(&mut self) -> i64;
}

impl<T> ToAddr for T {
    fn take_addr_const(&self) -> i64 {
        self as *const _ as i64
    }

    fn take_addr_mut(&mut self) -> i64 {
        self as *mut _ as i64
    }
}

pub fn jit_with_memory_config(program: &Program, heap_gc_size: Option<usize>, heap_log: Option<PathBuf>) -> Result<()> {
    let mut state = State::from(program)?;
    state.heap.set_gc_size(heap_gc_size);
    if let Some(log) = heap_log {
        state.heap.set_log(log);
    }
    let mut output = Output::new();
    jit_with(program, &mut state, &mut output)
}

#[allow(unused_variables)] // LATER(martin-t) remove
#[cfg(target_os = "windows")]
fn jit_with<W>(program: &Program, state: &mut State, output: &mut W) -> Result<()>
where
    W: Write,
{
    unimplemented!("Windows is currently not supported")
}

#[allow(unused_variables)] // TODO remove
#[cfg(not(target_os = "windows"))]
fn jit_with<W>(program: &Program, state: &mut State, output: &mut W) -> Result<()>
where
    W: Write,
{
    // eprintln!("Program:");
    // eprintln!("{}", program);
    // while let Some(address) = state.instruction_pointer.get() {
    //     let opcode = program.code.get(address)?;
    //     eval_opcode(program, state, output, opcode)?;
    // }
    // Ok(())
    run_assembler();
    todo!();
}

#[cfg(not(target_os = "windows"))]
fn run_assembler() {
    println!("Using assembler to avoid dead code warns");

    use crate::jit::asm_encoding::*;
    use crate::jit::asm_repr::*;
    use Instr::*;
    use Reg::*;

    let instrs = [Ret];
    let bytes = compile(&instrs).code;
    Encoding::deserialize_and_print(&bytes);

    let jit = memory::JitMemory::new(&bytes);
    let f: extern "sysv64" fn() -> () = unsafe { std::mem::transmute(jit.code) };
    f();

    println!("\n===================\n");

    // From https://asm.x32.dev/
    let code = [
        0x81, 0x02, 0xaa, 0x00, 0x00, 0x00, // add dword [rdx], 0xaa
        0x48, 0x81, 0x02, 0xaa, 0x00, 0x00, 0x00, // add qword [rdx], 0xaa
        0x03, 0x02, // add eax, [rdx]
        0x48, 0x03, 0x02, // add rax, [rdx]
    ];
    Encoding::deserialize_and_print_all(&code);

    println!("\n-------------------\n");

    let instrs = [
        AddMI(Mem::base(Rdx), 0xaa),
        AddMI(Mem::base(Rdx), 0xaa), // LATER(martin-t) Add a way to specify qword
        AddRM(Eax, Mem::base(Rdx)),
        AddRM(Rax, Mem::base(Rdx)),
    ];
    let code = compile(&instrs).code;
    Encoding::deserialize_and_print_all(&code);
}
