pub mod asm_encoding;
pub mod asm_repr;
pub mod memory;

use std::{fmt::Write, path::PathBuf};

use anyhow::Result;

use crate::bytecode::{
    program::Program,
    state::{Output, State},
};

pub fn jit_with_memory_config(program: &Program, heap_gc_size: Option<usize>, heap_log: Option<PathBuf>) -> Result<()> {
    let mut state = State::from(program)?;
    state.heap.set_gc_size(heap_gc_size);
    if let Some(log) = heap_log {
        state.heap.set_log(log);
    }
    let mut output = Output::new();
    jit_with(program, &mut state, &mut output)
}

#[allow(unused_variables)] // TODO remove
pub fn jit_with<W>(program: &Program, state: &mut State, output: &mut W) -> Result<()>
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

fn run_assembler() {
    println!("Using assembler to avoid dead code warns");

    use crate::jit::asm_encoding::*;
    use crate::jit::asm_repr::*;

    let instr = Instr::AddRM(
        Reg::R8,
        Mem {
            base: Some(Reg::Rdi),
            index: None,
            scale: 1,
            disp: 0xa,
        },
    );
    let bytes = instr.encode().to_bytes();
    print_hex(&bytes);

    Encoding::deserialize_and_print(&bytes);
}
