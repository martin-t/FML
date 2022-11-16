use crate::parser::AST;

#[macro_use]
mod helpers; // Keep this first so the macros are available everywhere

// LATER(martin-t) Remove rustfmt::skip
pub mod compiler;
pub mod debug;
pub mod interpreter;
pub mod opcodes;
pub mod program;
pub mod serializable;
pub mod heap;
pub mod state;

use anyhow::Result;

pub fn compile(ast: &AST) -> Result<program::Program> {
    compiler::compile(ast)
}
