use crate::parser::AST;

// LATER(martin-t) Remove rustfmt::skip
pub mod compiler;
pub mod debug;
pub mod interpreter;
#[rustfmt::skip] pub mod opcodes;
pub mod program;
pub mod serializable;
#[rustfmt::skip] #[macro_use] mod helpers;
pub mod heap;
#[rustfmt::skip] pub mod state;

use anyhow::Result;

pub fn compile(ast: &AST) -> Result<program::Program> {
    compiler::compile(ast)
}
