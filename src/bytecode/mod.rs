use crate::parser::AST;

// LATER(martin-t) Remove rustfmt::skip
pub(crate) mod opcodes;
pub(crate) mod compiler;
pub(crate) mod debug;
pub mod program;
pub mod serializable;
pub mod interpreter;
#[rustfmt::skip] #[macro_use] mod helpers;
#[rustfmt::skip] pub mod heap;
pub mod state;

use anyhow::Result;

pub fn compile(ast: &AST) -> Result<program::Program> {
    compiler::compile(ast)
}
