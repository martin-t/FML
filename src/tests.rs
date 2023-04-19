// LATER(martin-t) Put #[rustfmt::skip] only where necessary.
// Would be nice to format tests as well
// but it's too much code that needs manual fixes.
// Some parts (e.g serialized opcodes) are intentionally aligned
// so they make sense to a human reader and rustfmt would make a mess of it.
// Some parts are more readable with long lines but rustfmt would split them up.

mod asm_compiler;
mod asm_encoding;

#[rustfmt::skip]
mod bytecode;

#[rustfmt::skip]
mod compiler;

#[rustfmt::skip]
mod feeny;

#[rustfmt::skip]
mod interpreter;

mod jit_memory;

#[rustfmt::skip]
mod parser;
