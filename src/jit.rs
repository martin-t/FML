// x64 calling conventions:
// - https://en.m.wikipedia.org/wiki/X86_calling_conventions#x86-64_calling_conventions
//
// Microsoft x64 calling convention:
// Return value: RAX if fits, otherwise dest pointer passed in RCX
// Args 1-4: RCX, RDX, R8, R9
// Caller-saved / volatile: RAX, RCX, RDX, R8-11
// Callee-saved / nonvolatile: RBX, RBP, RSP, RDI, RSI, R12-R15
// Caller must allocate 32 bytes of shadow space
//
// System V AMD64 ABI:
// Return value: RAX, RDX if fits, otherwise dest pointer passed in RDI
// Args 1-6: RDI, RSI, RDX, RCX, R8, R9
// Caller-saved / volatile: RAX, RCX, RDX, RDI, RSI, R8-R11
// Callee-saved / nonvolatile: RBX, RBP, RSP, R12–R15
// No shadow space
// Source: https://raw.githubusercontent.com/wiki/hjl-tools/x86-psABI/x86-64-psABI-1.0.pdf p22-4
//
// Common:
// Order of args on the stack: right to left
// Caller cleanup
//
// Jit uses the System V calling convention even on Windows for simplicity.
// It also has the advantage that 2 values can be returned in registers.

pub mod asm_encoding;
pub mod asm_repr;

#[cfg_attr(unix, path = "jit/memory_unix.rs")]
#[cfg_attr(windows, path = "jit/memory_windows.rs")]
pub mod memory;

use std::{fmt::Write, mem};

use fnv::FnvHashMap;

use crate::{
    bytecode::{heap::Pointer, interpreter::*, opcodes::OpCode, program::*, state::State},
    jit::{
        asm_encoding::compile,
        asm_repr::{Instr, Mem, Reg},
        memory::JitMemory,
    },
};

pub trait VariableAddr: Sized {
    /// Take the address of an lvalue.
    ///
    /// Do **NOT** use this on rvalues or functions.
    /// - For evalues, it'll return an address but it'll be useless
    ///   becuse it'll be invalid by the time you can use it.
    /// - Functions are already pointers so taking then by reference
    ///   also creates a temporary and therefore useless address.
    fn var_addr_mut(&mut self) -> i64 {
        self as *mut _ as i64
    }

    /// Similar to `var_addr_mut`.
    ///
    /// Intentionally named "const" and not just `take_addr`
    /// to avoid using it accidentally instead of `take_addr_mut`.
    fn var_addr_const(&self) -> i64 {
        self as *const _ as i64
    }
}

impl<T> VariableAddr for T {}

pub trait RefToAddrMut {
    /// Take the address of a reference.
    fn ref_to_addr_mut(self) -> i64;
}

impl<T> RefToAddrMut for &mut T {
    fn ref_to_addr_mut(self) -> i64 {
        self as *mut _ as i64
    }
}

pub trait RefToAddrConst {
    /// Take the address of a reference.
    fn ref_to_addr_const(self) -> i64;
}

impl<T> RefToAddrConst for &T {
    fn ref_to_addr_const(self) -> i64 {
        self as *const _ as i64
    }
}

/// Take the address of a function.
///
/// Use this for functions instead of `.var_addr_mut()`.
#[macro_export]
macro_rules! fn_to_addr {
    ($f:expr) => {
        // We could cast directly to i64
        // but outside a macro that triggers a clippy lint.
        $f as usize as i64
    };
}

/// Create a function pointer to jitted code.
///
/// Use this instead of raw transmute or casts to make sure
/// the returned fn has the correct calling convention
/// on all supported platforms.
///
/// By default this points to the beginning of JIT memory
/// but you can optionally specify an offset.
#[macro_export]
macro_rules! jit_fn {
    // There's no way to take the whole fn signature like `$f:ty`
    // and just put calling convention in front of it using something like
    // `extern "sysv64" $f:ty`.
    // So we have to match the individual parts instead,
    // even the optional return type.
    ( $jit:expr, fn( $($args:ty),* ) $( -> $ret:ty )? ) => {
        unsafe {
            type F = extern "sysv64" fn( $($args),* ) $( -> $ret )?;
            ::std::mem::transmute::<_, F>($jit.code)
        }
    };
    ( $jit:expr, fn( $($args:ty),* ) $( -> $ret:ty )?, $offset:expr ) => {
        unsafe {
            type F = extern "sysv64" fn( $($args),* ) $( -> $ret )?;
            ::std::mem::transmute::<_, F>($jit.code.add($offset))
        }
    }
}

#[derive(Debug)]
struct JitState {
    cpi_to_addr_opt: FnvHashMap<usize, usize>,
    cpi_to_addr_fallback: FnvHashMap<usize, usize>,
    label_counter: usize,
}

impl JitState {
    fn new() -> Self {
        Self {
            cpi_to_addr_opt: FnvHashMap::default(),
            cpi_to_addr_fallback: FnvHashMap::default(),
            label_counter: usize::MAX,
        }
    }

    fn next_label(&mut self) -> usize {
        self.label_counter -= 1;
        self.label_counter
    }
}

/// A very primitive JIT compiler which converts each opcode
/// into a series of assembly instructions that in turn call
/// the corresponding interpreter functions
/// through jit_<opcode> functions because Result is not FFI-safe.
///
/// This provides a small (~10%) speedup because it avoids the main
/// interpreter loop and branching on each opcode.
///
/// Jumps, function calls and returns call into the interpreter
/// and then also perform a jump/call/return in assembly.
/// This means the interpreter's instruction pointer is no longer necessary
/// which can provide an additional speedup (another ~10%).
// LATER(martin-t) Split into a struct and methods.
pub fn jit_program<W>(program: &Program, state: &mut State, output: &mut W)
where
    W: Write,
{
    use Instr::*;
    use Reg::*;

    let mut methods = Vec::new();
    for cpi in 0..program.constant_pool.len() {
        let index = ConstantPoolIndex::from(cpi);
        let program_object = program.constant_pool.get(index).unwrap();
        if let ProgramObject::Method(method) = program_object {
            methods.push((cpi, method));
        }
    }

    let entry_cpi = program.entry.get().unwrap().as_usize();

    let mut jit_state = JitState::new();
    let mut instrs = Vec::new();

    // For each function, attempt to compile it into pure assembly
    // without calling into the interpreter.
    // This only works for a limited set of opcodes,
    // basically only functions working with integers.
    //
    // Bail otherwisem, we'll use the interpreter as fallback.
    //
    // LATER(martin-t) Tests should make sure the optimized verrsion is actually used.
    let mut cpi_to_label_opt = FnvHashMap::default();
    // LATER(martin-t) This should be a separate function but Rust supports
    // break/continue with labels and i have 52 hours to write 50 pages
    // so i don't care anymore.
    'int_fn: for &(cpi, method) in &methods {
        if state.debug.contains(" opt-disable ") {
            // Useful for testing JIT without optimized versions of functions.
            break;
        }

        if cpi == entry_cpi {
            // The entry function is a special case, ignore for now.
            continue;
        }

        let mut is = Vec::new();

        let label_fn = jit_state.next_label();
        is.push(Label(label_fn));

        // LATER(martin-t) Don't repeat push/pop, subtract/add rsp once, then just mov.
        // LATER(martin-t) Use a register allocator instead of abusing the stack.
        // LATER(martin-t) We're using 64 bit regs even though FML only has 32 bit ints.
        // LATER(martin-t) In case we decide to impl calls, we need to align the stack.
        // LATER(martin-t) Handle more than 6 args, more opcodes, methods, ...
        // LATER(martin-t) Add Push/Pop instructions for memory/immediates to avoid mov+push.

        // Functioin prologue.
        is.push(Push(Rbp));
        is.push(MovRR(Rbp, Rsp));

        // Save arguments to the stack.
        let arity: i32 = method.arity.value().into();
        if arity >= 1 {
            is.push(Push(Rdi));
        }
        if arity >= 2 {
            is.push(Push(Rsi));
        }
        if arity >= 3 {
            is.push(Push(Rdx));
        }
        if arity >= 4 {
            is.push(Push(Rcx));
        }
        if arity >= 5 {
            is.push(Push(R8));
        }
        if arity >= 6 {
            is.push(Push(R9));
        }
        if arity >= 7 {
            continue 'int_fn;
        }

        // Allocate space for locals.
        let locals_size: i32 = method.locals.value().into();
        is.push(SubRI(Rsp, locals_size * 8));

        let mut branch_next = None;

        let begin = method.code.start().value_usize();
        let end = begin + method.code.length();
        for i in begin..end {
            let address = Address::from_usize(i);
            let opcode = program.code.get(address).unwrap();

            let address_next = Address::from_usize(i + 1);
            let opcode_next = program.code.get(address_next);

            if state.debug.contains(" opt-opcodes ") {
                println!("address: {:?}, opcode: {:?}", address, opcode);
            }

            #[allow(unused_variables)]
            match opcode {
                OpCode::Literal { index } => {
                    let lit = program.constant_pool.get(index).unwrap();
                    match lit {
                        ProgramObject::Integer(val) => {
                            is.push(MovRI(Rax, (*val).into()));
                            is.push(Push(Rax));
                        }
                        _ => continue 'int_fn,
                    }
                }
                OpCode::GetLocal { index } => {
                    let index: i32 = index.value().into();
                    if index >= arity + locals_size {
                        continue 'int_fn;
                    }

                    let offset = -(index + 1) * 8;
                    is.push(MovRM(Rax, Mem::base_offset(Rbp, offset)));
                    is.push(Push(Rax));
                }
                OpCode::SetLocal { index } => {
                    let index: i32 = index.value().into();
                    if index >= arity + locals_size {
                        continue 'int_fn;
                    }

                    let offset = -(index + 1) * 8;
                    // is.push(Pop(Rax));
                    // is.push(Push(Rax));
                    is.push(MovRM(Rax, Mem::base(Rsp)));
                    is.push(MovMR(Mem::base_offset(Rbp, offset), Rax));
                }
                OpCode::GetGlobal { name } => continue 'int_fn,
                OpCode::SetGlobal { name } => continue 'int_fn,
                OpCode::Object { class } => continue 'int_fn,
                OpCode::Array => continue 'int_fn,
                OpCode::GetField { name } => continue 'int_fn,
                OpCode::SetField { name } => continue 'int_fn,
                OpCode::CallMethod { name, arity } => {
                    let name = program.constant_pool.get(name).unwrap();
                    let name = name.as_str().unwrap();

                    if arity.value() != 2 {
                        continue 'int_fn;
                    }

                    // LATER(martin-t) Massive HACK to make collatz.fml work.
                    // Supporting booleans properly requires
                    // 1) assembler support for 1 byte registers and more instructions
                    // 2) type analysis or tagging each value on the stack with its type
                    let is_bool_op = matches!(name, "==" | "!=" | "<" | ">" | "<=" | ">=");
                    let is_next_branch = matches!(opcode_next, Ok(OpCode::Branch { .. }));
                    if is_bool_op && !is_next_branch {
                        continue 'int_fn;
                    }

                    // Left operand is in Rax, right is in Rcx.
                    is.push(Pop(Rcx));
                    is.push(Pop(Rax));
                    match name {
                        "+" => {
                            is.push(AddRR(Rax, Rcx));
                            is.push(Push(Rax));
                        }
                        "-" => {
                            is.push(SubRR(Rax, Rcx));
                            is.push(Push(Rax));
                        }
                        "*" => {
                            is.push(ImulRR(Rax, Rcx));
                            is.push(Push(Rax));
                        }
                        "/" => {
                            is.push(Cqo);
                            is.push(IdivR(Rcx));
                            is.push(Push(Rax));
                        }
                        "%" => {
                            is.push(Cqo);
                            is.push(IdivR(Rcx));
                            is.push(MovRR(Rax, Rdx));
                            is.push(Push(Rax));
                        }
                        "<=" => {
                            is.push(CmpRR(Rax, Rcx));
                            branch_next = Some(JleLabel(666));
                        }
                        "<" => {
                            is.push(CmpRR(Rax, Rcx));
                            branch_next = Some(JlLabel(666));
                        }
                        ">=" => {
                            is.push(CmpRR(Rax, Rcx));
                            branch_next = Some(JgeLabel(666));
                        }
                        ">" => {
                            is.push(CmpRR(Rax, Rcx));
                            branch_next = Some(JgLabel(666));
                        }
                        "==" => {
                            is.push(CmpRR(Rax, Rcx));
                            branch_next = Some(JeLabel(666));
                        }
                        "!=" => {
                            is.push(CmpRR(Rax, Rcx));
                            branch_next = Some(JneLabel(666));
                        }

                        _ => continue 'int_fn,
                    }
                }
                OpCode::CallFunction { name, arity } => continue 'int_fn,
                OpCode::Label { name } => {
                    // Make sure the label doesn't collide with others.
                    // LATER(martin-t) This is a HACK.
                    let label = usize::MAX / 2 + name.as_usize();
                    is.push(Label(label));
                }
                OpCode::Print { format, arity } => continue 'int_fn,
                OpCode::Jump { label } => {
                    let label = usize::MAX / 2 + label.as_usize();
                    is.push(JmpLabel(label));
                }
                OpCode::Branch { label } => {
                    let label = usize::MAX / 2 + label.as_usize();
                    // LATER(martin-t) Massive HACK to make collatz.fml work.
                    match branch_next.unwrap() {
                        JleLabel(_) => is.push(JleLabel(label)),
                        JlLabel(_) => is.push(JlLabel(label)),
                        JgeLabel(_) => is.push(JgeLabel(label)),
                        JgLabel(_) => is.push(JgLabel(label)),
                        JeLabel(_) => is.push(JeLabel(label)),
                        JneLabel(_) => is.push(JneLabel(label)),
                        _ => unreachable!(),
                    }
                }
                OpCode::Return => {
                    // Get the return value from the "operand stack".
                    is.push(Pop(Rax));

                    // Deallocate space for locals and arguments.
                    let locals_size: i32 = method.locals.value().into();
                    is.push(AddRI(Rsp, (arity + locals_size) * 8));

                    // Function epilogue.
                    is.push(Pop(Rbp));

                    is.push(Ret);
                }
                OpCode::Drop => {
                    is.push(Pop(Rax));
                }
            }
        }

        if state.debug.contains(" opt-instrs ") {
            for instr in &is {
                eprintln!("{:x}", instr);
            }
        }

        // If we got here, the function is simple enough
        // to be optimized into pure assembly.
        instrs.extend(is);
        cpi_to_label_opt.insert(cpi, label_fn);
    }

    // For each function, compile it into assembly opcode by opcode.
    // Call into the interpreter for most opcodes to do the real work.
    for &(cpi, method) in &methods {
        // Use the index of the method itself because it's unique,
        // not of its name because those get reused.
        instrs.push(Label(cpi));

        if cpi == entry_cpi {
            // Save these nonvolatile registers so we can use them
            // to store data for jit_<opcode> functions.
            // Since any Rust functions we call have to restore them
            // before returning, they will always have the same values
            // in jitted code as long as we call all other jitted FML functions
            // directly without going through Rust.
            // Therefore we don't need to pass them as arguments
            // into other jitted FML functions.
            //
            // This also aligns the stack.
            instrs.push(Push(R12));
            instrs.push(Push(R13));
            instrs.push(Push(R14));
            instrs.push(Push(R15));
            // Dummy to align stack
            instrs.push(Push(Rax));
            // ^ Don't forget to update epilogue when changing this.

            // Now save the arguments.
            // We could use program/state/... as immediates
            // like `MovRI(Rdi, program.ref_to_addr_const())`
            // but passing them into entry as arguments
            // might play a little nicer with Rust's memory model.
            //
            // E.g. if we mutably borrow state here
            // and turn the reference into a pointer and then an integer,
            // it is stil technically borrowed in Rust's stacked borrows model.
            // If we them use state before calling entry, it could be UB.
            //
            // However, this is mostly theoretical since rustc doesn't
            // use the stacked borrows model (for optimizations) yet.
            instrs.push(MovRR(R12, Rdi));
            instrs.push(MovRR(R13, Rsi));
            instrs.push(MovRR(R14, Rdx));
            instrs.push(MovRR(R15, Rcx));

            // When adding here, make sure the stack stays aligned.
        } else {
            // Even though the FML methods can take arguments,
            // the jitted fn representing them doesn't
            // so the type is always just `fn()`.
            // Arguments and return values are handled by the interpreter's stack.

            // Align stack
            instrs.push(Push(Rax));
            // ^ Don't forget to update epilogue when changing this.
        }

        let begin = method.code.start().value_usize();
        let end = begin + method.code.length();
        for i in begin..end {
            let address = Address::from_usize(i);
            let opcode = program.code.get(address).unwrap();

            // This must be before the opcode's instructions
            // because they might include a jump and this would never run.
            if state.debug.contains(" ds ") {
                instrs.push(MovRI(Rdi, state.ref_to_addr_mut()));
                instrs.push(MovRI(Rax, fn_to_addr!(debug_state)));
                instrs.push(CallAbsR(Rax));
            }

            match opcode {
                OpCode::Literal { index } => {
                    instrs.push(MovRR(Rdi, R12));
                    instrs.push(MovRR(Rsi, R13));
                    instrs.push(MovRI(Rdx, index.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_literal)));
                    instrs.push(CallAbsR(Rax));
                }
                OpCode::GetLocal { index } => {
                    instrs.push(MovRR(Rdi, R13));
                    instrs.push(MovRI(Rsi, index.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_get_local)));
                    instrs.push(CallAbsR(Rax));
                }
                OpCode::SetLocal { index } => {
                    instrs.push(MovRR(Rdi, R13));
                    instrs.push(MovRI(Rsi, index.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_set_local)));
                    instrs.push(CallAbsR(Rax));
                }
                OpCode::GetGlobal { name } => {
                    instrs.push(MovRR(Rdi, R12));
                    instrs.push(MovRR(Rsi, R13));
                    instrs.push(MovRI(Rdx, name.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_get_global)));
                    instrs.push(CallAbsR(Rax));
                }
                OpCode::SetGlobal { name } => {
                    instrs.push(MovRR(Rdi, R12));
                    instrs.push(MovRR(Rsi, R13));
                    instrs.push(MovRI(Rdx, name.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_set_global)));
                    instrs.push(CallAbsR(Rax));
                }
                OpCode::Object { class } => {
                    instrs.push(MovRR(Rdi, R12));
                    instrs.push(MovRR(Rsi, R13));
                    instrs.push(MovRI(Rdx, class.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_object)));
                    instrs.push(CallAbsR(Rax));
                }
                OpCode::Array => {
                    instrs.push(MovRR(Rdi, R13));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_array)));
                    instrs.push(CallAbsR(Rax));
                }
                OpCode::GetField { name } => {
                    instrs.push(MovRR(Rdi, R12));
                    instrs.push(MovRR(Rsi, R13));
                    instrs.push(MovRI(Rdx, name.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_get_field)));
                    instrs.push(CallAbsR(Rax));
                }
                OpCode::SetField { name } => {
                    instrs.push(MovRR(Rdi, R12));
                    instrs.push(MovRR(Rsi, R13));
                    instrs.push(MovRI(Rdx, name.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_set_field)));
                    instrs.push(CallAbsR(Rax));
                }
                OpCode::CallMethod { name, arity } => {
                    if state.debug.contains(" cmud ") {
                        instrs.push(Ud2);
                    }
                    instrs.push(MovRR(Rdi, R12));
                    instrs.push(MovRR(Rsi, R13));
                    instrs.push(MovRI(Rdx, name.value().into()));
                    instrs.push(MovRI(Rcx, arity.value().into()));
                    instrs.push(MovRR(R8, R15));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_call_method)));
                    // Call this interpreter function
                    // to determine the offset of the FML function.
                    instrs.push(CallAbsR(Rax));
                    // Check if the function is a builtin.
                    instrs.push(CmpRI(Rax, 0));
                    let label_skip = jit_state.next_label();
                    instrs.push(JeLabel(label_skip));
                    if state.debug.contains(" cmud2 ") {
                        // Only crash if it's a user-defined method.
                        instrs.push(Ud2);
                    }
                    // Now call the actual FML function.
                    instrs.push(CallAbsR(Rax));
                    instrs.push(Label(label_skip));
                }
                OpCode::CallFunction { name, arity } => {
                    if state.debug.contains(" cfud ") {
                        instrs.push(Ud2);
                    }
                    instrs.push(MovRR(Rdi, R12));
                    instrs.push(MovRR(Rsi, R13));
                    instrs.push(MovRI(Rdx, name.value().into()));
                    instrs.push(MovRI(Rcx, arity.value().into()));
                    instrs.push(MovRR(R8, R15));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_call_function)));
                    // Call this interpreter function
                    // to determine the offset of the FML function.
                    instrs.push(CallAbsR(Rax));
                    // Check if the function was jit-optimized.
                    instrs.push(CmpRI(Rax, 0));
                    let label_skip = jit_state.next_label();
                    instrs.push(JeLabel(label_skip));
                    if state.debug.contains(" cfud2 ") {
                        // Only crash if it's a jit-optimized function.
                        instrs.push(Ud2);
                    }
                    // Now call the actual FML function.
                    instrs.push(CallAbsR(Rax));
                    instrs.push(Label(label_skip));
                }
                OpCode::Label { name } => {
                    // If we wanted state.instruction_pointer
                    // to always match what the interpreter would do,
                    // we'd call jit_label here after the label instruction.
                    //
                    // The order of instructions would be important.
                    // The asm label should be before calling jit_label,
                    // to match behavior of the bytecode interpreter.
                    //
                    // However if all opcodes are jitted,
                    // we don't need the instruction pointer anymore.
                    // So we can put if after jit_label or better yet
                    // we don't need to call jit_label at all.
                    // We can then also stop bumping instruction pointer
                    // in the eval_* functions.
                    //
                    // Currently this provides only a small speedup
                    // probably because the interpreter
                    // does a lot of other slow things.

                    instrs.push(Label(name.value().into()));
                }
                OpCode::Print { format, arity } => {
                    instrs.push(MovRR(Rdi, R12));
                    instrs.push(MovRR(Rsi, R13));
                    instrs.push(MovRR(Rdx, R14));
                    instrs.push(MovRI(Rcx, format.value().into()));
                    instrs.push(MovRI(R8, arity.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_print::<W>)));
                    instrs.push(CallAbsR(Rax));
                }
                OpCode::Jump { label } => {
                    instrs.push(MovRR(Rdi, R12));
                    instrs.push(MovRI(Rsi, label.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_jump)));
                    instrs.push(CallAbsR(Rax));
                    instrs.push(JmpLabel(label.value().into()));
                }
                OpCode::Branch { label } => {
                    instrs.push(MovRR(Rdi, R12));
                    instrs.push(MovRR(Rsi, R13));
                    instrs.push(MovRI(Rdx, label.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_branch)));
                    instrs.push(CallAbsR(Rax));
                    instrs.push(CmpRI(Eax, 1));
                    instrs.push(JeLabel(label.value().into()));
                }
                OpCode::Return => {
                    instrs.push(MovRR(Rdi, R13));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_return)));
                    instrs.push(CallAbsR(Rax));

                    assert_ne!(cpi, entry_cpi, "entry function should not have Return opcode");
                    instrs.push(Pop(Rcx)); // Unalign stack
                    instrs.push(Ret);
                }
                OpCode::Drop => {
                    instrs.push(MovRR(Rdi, R13));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_drop)));
                    instrs.push(CallAbsR(Rax));
                }
            }
        }

        // This epilogue is only needed for the entry function,
        // all other functions end with a Return opcode.
        if cpi == entry_cpi {
            if state.debug.contains(" ds ") {
                instrs.push(MovRI(Rdi, state.ref_to_addr_mut()));
                instrs.push(MovRI(Rax, fn_to_addr!(debug_state)));
                instrs.push(CallAbsR(Rax));
            }

            // Restore saved registers (and unalign stack)
            instrs.push(Pop(Rcx));
            instrs.push(Pop(R15));
            instrs.push(Pop(R14));
            instrs.push(Pop(R13));
            instrs.push(Pop(R12));
            // The entry function doesn't have a Return opcode in bytecode
            // but it needs a return in assembly so we get out of it.
            instrs.push(Ret);
        }

        // Assert every jitted FML function ends with a return.
        assert_eq!(instrs.last(), Some(&Ret));
    }

    if state.debug.contains(" instrs ") {
        for instr in &instrs {
            eprintln!("{:x}", instr);
        }
    }

    let compiled = compile(&instrs);
    if state.debug.contains(" compiled ") {
        asm_encoding::eprint_asm(&compiled.code);
    }
    let jit = JitMemory::new(&compiled.code);

    for &(cpi, _) in &methods {
        if cpi == entry_cpi {
            // We can't call the entry function directly
            // so don't even save its offset.
            continue;
        }

        let offset_fallback = compiled.label_offsets[&cpi];
        let addr_fallback = jit.code as usize + offset_fallback;
        assert_ne!(addr_fallback, 0);
        jit_state.cpi_to_addr_fallback.insert(cpi, addr_fallback);
        if state.debug.contains(" offsets-jit ") {
            eprintln!("fallback cpi: {cpi}, offset: {offset_fallback}, addr: {addr_fallback:#x}");
        }

        if let Some(&label_fn_opt) = cpi_to_label_opt.get(&cpi) {
            let offset_opt = compiled.label_offsets[&label_fn_opt];
            let addr_opt = jit.code as usize + offset_opt;
            assert_ne!(addr_opt, 0);
            jit_state.cpi_to_addr_opt.insert(cpi, addr_opt);
            if state.debug.contains(" offsets-jit ") {
                eprintln!("opt cpi: {cpi}, offset: {offset_opt}, addr: {addr_opt:#x}");
            }
        }
    }

    let entry_offset = compiled.label_offsets[&entry_cpi];
    let entry = jit_fn!(jit, fn(i64, i64, i64, i64), entry_offset);

    entry(
        program.ref_to_addr_const(),
        state.ref_to_addr_mut(),
        output.ref_to_addr_mut(),
        jit_state.var_addr_const(),
    );
}

#[allow(dead_code)]
extern "sysv64" fn jit_step_with<W>(program: &Program, state: &mut State, output: &mut W) -> i32
where
    W: Write,
{
    if let Some(_address) = state.instruction_pointer.get() {
        step_with(program, state, output).unwrap();
        1
    } else {
        0
    }
}

/// This and the other jit_<opcode> functions stand between
/// the jitted code and the interpreter's eval_<opcode> functions.
///
/// This is necessary because the jitted code can't call into Rust directly.
/// Rust fns have an unstable/unspecified calling convention
/// and some types used by eval_<opcode> such as Result are not FFI-safe.
///
/// We also _should_ to catch panics here because unwinding through
/// an FFI boundary is very likely undefined behavior.
/// This is not certain - the Rustonomicon recently changed
/// with the introduction of *-unwind ABI's (which are still unstable)
/// and the wording is a bit unclear.
///
/// Unfortunately `&mut State` is not UnwindSave so we can't use catch_unwind.
/// So for now we use panic = "abort" in Cargo.toml.
/// This is not ideal because it means we still have UB in test and bench
/// profiles which don't support panic = "abort".
// LATER(martin-t) Find a way to catch panics here.
extern "sysv64" fn jit_literal(program: &Program, state: &mut State, literal_index: ConstantPoolIndex) {
    eval_literal(program, state, literal_index).unwrap();
}

extern "sysv64" fn jit_get_local(state: &mut State, local_index: LocalIndex) {
    eval_get_local(state, local_index).unwrap();
}

extern "sysv64" fn jit_set_local(state: &mut State, local_index: LocalIndex) {
    eval_set_local(state, local_index).unwrap();
}

extern "sysv64" fn jit_get_global(program: &Program, state: &mut State, global_index: ConstantPoolIndex) {
    eval_get_global(program, state, global_index).unwrap();
}

extern "sysv64" fn jit_set_global(program: &Program, state: &mut State, global_index: ConstantPoolIndex) {
    eval_set_global(program, state, global_index).unwrap();
}

extern "sysv64" fn jit_object(program: &Program, state: &mut State, class_index: ConstantPoolIndex) {
    eval_object(program, state, class_index).unwrap();
}

extern "sysv64" fn jit_array(state: &mut State) {
    eval_array(state).unwrap();
}

extern "sysv64" fn jit_get_field(program: &Program, state: &mut State, name_index: ConstantPoolIndex) {
    eval_get_field(program, state, name_index).unwrap();
}

extern "sysv64" fn jit_set_field(program: &Program, state: &mut State, name_index: ConstantPoolIndex) {
    eval_set_field(program, state, name_index).unwrap();
}

extern "sysv64" fn jit_call_method(
    program: &Program,
    state: &mut State,
    name_index: ConstantPoolIndex,
    arity: Arity,
    jit_state: &JitState,
) -> usize {
    // LATER(martin-t) Make a map which converts opcode address to asm addr
    //  so that eval_call_* functions don't have to return a tuple.

    let (method_index, _) = eval_call_method(program, state, name_index, arity).unwrap();

    if let Some(method_index) = method_index {
        let addr = jit_state.cpi_to_addr_fallback[&method_index.as_usize()];
        if state.debug.contains(" offsets-call ") {
            eprintln!("returning method cpi: {method_index}, addr: {addr:#x}");
        }
        addr
    } else {
        0
    }
}

extern "sysv64" fn jit_call_function(
    program: &Program,
    state: &mut State,
    name_index: ConstantPoolIndex,
    arity: Arity,
    jit_state: &JitState,
) -> usize {
    // The goal here is to check if the functiont to be called
    // has an optimized jitted version
    // and if that version can be used (e.g. if the arguments are all integers).
    // However, this is supposed to be a cheap check
    // so it actually pays off performance-wise.
    // LATER(martin-t) Don't call eval_call_function here, just check the types.
    let (method_index, _) = eval_call_function(program, state, name_index, arity).unwrap();

    let arity = arity.to_usize();
    if let Some(&addr_opt) = jit_state.cpi_to_addr_opt.get(&method_index.as_usize()) {
        let arguments = &state.frame_stack.frames().last().unwrap().locals()[0..arity];
        // LATER(martin-t) Handle any number of arguments.
        //  This cannot be done generically in Rust.
        //  Either only support a fixed number or do this in assembly.
        if arity <= 6 && arguments.iter().all(|local| local.as_i32().is_ok()) {
            if state.debug.contains(" offsets-call ") {
                eprintln!("using optimized function cpi: {method_index}, addr_opt: {addr_opt:#x}");
            }
            let ret = match arity {
                0 => {
                    let f: extern "sysv64" fn() -> i32 = unsafe { mem::transmute(addr_opt) };
                    let ret = f();
                    Pointer::Integer(ret)
                }
                1 => {
                    let f: extern "sysv64" fn(i32) -> i32 = unsafe { mem::transmute(addr_opt) };
                    let arg = arguments[0].as_i32().unwrap();
                    let ret = f(arg);
                    Pointer::Integer(ret)
                }
                2 => {
                    let f: extern "sysv64" fn(i32, i32) -> i32 = unsafe { mem::transmute(addr_opt) };
                    let arg1 = arguments[0].as_i32().unwrap();
                    let arg2 = arguments[1].as_i32().unwrap();
                    let ret = f(arg1, arg2);
                    Pointer::Integer(ret)
                }
                3 => {
                    let f: extern "sysv64" fn(i32, i32, i32) -> i32 = unsafe { mem::transmute(addr_opt) };
                    let arg1 = arguments[0].as_i32().unwrap();
                    let arg2 = arguments[1].as_i32().unwrap();
                    let arg3 = arguments[2].as_i32().unwrap();
                    let ret = f(arg1, arg2, arg3);
                    Pointer::Integer(ret)
                }
                4 => {
                    let f: extern "sysv64" fn(i32, i32, i32, i32) -> i32 = unsafe { mem::transmute(addr_opt) };
                    let arg1 = arguments[0].as_i32().unwrap();
                    let arg2 = arguments[1].as_i32().unwrap();
                    let arg3 = arguments[2].as_i32().unwrap();
                    let arg4 = arguments[3].as_i32().unwrap();
                    let ret = f(arg1, arg2, arg3, arg4);
                    Pointer::Integer(ret)
                }
                5 => {
                    let f: extern "sysv64" fn(i32, i32, i32, i32, i32) -> i32 = unsafe { mem::transmute(addr_opt) };
                    let arg1 = arguments[0].as_i32().unwrap();
                    let arg2 = arguments[1].as_i32().unwrap();
                    let arg3 = arguments[2].as_i32().unwrap();
                    let arg4 = arguments[3].as_i32().unwrap();
                    let arg5 = arguments[4].as_i32().unwrap();
                    let ret = f(arg1, arg2, arg3, arg4, arg5);
                    Pointer::Integer(ret)
                }
                6 => {
                    let f: extern "sysv64" fn(i32, i32, i32, i32, i32, i32) -> i32 =
                        unsafe { mem::transmute(addr_opt) };
                    let arg1 = arguments[0].as_i32().unwrap();
                    let arg2 = arguments[1].as_i32().unwrap();
                    let arg3 = arguments[2].as_i32().unwrap();
                    let arg4 = arguments[3].as_i32().unwrap();
                    let arg5 = arguments[4].as_i32().unwrap();
                    let arg6 = arguments[5].as_i32().unwrap();
                    let ret = f(arg1, arg2, arg3, arg4, arg5, arg6);
                    Pointer::Integer(ret)
                }
                _ => unreachable!(),
            };
            state.operand_stack.push(ret);
            eval_return(state).unwrap();
            return 0;
        }
    }

    let addr = jit_state.cpi_to_addr_fallback[&method_index.as_usize()];
    if state.debug.contains(" offsets-call ") {
        eprintln!("returning fallback function cpi: {method_index}, addr: {addr:#x}");
    }
    addr
}

extern "sysv64" fn jit_print<W>(
    program: &Program,
    state: &mut State,
    output: &mut W,
    format: ConstantPoolIndex,
    arity: Arity,
) where
    W: Write,
{
    eval_print(program, state, output, format, arity).unwrap();
}

extern "sysv64" fn jit_jump(program: &Program, label_index: ConstantPoolIndex) {
    eval_jump(program, label_index).unwrap();
}

/// This has to return an i32 so we can read it through EAX.
/// Returning a bool would only set the low byte of EAX
/// (the AL register) and the rest would be undefined.
///
/// Our limited assembler doesn't support reading AL, only EAX.
extern "sysv64" fn jit_branch(program: &Program, state: &mut State, label_index: ConstantPoolIndex) -> i32 {
    eval_branch(program, state, label_index).unwrap().is_some().into()
}

extern "sysv64" fn jit_return(state: &mut State) {
    eval_return(state).unwrap();
}

extern "sysv64" fn jit_drop(state: &mut State) {
    eval_drop(state).unwrap();
}
