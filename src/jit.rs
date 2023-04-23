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

use std::{fmt::Write, path::PathBuf};

use anyhow::Result;

use crate::{
    bytecode::{
        interpreter::*,
        opcodes::OpCode::{self, *},
        program::*,
        state::{State, StdOutput},
    },
    jit::{asm_encoding::compile, memory::JitMemory},
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

pub fn jit_with_memory_config(program: &Program, heap_gc_size: Option<usize>, heap_log: Option<PathBuf>) -> Result<()> {
    let mut state = State::from(program)?;
    state.heap.set_gc_size(heap_gc_size);
    if let Some(log) = heap_log {
        state.heap.set_log(log);
    }
    let mut output = StdOutput::new();
    jit_with(program, &mut state, &mut output)
}

#[allow(unused_variables)] // TODO remove
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
    let f = jit_fn!(jit, fn());
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

pub fn is_jittable(program: &Program) -> bool {
    for opcode in program.code.iter() {
        match opcode {
            Literal { .. } => {}
            GetLocal { .. } => {},
            SetLocal { .. } => {},
            GetGlobal { .. } => {}
            SetGlobal { .. } => {}
            Object { .. } => {}
            Array => {}
            GetField { .. } => {}
            SetField { .. } => {}
            CallMethod { .. } => return false,
            CallFunction { .. } => return false,
            Label { .. } => {}
            Print { .. } => {}
            Jump { .. } => {}
            Branch { .. } => {}
            Return => return false,
            Drop => {}
        }
    }
    true
}

pub fn jit_program<W>(program: &Program, state: &mut State, output: &mut W)
where
    W: Write,
{
    //println!("jitting");

    // TODO Continue here: Add asm labels for functions, impl calls.

    use crate::jit::asm_repr::*;
    use Instr::*;
    use Reg::*;

    let mut instrs = Vec::new();
    instrs.push(Push(Rax)); // Align stack for calls
    // TODO keep program and state in non-volatile registers?
    #[allow(unused_variables)] // TODO remove
    for opcode in program.code.iter() {
        match opcode {
            OpCode::Literal { index } => {
                instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                instrs.push(MovRI(Rdx, index.ref_to_addr_const()));
                instrs.push(MovRI(Rax, fn_to_addr!(jit_literal)));
                instrs.push(CallAbsR(Rax));
            }
            OpCode::GetLocal { index } => {
                instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                instrs.push(MovRI(Rdx, index.ref_to_addr_const()));
                instrs.push(MovRI(Rax, fn_to_addr!(jit_get_local)));
                instrs.push(CallAbsR(Rax));
            }
            OpCode::SetLocal { index } => {
                instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                instrs.push(MovRI(Rdx, index.ref_to_addr_const()));
                instrs.push(MovRI(Rax, fn_to_addr!(jit_set_local)));
                instrs.push(CallAbsR(Rax));
            }
            OpCode::GetGlobal { name } => {
                instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                instrs.push(MovRI(Rdx, name.ref_to_addr_const()));
                instrs.push(MovRI(Rax, fn_to_addr!(jit_get_global)));
                instrs.push(CallAbsR(Rax));
            }
            OpCode::SetGlobal { name } => {
                instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                instrs.push(MovRI(Rdx, name.ref_to_addr_const()));
                instrs.push(MovRI(Rax, fn_to_addr!(jit_set_global)));
                instrs.push(CallAbsR(Rax));
            }
            OpCode::Object { class } => {
                instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                instrs.push(MovRI(Rdx, class.ref_to_addr_const()));
                instrs.push(MovRI(Rax, fn_to_addr!(jit_object)));
                instrs.push(CallAbsR(Rax));
            }
            OpCode::Array => {
                instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                instrs.push(MovRI(Rax, fn_to_addr!(jit_array)));
                instrs.push(CallAbsR(Rax));
            }
            OpCode::GetField { name } => {
                instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                instrs.push(MovRI(Rdx, name.ref_to_addr_const()));
                instrs.push(MovRI(Rax, fn_to_addr!(jit_get_field)));
                instrs.push(CallAbsR(Rax));
            }
            OpCode::SetField { name } => {
                instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                instrs.push(MovRI(Rdx, name.ref_to_addr_const()));
                instrs.push(MovRI(Rax, fn_to_addr!(jit_set_field)));
                instrs.push(CallAbsR(Rax));
            }
            OpCode::CallMethod { name, arity } => unimplemented!(),
            OpCode::CallFunction { name, arity } => unimplemented!(),
            OpCode::Label { name } => {
                // Intentionally first the label in assembly, then the call to jit_label
                // so that the instruction pointer is updated properly.
                // LATER(martin-t) Optimize?
                //  Interpreter doesn't need label instructions at all, remove when loading.
                //  Once all opcodes are jitted, we don't need the instruction pointer anymore.
                //  Also check other instructions.
                instrs.push(Label(name.value().into()));
                instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                instrs.push(MovRI(Rax, fn_to_addr!(jit_label)));
                instrs.push(CallAbsR(Rax));
            }
            OpCode::Print { format, arity } => {
                instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                instrs.push(MovRI(Rdx, output.ref_to_addr_mut()));
                instrs.push(MovRI(Rcx, format.ref_to_addr_const()));
                instrs.push(MovRI(R8, arity.ref_to_addr_const()));
                instrs.push(MovRI(Rax, fn_to_addr!(jit_print::<W>)));
                instrs.push(CallAbsR(Rax));
            }
            OpCode::Jump { label } => {
                instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                instrs.push(MovRI(Rdx, label.ref_to_addr_const()));
                instrs.push(MovRI(Rax, fn_to_addr!(jit_jump)));
                instrs.push(CallAbsR(Rax));
                instrs.push(JmpLabel(label.value().into()));
            }
            OpCode::Branch { label } => {
                instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                instrs.push(MovRI(Rdx, label.ref_to_addr_const()));
                instrs.push(MovRI(Rax, fn_to_addr!(jit_branch)));
                instrs.push(CallAbsR(Rax));
                instrs.push(CmpRI(Rax, 1));
                instrs.push(JeLabel(label.value().into()));
            }
            OpCode::Return => unimplemented!(),
            OpCode::Drop => {
                instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                instrs.push(MovRI(Rax, fn_to_addr!(jit_drop)));
                instrs.push(CallAbsR(Rax));
            }
        }
    }
    instrs.push(Pop(Rcx)); // Unalign stack
    instrs.push(Ret);

    let code = compile(&instrs).code;
    let jit = JitMemory::new(&code);
    let f = jit_fn!(jit, fn());
    f();
}

pub extern "sysv64" fn jit_literal(program: &Program, state: &mut State, literal_index: &ConstantPoolIndex) {
    eval_literal(program, state, literal_index).unwrap();
}

pub extern "sysv64" fn jit_get_local(program: &Program, state: &mut State, local_index: &LocalIndex) {
    eval_get_local(program, state, local_index).unwrap();
}

pub extern "sysv64" fn jit_set_local(program: &Program, state: &mut State, local_index: &LocalIndex) {
    eval_set_local(program, state, local_index).unwrap();
}

pub extern "sysv64" fn jit_get_global(program: &Program, state: &mut State, global_index: &ConstantPoolIndex) {
    eval_get_global(program, state, global_index).unwrap();
}

pub extern "sysv64" fn jit_set_global(program: &Program, state: &mut State, global_index: &ConstantPoolIndex) {
    eval_set_global(program, state, global_index).unwrap();
}

pub extern "sysv64" fn jit_object(program: &Program, state: &mut State, class_index: &ConstantPoolIndex) {
    eval_object(program, state, class_index).unwrap();
}

pub extern "sysv64" fn jit_array(program: &Program, state: &mut State) {
    eval_array(program, state).unwrap();
}

pub extern "sysv64" fn jit_get_field(program: &Program, state: &mut State, name_index: &ConstantPoolIndex) {
    eval_get_field(program, state, name_index).unwrap();
}

pub extern "sysv64" fn jit_set_field(program: &Program, state: &mut State, name_index: &ConstantPoolIndex) {
    eval_set_field(program, state, name_index).unwrap();
}

// pub extern "sysv64" fn jit_call_method(
//     program: &Program,
//     state: &mut State,
//     name_index: &ConstantPoolIndex,
//     arity: &Arity,
// ) {
//     eval_call_method(program, state, name_index, arity).unwrap();
// }

// pub extern "sysv64" fn jit_call_function(
//     program: &Program,
//     state: &mut State,
//     name_index: &ConstantPoolIndex,
//     arity: &Arity,
// ) {
//     eval_call_function(program, state, name_index, arity).unwrap();
// }

pub extern "sysv64" fn jit_print<W>(
    program: &Program,
    state: &mut State,
    output: &mut W,
    format: &ConstantPoolIndex,
    arity: &Arity,
) where
    W: Write,
{
    eval_print(program, state, output, format, arity).unwrap();
}

pub extern "sysv64" fn jit_label(program: &Program, state: &mut State) {
    eval_label(program, state).unwrap();
}

pub extern "sysv64" fn jit_jump(program: &Program, state: &mut State, label_index: &ConstantPoolIndex) {
    eval_jump(program, state, label_index).unwrap();
}

pub extern "sysv64" fn jit_branch(program: &Program, state: &mut State, label_index: &ConstantPoolIndex) -> bool {
    eval_branch(program, state, label_index).unwrap()
}

// pub extern "sysv64" fn jit_return(program: &Program, state: &mut State) {
//     eval_return(program, state).unwrap();
// }

pub extern "sysv64" fn jit_drop(program: &Program, state: &mut State) {
    eval_drop(program, state).unwrap();
}
