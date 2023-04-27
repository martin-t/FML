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

use std::fmt::Write;

use fnv::FnvHashMap;

use crate::{
    bytecode::{
        interpreter::*,
        opcodes::OpCode::{self, *},
        program::*,
        state::State,
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

#[allow(dead_code)] // TODO remove
pub fn is_jittable(program: &Program) -> bool {
    for opcode in program.code.iter() {
        match opcode {
            Literal { .. } => {}
            GetLocal { .. } => {}
            SetLocal { .. } => {}
            GetGlobal { .. } => {}
            SetGlobal { .. } => {}
            Object { .. } => {}
            Array => {}
            GetField { .. } => {}
            SetField { .. } => {}
            CallMethod { .. } => {}
            CallFunction { .. } => {}
            Label { .. } => {}
            Print { .. } => {}
            Jump { .. } => {}
            Branch { .. } => {}
            Return => {}
            Drop => {}
        }
    }
    true
}

pub fn jit_program<W>(program: &Program, state: &mut State, output: &mut W)
where
    W: Write,
{
    use crate::jit::asm_repr::*;
    use Instr::*;
    use Reg::*;

    //println!("jitting");

    let mut methods = Vec::new();
    for cpi in 0..program.constant_pool.len() {
        let index = ConstantPoolIndex::from(cpi);
        let program_object = program.constant_pool.get(index).unwrap();
        if let ProgramObject::Method(method) = program_object {
            methods.push((cpi, method));
        }
    }

    let entry_cpi = program.entry.get().unwrap().as_usize();

    let mut label_counter = usize::MAX;
    let mut next_label = || {
        label_counter -= 1;
        label_counter
    };

    let mut instrs = Vec::new();
    for &(cpi, method) in &methods {
        // Use the index of the method itself because it's unique,
        // not of its name because those get reused.
        instrs.push(Label(cpi));

        if cpi == entry_cpi {
            // Save R15 so we an use it to store data for jit functions.
            // This also aligns the stack.
            instrs.push(Push(R15));
            instrs.push(MovRR(R15, Rdi));

            // When adding here, make sure the stack stays aligned.
        } else {
            instrs.push(Push(Rax)); // Align stack
        }

        // TODO keep program and state in non-volatile registers?

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
                    instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                    instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                    instrs.push(MovRI(Rdx, index.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_literal)));
                    instrs.push(CallAbsR(Rax));
                }
                OpCode::GetLocal { index } => {
                    instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                    instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                    instrs.push(MovRI(Rdx, index.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_get_local)));
                    instrs.push(CallAbsR(Rax));
                }
                OpCode::SetLocal { index } => {
                    instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                    instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                    instrs.push(MovRI(Rdx, index.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_set_local)));
                    instrs.push(CallAbsR(Rax));
                }
                OpCode::GetGlobal { name } => {
                    instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                    instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                    instrs.push(MovRI(Rdx, name.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_get_global)));
                    instrs.push(CallAbsR(Rax));
                }
                OpCode::SetGlobal { name } => {
                    instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                    instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                    instrs.push(MovRI(Rdx, name.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_set_global)));
                    instrs.push(CallAbsR(Rax));
                }
                OpCode::Object { class } => {
                    instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                    instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                    instrs.push(MovRI(Rdx, class.value().into()));
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
                    instrs.push(MovRI(Rdx, name.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_get_field)));
                    instrs.push(CallAbsR(Rax));
                }
                OpCode::SetField { name } => {
                    instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                    instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                    instrs.push(MovRI(Rdx, name.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_set_field)));
                    instrs.push(CallAbsR(Rax));
                }
                OpCode::CallMethod { name, arity } => {
                    if state.debug.contains(" cmud ") {
                        instrs.push(Ud2);
                    }
                    instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                    instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                    instrs.push(MovRI(Rdx, name.value().into()));
                    instrs.push(MovRI(Rcx, arity.value().into()));
                    instrs.push(MovRR(R8, R15));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_call_method)));
                    // Call this interpreter function
                    // to determine the offset of the FML function.
                    instrs.push(CallAbsR(Rax));
                    // Check if the function is a builtin.
                    instrs.push(CmpRI(Rax, 0));
                    let label = next_label();
                    instrs.push(JeLabel(label));
                    if state.debug.contains(" cmud2 ") {
                        // Only crash if it's a user-defined method.
                        instrs.push(Ud2);
                    }
                    // Now call the actual FML function.
                    instrs.push(CallAbsR(Rax));
                    instrs.push(Label(label));
                }
                OpCode::CallFunction { name, arity } => {
                    if state.debug.contains(" cfud ") {
                        instrs.push(Ud2);
                    }
                    instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                    instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                    instrs.push(MovRI(Rdx, name.value().into()));
                    instrs.push(MovRI(Rcx, arity.value().into()));
                    instrs.push(MovRR(R8, R15));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_call_function)));
                    // Call this interpreter function
                    // to determine the offset of the FML function.
                    instrs.push(CallAbsR(Rax));
                    // Now call the actual FML function.
                    instrs.push(CallAbsR(Rax));
                }
                OpCode::Label { name } => {
                    // Intentionally first the label in assembly, then the call to jit_label
                    // so that the instruction pointer is updated properly.
                    // LATER(martin-t) Optimize?
                    //  Interpreter doesn't need label instructions at all, remove when loading.
                    //  Once all opcodes are jitted, we don't need the instruction pointer anymore.
                    //  Also check other instructions.
                    // FIXME Why does position of label have no effect? Make simple test program?
                    // FIXME But first check notes, no fn calls necessary to get UB.
                    if !state.debug.contains(" label_end ") {
                        instrs.push(Label(name.value().into()));
                    }
                    instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                    instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_label)));
                    instrs.push(CallAbsR(Rax));
                    if state.debug.contains(" label_end ") {
                        instrs.push(Label(name.value().into()));
                    }
                }
                OpCode::Print { format, arity } => {
                    instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                    instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                    instrs.push(MovRI(Rdx, output.ref_to_addr_mut()));
                    instrs.push(MovRI(Rcx, format.value().into()));
                    instrs.push(MovRI(R8, arity.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_print::<W>)));
                    instrs.push(CallAbsR(Rax));
                }
                OpCode::Jump { label } => {
                    instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                    instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                    instrs.push(MovRI(Rdx, label.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_jump)));
                    instrs.push(CallAbsR(Rax));
                    instrs.push(JmpLabel(label.value().into()));
                }
                OpCode::Branch { label } => {
                    instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                    instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                    instrs.push(MovRI(Rdx, label.value().into()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_branch)));
                    instrs.push(CallAbsR(Rax));
                    instrs.push(CmpRI(Eax, 1));
                    instrs.push(JeLabel(label.value().into()));
                }
                OpCode::Return => {
                    instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                    instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
                    instrs.push(MovRI(Rax, fn_to_addr!(jit_return)));
                    instrs.push(CallAbsR(Rax));

                    assert_ne!(cpi, entry_cpi, "entry function should not have Return opcode");
                    instrs.push(Pop(Rcx)); // Unalign stack
                    instrs.push(Ret);
                }
                OpCode::Drop => {
                    instrs.push(MovRI(Rdi, program.ref_to_addr_const()));
                    instrs.push(MovRI(Rsi, state.ref_to_addr_mut()));
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
            instrs.push(Pop(R15));
            // The entry function doesn't have a Return opcode in bytecode
            // but it needs a return in assembly so we get out of it.
            instrs.push(Ret);
        }

        // Assert every jit function ends with a return.
        assert_eq!(instrs.last(), Some(&Ret));
    }

    if state.debug.contains(" instrs ") {
        for instr in &instrs {
            println!("{:x}", instr);
        }
    }

    let compiled = compile(&instrs);
    // asm_encoding::print_asm(&compiled.code);
    let jit = JitMemory::new(&compiled.code);

    let mut cpi_to_fn = FnvHashMap::default();
    for &(cpi, _) in &methods {
        if cpi == entry_cpi {
            continue;
        }

        let offset = compiled.label_offsets[&cpi];
        // Even though the FML methods can take arguments,
        // the jitted fn representing them doesn't so the type is always just `fn()`.
        // Arguments and return values are handled by the interpreter's stack.
        let fn_ptr = jit_fn!(jit, fn(), offset);
        // println!("cpi: {cpi}, offset: {offset}, addr: {:#x}", fn_to_addr!(fn_ptr));
        cpi_to_fn.insert(cpi, fn_ptr);
    }

    let entry_offset = compiled.label_offsets[&entry_cpi];
    let entry = jit_fn!(jit, fn(i64), entry_offset);

    entry(cpi_to_fn.var_addr_const());
}

#[allow(dead_code)]
extern "sysv64" fn jit_step_with<W>(program: &Program, state: &mut State, output: &mut W) -> bool
where
    W: Write,
{
    if let Some(_address) = state.instruction_pointer.get() {
        step_with(program, state, output).unwrap();
        true
    } else {
        false
    }
}

extern "sysv64" fn jit_literal(program: &Program, state: &mut State, literal_index: ConstantPoolIndex) {
    eval_literal(program, state, literal_index).unwrap();
}

extern "sysv64" fn jit_get_local(program: &Program, state: &mut State, local_index: LocalIndex) {
    eval_get_local(program, state, local_index).unwrap();
}

extern "sysv64" fn jit_set_local(program: &Program, state: &mut State, local_index: LocalIndex) {
    eval_set_local(program, state, local_index).unwrap();
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

extern "sysv64" fn jit_array(program: &Program, state: &mut State) {
    eval_array(program, state).unwrap();
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
    cpi_to_fn: &FnvHashMap<usize, fn()>,
) -> i64 {
    let method_index = eval_call_method(program, state, name_index, arity).unwrap();
    if let Some(method_index) = method_index {
        let f = cpi_to_fn[&method_index.as_usize()];
        // println!("returning cpi: {}, addr: {:#x}", method_index.as_usize(), fn_to_addr!(f));
        fn_to_addr!(f)
    } else {
        0
    }
}

extern "sysv64" fn jit_call_function(
    program: &Program,
    state: &mut State,
    name_index: ConstantPoolIndex,
    arity: Arity,
    cpi_to_fn: &FnvHashMap<usize, fn()>,
) -> i64 {
    let method_index = eval_call_function(program, state, name_index, arity).unwrap();
    let f = cpi_to_fn[&method_index.as_usize()];
    // println!("returning cpi: {}, addr: {:#x}", method_index.as_usize(), fn_to_addr!(f));
    fn_to_addr!(f)
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

extern "sysv64" fn jit_label(program: &Program, state: &mut State) {
    eval_label(program, state).unwrap();
}

extern "sysv64" fn jit_jump(program: &Program, state: &mut State, label_index: ConstantPoolIndex) {
    eval_jump(program, state, label_index).unwrap();
}

extern "sysv64" fn jit_branch(program: &Program, state: &mut State, label_index: ConstantPoolIndex) -> i32 {
    eval_branch(program, state, label_index).unwrap().into()
}

extern "sysv64" fn jit_return(program: &Program, state: &mut State) {
    eval_return(program, state).unwrap();
}

extern "sysv64" fn jit_drop(program: &Program, state: &mut State) {
    eval_drop(program, state).unwrap();
}
