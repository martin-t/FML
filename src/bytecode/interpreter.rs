use std::fmt::Write;

use crate::bytecode::heap::*;
use crate::bytecode::opcodes::OpCode;

use anyhow::{bail, ensure, Context, Result};

use crate::veccat;

use crate::bytecode::program::*;
use crate::bytecode::state::*;
use indexmap::map::IndexMap;
use std::path::PathBuf;

trait OpCodeEvaluationResult<T> {
    fn attach(self, opcode: &OpCode) -> Result<T>;
}

impl<T> OpCodeEvaluationResult<T> for Result<T> {
    #[inline(always)]
    fn attach(self, opcode: &OpCode) -> Result<T> {
        self.with_context(|| format!("Error evaluating {}:", opcode))
    }
}

#[allow(dead_code)]
pub fn evaluate(program: &Program) -> Result<()> {
    let mut state = State::from(program)?;
    let mut output = Output::new();
    evaluate_with(program, &mut state, &mut output)
}

pub fn evaluate_with_memory_config(
    program: &Program,
    heap_gc_size: Option<usize>,
    heap_log: Option<PathBuf>,
) -> Result<()> {
    let mut state = State::from(program)?;
    state.heap.set_gc_size(heap_gc_size);
    if let Some(log) = heap_log {
        state.heap.set_log(log);
    }
    let mut output = Output::new();
    evaluate_with(program, &mut state, &mut output)
}

pub fn evaluate_with<W>(program: &Program, state: &mut State, output: &mut W) -> Result<()>
where
    W: Write,
{
    // eprintln!("Program:");
    // eprintln!("{}", program);
    while let Some(address) = state.instruction_pointer.get() {
        let opcode = program.code.get(address)?;
        eval_opcode(program, state, output, opcode)?;
    }
    Ok(())
}

#[allow(dead_code)]
pub fn step_with<W>(program: &Program, state: &mut State, output: &mut W) -> Result<()>
where
    W: Write,
{
    let address = state.instruction_pointer.get().with_context(|| "Nothing to execute.")?;
    let opcode = program.code.get(address)?;
    eval_opcode(program, state, output, opcode)
}

#[rustfmt::skip]
pub fn eval_opcode<W>(
    program: &Program,
    state: &mut State,
    output: &mut W,
    opcode: &OpCode,
) -> Result<()>
where
    W: Write,
{
    match opcode {
        OpCode::Literal { index } => eval_literal(program, state, index),
        OpCode::GetLocal { index } => eval_get_local(program, state, index),
        OpCode::SetLocal { index } => eval_set_local(program, state, index),
        OpCode::GetGlobal { name } => eval_get_global(program, state, name),
        OpCode::SetGlobal { name } => eval_set_global(program, state, name),
        OpCode::Object { class } => eval_object(program, state, class),
        OpCode::Array => eval_array(program, state),
        OpCode::GetField { name } => eval_get_field(program, state, name),
        OpCode::SetField { name } => eval_set_field(program, state, name),
        OpCode::CallMethod { name, arguments } => eval_call_method(program, state, name, arguments),
        OpCode::CallFunction { name, arguments } => eval_call_function(program, state, name, arguments),
        OpCode::Label { .. } => eval_label(program, state),
        OpCode::Print { format, arguments } => eval_print(program, state, output, format, arguments),
        OpCode::Jump { label } => eval_jump(program, state, label),
        OpCode::Branch { label } => eval_branch(program, state, label),
        OpCode::Return => eval_return(program, state),
        OpCode::Drop => eval_drop(program, state),
    }
    .attach(opcode)
}

#[inline(always)]
pub fn eval_literal(program: &Program, state: &mut State, index: &ConstantPoolIndex) -> Result<()> {
    let program_object = program.constant_pool.get(index)?;
    let pointer = Pointer::from_literal(program_object)?;
    state.operand_stack.push(pointer);
    state.instruction_pointer.bump(program);
    Ok(())
}

#[inline(always)]
pub fn eval_get_local(program: &Program, state: &mut State, index: &LocalFrameIndex) -> Result<()> {
    // LATER(kondziu) rename LocalFrameIndex to FrameIndex
    let frame = state.frame_stack.get_locals()?;
    let pointer = *frame.get(index)?;
    state.operand_stack.push(pointer);
    state.instruction_pointer.bump(program);
    Ok(())
}

#[inline(always)]
pub fn eval_set_local(program: &Program, state: &mut State, index: &LocalFrameIndex) -> Result<()> {
    let pointer = *state.operand_stack.peek()?;
    let frame = state.frame_stack.get_locals_mut()?;
    frame.set(index, pointer)?;
    state.instruction_pointer.bump(program);
    Ok(())
}

#[inline(always)]
pub fn eval_get_global(program: &Program, state: &mut State, index: &ConstantPoolIndex) -> Result<()> {
    let program_object = program.constant_pool.get(index)?;
    let name = program_object.as_str()?;
    let pointer = *state.frame_stack.globals.get(name)?;
    state.operand_stack.push(pointer);
    state.instruction_pointer.bump(program);
    Ok(())
}

#[inline(always)]
pub fn eval_set_global(program: &Program, state: &mut State, index: &ConstantPoolIndex) -> Result<()> {
    let program_object = program.constant_pool.get(index)?;
    let name = program_object.as_str()?.to_owned();
    let pointer = *state.operand_stack.peek()?;
    state.frame_stack.globals.update(name, pointer)?;
    state.instruction_pointer.bump(program);
    Ok(())
}

#[inline(always)]
pub fn eval_object(program: &Program, state: &mut State, index: &ConstantPoolIndex) -> Result<()> {
    let program_object = program.constant_pool.get(index)?;
    let members = program_object
        .as_class_definition()?
        .iter()
        .map(|index| program.constant_pool.get(index))
        .collect::<Result<Vec<&ProgramObject>>>()?;

    let mut slots = Vec::new();
    let mut methods = IndexMap::new();

    for member in members {
        // LATER(kondziu) this could probably be a method in ProgramObject, something like: `create prototype object om class`
        match member {
            ProgramObject::Slot { name: index } => {
                let program_object = program.constant_pool.get(index)?;
                let name = program_object.as_str()?;
                slots.push(name);
            }
            ProgramObject::Method(method) => {
                // LATER(kondziu), probably don't need to store methods, tbh, just the class, which would simplify this a lot
                let program_object = program.constant_pool.get(&method.name)?;
                let name = program_object.as_str()?.to_owned();
                let previous = methods.insert(name.clone(), member.clone());
                ensure!(
                    previous.is_none(),
                    "Member method `{}` has a non-unique name within object.",
                    name
                )
            }
            _ => bail!("Class members must be either Methods or Slots, but found `{}`.", member),
        }
    }

    let mut fields = IndexMap::new();
    for name in slots.into_iter().rev() {
        let pointer = state.operand_stack.pop()?;
        let previous = fields.insert(name.to_owned(), pointer);
        ensure!(
            previous.is_none(),
            "Member field `{}` has a non-unique name in object",
            name
        )
    }
    fields = fields.into_iter().rev().collect();

    let parent = state.operand_stack.pop()?;

    let object = HeapObject::new_object(parent, fields, methods);

    let heap_index = state
        .heap
        .allocate(&mut state.frame_stack, &mut state.operand_stack, object); // LATER(kondziu) simplify
    state.operand_stack.push(Pointer::from(heap_index));
    state.instruction_pointer.bump(program);
    Ok(())
}

#[inline(always)]
pub fn eval_array(program: &Program, state: &mut State) -> Result<()> {
    let initializer = state.operand_stack.pop()?;
    let size = state.operand_stack.pop()?;

    let n = size.as_i32()?;
    ensure!(
        n >= 0,
        "Negative value `{}` cannot be used to specify the size of an array.",
        n
    );

    let elements = vec![initializer; n as usize];
    let array = HeapObject::from_pointers(elements);

    let heap_index = state
        .heap
        .allocate(&mut state.frame_stack, &mut state.operand_stack, array);
    state.operand_stack.push(Pointer::from(heap_index));
    state.instruction_pointer.bump(program);
    Ok(())
}

#[inline(always)]
pub fn eval_get_field(program: &Program, state: &mut State, index: &ConstantPoolIndex) -> Result<()> {
    let program_object = program.constant_pool.get(index)?;
    let name = program_object.as_str()?;
    let pointer = state.operand_stack.pop()?;
    let heap_pointer = pointer.into_heap_reference()?;
    let object = state.heap.dereference(&heap_pointer)?;

    let object_instance = object.as_object_instance()?;
    let pointer = object_instance.get_field(name)?;
    state.operand_stack.push(*pointer);
    state.instruction_pointer.bump(program);
    Ok(())
}

#[inline(always)]
pub fn eval_set_field(program: &Program, state: &mut State, index: &ConstantPoolIndex) -> Result<()> {
    let program_object = program.constant_pool.get(index)?;
    let name = program_object.as_str()?;
    let value_pointer = state.operand_stack.pop()?;
    let object_pointer = state.operand_stack.pop()?;
    let heap_pointer = object_pointer.into_heap_reference()?;
    let object = state.heap.dereference_mut(&heap_pointer)?;

    let object_instance = object.as_object_instance_mut()?;
    object_instance.set_field(name, value_pointer)?;
    state.operand_stack.push(value_pointer);
    state.instruction_pointer.bump(program);
    Ok(())
}

#[inline(always)]
pub fn eval_call_method(
    program: &Program,
    state: &mut State,
    index: &ConstantPoolIndex,
    arguments: &Arity,
) -> Result<()> {
    ensure!(
        arguments.to_usize() > 0,
        "All method calls require at least 1 parameter (receiver)"
    );

    let program_object = program.constant_pool.get(index)?;
    let method_name = program_object.as_str()?;

    let argument_pointers = state.operand_stack.pop_sequence(arguments.to_usize() - 1)?;
    let receiver_pointer = state.operand_stack.pop()?;

    dispatch_method(program, state, receiver_pointer, method_name, argument_pointers)
}

fn dispatch_method(
    program: &Program,
    state: &mut State,
    receiver_pointer: Pointer,
    method_name: &str,
    argument_pointers: Vec<Pointer>,
) -> Result<()> {
    match receiver_pointer {
        Pointer::Null => {
            dispatch_null_method(method_name, argument_pointers)?.push_onto(&mut state.operand_stack);
            state.instruction_pointer.bump(program);
        }
        Pointer::Integer(i) => {
            dispatch_integer_method(&i, method_name, argument_pointers)?.push_onto(&mut state.operand_stack);
            state.instruction_pointer.bump(program);
        }
        Pointer::Boolean(b) => {
            dispatch_boolean_method(&b, method_name, argument_pointers)?.push_onto(&mut state.operand_stack);
            state.instruction_pointer.bump(program);
        }
        Pointer::Reference(index) => match state.heap.dereference_mut(&index)? {
            HeapObject::Array(array) => {
                dispatch_array_method(array, method_name, argument_pointers)?.push_onto(&mut state.operand_stack);
                state.instruction_pointer.bump(program);
            }
            HeapObject::Object(_) => {
                dispatch_object_method(program, state, receiver_pointer, method_name, argument_pointers)?
            }
        },
    }
    Ok(())
}

fn dispatch_null_method(method_name: &str, argument_pointers: Vec<Pointer>) -> Result<Pointer> {
    ensure!(
        argument_pointers.len() == 1,
        "Invalid number of arguments for method `{}` in object `null`",
        method_name
    );

    let argument = argument_pointers.last().unwrap();

    #[rustfmt::skip]
    let pointer = match (method_name, argument) {
        ("==", Pointer::Null) | ("eq", Pointer::Null)  => Pointer::from(true),
        ("==", _) | ("eq", _)                          => Pointer::from(false),
        ("!=", Pointer::Null) | ("neq", Pointer::Null) => Pointer::from(false),
        ("!=", _) | ("neq", _)                         => Pointer::from(true),
        _ => bail!(
            "Call method error: no method `{}` in object `null`",
            method_name
        ),
    };

    Ok(pointer)
}

fn dispatch_integer_method(receiver: &i32, method_name: &str, argument_pointers: Vec<Pointer>) -> Result<Pointer> {
    ensure!(
        argument_pointers.len() == 1,
        "Invalid number of arguments for method `{}` in object `{}`",
        method_name,
        receiver
    );

    let argument_pointer = argument_pointers.last().unwrap();

    #[rustfmt::skip]
    let pointer = match (method_name, argument_pointer) {
        ("+",  Pointer::Integer(argument)) => Pointer::from(receiver +  argument),
        ("-",  Pointer::Integer(argument)) => Pointer::from(receiver -  argument),
        ("*",  Pointer::Integer(argument)) => Pointer::from(receiver *  argument),
        ("/",  Pointer::Integer(argument)) => Pointer::from(receiver /  argument),
        ("%",  Pointer::Integer(argument)) => Pointer::from(receiver %  argument),
        ("<=", Pointer::Integer(argument)) => Pointer::from(receiver <= argument),
        (">=", Pointer::Integer(argument)) => Pointer::from(receiver >= argument),
        ("<",  Pointer::Integer(argument)) => Pointer::from(receiver <  argument),
        (">",  Pointer::Integer(argument)) => Pointer::from(receiver >  argument),
        ("==", Pointer::Integer(argument)) => Pointer::from(receiver == argument),
        ("!=", Pointer::Integer(argument)) => Pointer::from(receiver != argument),
        ("==", _) => Pointer::from(false),
        ("!=", _) => Pointer::from(true),

        ("add", Pointer::Integer(argument)) => Pointer::from(receiver +  argument),
        ("sub", Pointer::Integer(argument)) => Pointer::from(receiver -  argument),
        ("mul", Pointer::Integer(argument)) => Pointer::from(receiver *  argument),
        ("div", Pointer::Integer(argument)) => Pointer::from(receiver /  argument),
        ("mod", Pointer::Integer(argument)) => Pointer::from(receiver %  argument),
        ("le",  Pointer::Integer(argument)) => Pointer::from(receiver <= argument),
        ("ge",  Pointer::Integer(argument)) => Pointer::from(receiver >= argument),
        ("lt",  Pointer::Integer(argument)) => Pointer::from(receiver <  argument),
        ("gt",  Pointer::Integer(argument)) => Pointer::from(receiver >  argument),
        ("eq",  Pointer::Integer(argument)) => Pointer::from(receiver == argument),
        ("neq", Pointer::Integer(argument)) => Pointer::from(receiver != argument),
        ("eq", _) => Pointer::from(false),
        ("neq", _) => Pointer::from(true),

        // ==, !=, eq, neq are already fully covered above.
        (method, argument)
            if method == "+"
                || method == "-"
                || method == "*"
                || method == "/"
                || method == "%"
                || method == "<="
                || method == ">="
                || method == "<"
                || method == ">"
                || method == "add"
                || method == "sub"
                || method == "mul"
                || method == "div"
                || method == "mod"
                || method == "le"
                || method == "ge"
                || method == "lt"
                || method == "gt" =>
        {
            bail!("Call method error: method {} is not defined in object `{}` for argument `{}` (expecting integer argument)", 
                  method, receiver, argument)
        }

        (method, argument) => bail!(
            "Call method error: no method `{}` in object `{}` (argment is `{}`)",
            method,
            receiver,
            argument
        ),
    };
    Ok(pointer)
}

fn dispatch_boolean_method(receiver: &bool, method_name: &str, argument_pointers: Vec<Pointer>) -> Result<Pointer> {
    ensure!(
        argument_pointers.len() == 1,
        "Invalid number of arguments for method `{}` in object `{}`",
        method_name,
        receiver
    );

    let argument_pointer = argument_pointers.last().unwrap();

    #[rustfmt::skip]
    let pointer = match (method_name, argument_pointer) {
        ("&",  Pointer::Boolean(argument)) => Pointer::from(*receiver && *argument),
        ("|",  Pointer::Boolean(argument)) => Pointer::from(*receiver || *argument),
        ("==", Pointer::Boolean(argument)) => Pointer::from(*receiver == *argument),
        ("!=", Pointer::Boolean(argument)) => Pointer::from(*receiver != *argument),
        ("==", _) => Pointer::from(false),
        ("!=", _) => Pointer::from(true),

        ("and", Pointer::Boolean(argument)) => Pointer::from(*receiver && *argument),
        ("or",  Pointer::Boolean(argument)) => Pointer::from(*receiver || *argument),
        ("eq",  Pointer::Boolean(argument)) => Pointer::from(*receiver == *argument),
        ("neq", Pointer::Boolean(argument)) => Pointer::from(*receiver != *argument),
        ("eq",  _) => Pointer::from(false),
        ("neq", _) => Pointer::from(true),

        (method, argument)
            if method == "&" || method == "|" || method == "and" || method == "or" =>
        {
            bail!("Call method error: method {} is not defined in object `{}` for argument `{}` (expecting boolean argument)", 
                  method, receiver, argument)
        }

        (method, argument) => bail!(
            "Call method error: no method `{}` in object `{}` (argment is `{}`)",
            method,
            receiver,
            argument
        ),
    };
    Ok(pointer)
}

fn dispatch_array_method(
    array: &mut ArrayInstance,
    method_name: &str,
    argument_pointers: Vec<Pointer>,
) -> Result<Pointer> {
    match method_name {
        "get" => dispatch_array_get_method(array, method_name, argument_pointers),
        "set" => dispatch_array_set_method(array, method_name, argument_pointers),
        // LATER(martin-t) Would be nice to print arguments as well
        _ => bail!("Call method error: no method `{}` in array `{}`", method_name, array),
    }
}

fn dispatch_array_get_method(
    array: &mut ArrayInstance,
    method_name: &str,
    argument_pointers: Vec<Pointer>,
) -> Result<Pointer> {
    ensure!(
        argument_pointers.len() == 1,
        "Invalid number of arguments for method `{}` in array `{}`, expecting 1",
        method_name,
        array
    );

    let index_pointer = argument_pointers.first().unwrap();
    let index = index_pointer.as_usize()?;
    array.get_element(index).map(|e| *e)
}

fn dispatch_array_set_method(
    array: &mut ArrayInstance,
    method_name: &str,
    argument_pointers: Vec<Pointer>,
) -> Result<Pointer> {
    ensure!(
        argument_pointers.len() == 2,
        "Invalid number of arguments for method `{}` in array `{}`, expecting 2",
        method_name,
        array
    );

    let index_pointer = argument_pointers.first().unwrap();
    let value_pointer = argument_pointers.last().unwrap();
    let index = index_pointer.as_usize()?;
    array.set_element(index, *value_pointer).map(|e| *e)
}

fn dispatch_object_method(
    program: &Program,
    state: &mut State,
    receiver_pointer: Pointer,
    method_name: &str,
    argument_pointers: Vec<Pointer>,
) -> Result<()> {
    let heap_reference = receiver_pointer.into_heap_reference()?; // Should never fail.
    let heap_object = state.heap.dereference_mut(&heap_reference)?; // Should never fail.
    let object_instance = heap_object.as_object_instance_mut()?; // Should never fail.
    let parent_pointer = object_instance.parent;

    let method_option = object_instance.methods.get(method_name).cloned();

    match method_option {
        Some(method) => {
            let method = method.as_method()?;
            eval_call_object_method(program, state, method, method_name, receiver_pointer, argument_pointers)
        }
        None if object_instance.parent.is_null() => {
            // LATER(martin-t) Would be nice to print arguments as well
            bail!(
                "Call method error: no method `{}` in object `{}`",
                method_name,
                object_instance
            );
        }
        None => dispatch_method(program, state, parent_pointer, method_name, argument_pointers),
    }
}

fn eval_call_object_method(
    program: &Program,
    state: &mut State,
    method: &Method,
    method_name: &str,
    pointer: Pointer,
    argument_pointers: Vec<Pointer>,
) -> Result<()> {
    ensure!(
        argument_pointers.len() == method.parameters.to_usize() - 1,
        "Method `{}` requires {} arguments, but {} were supplied",
        method_name,
        method.parameters,
        argument_pointers.len()
    );

    let local_pointers = vec![Pointer::Null; method.locals.to_usize()];

    state.instruction_pointer.bump(program);
    let frame = Frame::from(
        state.instruction_pointer.get(),
        veccat!(vec![pointer], argument_pointers, local_pointers),
    );
    state.frame_stack.push(frame);
    state.instruction_pointer.set(Some(*method.code.start()));
    Ok(())
}

#[inline(always)]
pub fn eval_call_function(
    program: &Program,
    state: &mut State,
    index: &ConstantPoolIndex,
    arguments: &Arity,
) -> Result<()> {
    let program_object = program.constant_pool.get(index)?;
    let name = program_object.as_str()?;
    let function_index = state.frame_stack.functions.get(name)?;
    let function = program.constant_pool.get(function_index)?.as_method()?;

    ensure!(
        arguments == &function.parameters,
        "Function `{}` requires {} arguments, but {} were supplied",
        name,
        function.parameters,
        arguments
    );

    let argument_pointers = state.operand_stack.pop_sequence(arguments.to_usize())?;
    let local_pointers = vec![Pointer::Null; function.locals.to_usize()];

    state.instruction_pointer.bump(program);
    let frame = Frame::from(
        state.instruction_pointer.get(),
        veccat!(argument_pointers, local_pointers),
    );
    state.frame_stack.push(frame);
    state.instruction_pointer.set(Some(*function.code.start()));
    Ok(())
}

#[rustfmt::skip] // This should apply just to the match but attributes are not allowed on exprs yet
#[inline(always)]
pub fn eval_print<W>(
    program: &Program,
    state: &mut State,
    output: &mut W,
    index: &ConstantPoolIndex,
    arguments: &Arity,
) -> Result<()>
where
    W: Write,
{
    let program_object = program.constant_pool.get(index)?;
    let format = program_object.as_str()?;
    let mut argument_pointers = state
        .operand_stack
        .pop_reverse_sequence(arguments.to_usize())?;

    let mut escaped = false;

    for character in format.chars() {
        match (escaped, character) {
            (true,  '~' ) => { output.write_char('~')?;  escaped = false; },
            (true,  '\\') => { output.write_char('\\')?; escaped = false; },
            (true,  '"' ) => { output.write_char('"')?;  escaped = false; },
            (true,  'n' ) => { output.write_char('\n')?; escaped = false; },
            (true,  't' ) => { output.write_char('\t')?; escaped = false; },
            (true,  'r' ) => { output.write_char('\r')?; escaped = false; },
            (true,  chr  ) => { bail!("Unknown control sequence \\{}", chr) },
            (false, '\\') => {                           escaped = true;  },
            (_,    '~'  ) => {
                let argument = argument_pointers.pop()
                    .with_context(|| "Not enough arguments for format `{}`")?;
                output.write_str(argument.evaluate_as_string(&state.heap)?.as_str())?
            }
            (_, chr) => output.write_char(chr)?,
        }
    }
    ensure!(
        argument_pointers.is_empty(),
        "{} unused arguments for format `{}`",
        argument_pointers.len(),
        format
    );

    state.operand_stack.push(Pointer::Null);
    state.instruction_pointer.bump(program);
    Ok(())
}

#[inline(always)]
pub fn eval_label(program: &Program, state: &mut State) -> Result<()> {
    state.instruction_pointer.bump(program);
    Ok(())
}

#[inline(always)]
pub fn eval_jump(program: &Program, state: &mut State, index: &ConstantPoolIndex) -> Result<()> {
    let program_object = program.constant_pool.get(index)?;
    let name = program_object.as_str()?;
    let address = *program.labels.get(name)?;
    state.instruction_pointer.set(Some(address));
    Ok(())
}

#[inline(always)]
pub fn eval_branch(program: &Program, state: &mut State, index: &ConstantPoolIndex) -> Result<()> {
    let program_object = program.constant_pool.get(index)?;
    let name = program_object.as_str()?;
    let pointer = state.operand_stack.pop()?;
    if !pointer.evaluate_as_condition() {
        state.instruction_pointer.bump(program);
    } else {
        let address = *program.labels.get(name)?;
        state.instruction_pointer.set(Some(address));
    }
    Ok(())
}

#[inline(always)]
pub fn eval_return(_program: &Program, state: &mut State) -> Result<()> {
    let frame = state.frame_stack.pop()?;
    state.instruction_pointer.set(frame.return_address);
    Ok(())
}

#[inline(always)]
pub fn eval_drop(program: &Program, state: &mut State) -> Result<()> {
    state.operand_stack.pop()?;
    state.instruction_pointer.bump(program);
    Ok(())
}
