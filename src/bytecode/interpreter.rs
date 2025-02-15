use std::fmt::Write;
use std::path::PathBuf;

use anyhow::{bail, ensure, Context, Result};
use cpu_time::ProcessTime;
use indexmap::map::IndexMap;

use crate::bytecode::heap::*;
use crate::bytecode::opcodes::OpCode;
use crate::bytecode::program::*;
use crate::bytecode::state::*;
use crate::jit::*;

#[allow(dead_code)]
pub fn evaluate(program: &Program) -> Result<()> {
    let mut state = State::from(program)?;
    let mut output = StdOutput::new();
    evaluate_with(program, &mut state, &mut output)
}

pub fn evaluate_with_memory_config(
    program: &Program,
    heap_gc_size: Option<usize>,
    heap_log: Option<PathBuf>,
    jit: bool,
    debug: String,
) -> Result<()> {
    let mut state = State::from(program)?;
    state.heap.set_gc_size(heap_gc_size);
    if let Some(log) = heap_log {
        state.heap.set_log(log);
    }
    // Surround with spaces so we can use .contains(" whatever ")
    // also with spaces and not worry about matching a subword.
    // E.g. " ds " can enable debug_state but won't accidentally match " methods ".
    state.debug = format!(" {} ", debug);
    let mut output = StdOutput::new();
    evaluate_with_jit(program, &mut state, &mut output, jit)
}

pub fn evaluate_with<W>(program: &Program, state: &mut State, output: &mut W) -> Result<()>
where
    W: Write,
{
    evaluate_with_jit(program, state, output, false)
}

pub fn evaluate_with_jit<W>(program: &Program, state: &mut State, output: &mut W, jit: bool) -> Result<()>
where
    W: Write,
{
    if state.debug.contains(" program ") {
        eprintln!("Program:");
        eprintln!("{}", program);
    }

    let start = ProcessTime::now();

    if jit {
        jit_program(program, state, output);
    } else if state.debug.contains(" ds ") {
        while let Some(address) = state.instruction_pointer.get() {
            let opcode = program.code.get(address)?;
            debug_state(state);
            eval_opcode(program, state, output, opcode)?;
        }
        debug_state(state);
    } else {
        while let Some(address) = state.instruction_pointer.get() {
            let opcode = program.code.get(address)?;
            eval_opcode(program, state, output, opcode)?;
        }
    }

    if state.debug.contains(" timing ") {
        eprintln!("Elapsed: {:.3} ms", start.elapsed().as_secs_f64() * 1000.0);
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

pub fn eval_opcode<W>(program: &Program, state: &mut State, output: &mut W, opcode: OpCode) -> Result<()>
where
    W: Write,
{
    eval_opcode_inner(program, state, output, opcode).with_context(|| format!("Error evaluating {opcode}:"))
}

pub fn eval_opcode_inner<W>(program: &Program, state: &mut State, output: &mut W, opcode: OpCode) -> Result<()>
where
    W: Write,
{
    match opcode {
        OpCode::Literal { index } => {
            eval_literal(program, state, index)?;
            state.instruction_pointer.bump(program);
        }
        OpCode::GetLocal { index } => {
            eval_get_local(state, index)?;
            state.instruction_pointer.bump(program);
        }
        OpCode::SetLocal { index } => {
            eval_set_local(state, index)?;
            state.instruction_pointer.bump(program);
        }
        OpCode::GetGlobal { name } => {
            eval_get_global(program, state, name)?;
            state.instruction_pointer.bump(program);
        }
        OpCode::SetGlobal { name } => {
            eval_set_global(program, state, name)?;
            state.instruction_pointer.bump(program);
        }
        OpCode::Object { class } => {
            eval_object(program, state, class)?;
            state.instruction_pointer.bump(program);
        }
        OpCode::Array => {
            eval_array(state)?;
            state.instruction_pointer.bump(program);
        }
        OpCode::GetField { name } => {
            eval_get_field(program, state, name)?;
            state.instruction_pointer.bump(program);
        }
        OpCode::SetField { name } => {
            eval_set_field(program, state, name)?;
            state.instruction_pointer.bump(program);
        }
        OpCode::CallMethod { name, arity } => {
            let (_, address) = eval_call_method(program, state, name, arity)?;
            if let Some(address) = address {
                state.instruction_pointer.set(Some(address));
            } else {
                state.instruction_pointer.bump(program);
            }
        }
        OpCode::CallFunction { name, arity } => {
            let (_, address) = eval_call_function(program, state, name, arity)?;
            state.instruction_pointer.set(Some(address));
        }
        OpCode::Label { .. } => {
            state.instruction_pointer.bump(program);
        }
        OpCode::Print { format, arity } => {
            eval_print(program, state, output, format, arity)?;
            state.instruction_pointer.bump(program);
        }
        OpCode::Jump { label } => {
            let address = eval_jump(program, label)?;
            state.instruction_pointer.set(Some(address));
        }
        OpCode::Branch { label } => {
            let dest = eval_branch(program, state, label)?;
            if let Some(address) = dest {
                state.instruction_pointer.set(Some(address));
            } else {
                state.instruction_pointer.bump(program);
            }
        }
        OpCode::Return => {
            let ip = eval_return(state)?;
            state.instruction_pointer.set(ip);
        }
        OpCode::Drop => {
            eval_drop(state)?;
            state.instruction_pointer.bump(program);
        }
    };

    Ok(())
}

#[allow(dead_code)]
pub fn debug_state(state: &State) {
    println!("State:");
    println!("{:?}", state.operand_stack);
    println!("{:?}", state.frame_stack);
    println!("{:?}", state.instruction_pointer);
    println!("{:?}", state.heap);
    println!();
}

#[inline(always)]
pub fn eval_literal(program: &Program, state: &mut State, literal_index: ConstantPoolIndex) -> Result<()> {
    let literal_object = program.constant_pool.get(literal_index)?;
    let value = Pointer::from_literal(literal_object)?;
    state.operand_stack.push(value);

    Ok(())
}

#[inline(always)]
pub fn eval_get_local(state: &mut State, local_index: LocalIndex) -> Result<()> {
    let frame = state.frame_stack.get_locals()?;
    let value = *frame.get(local_index)?;
    state.operand_stack.push(value);

    Ok(())
}

#[inline(always)]
pub fn eval_set_local(state: &mut State, local_index: LocalIndex) -> Result<()> {
    let new_value = *state.operand_stack.peek()?;
    let frame = state.frame_stack.get_locals_mut()?;
    frame.set(local_index, new_value)?;

    Ok(())
}

#[inline(always)]
pub fn eval_get_global(program: &Program, state: &mut State, global_index: ConstantPoolIndex) -> Result<()> {
    let global_name = program.constant_pool.get(global_index)?;
    let global_name = global_name.as_str()?;
    let global = *state.frame_stack.globals.get(global_name)?;
    state.operand_stack.push(global);

    Ok(())
}

#[inline(always)]
pub fn eval_set_global(program: &Program, state: &mut State, global_index: ConstantPoolIndex) -> Result<()> {
    let global_object = program.constant_pool.get(global_index)?;
    let global_name = global_object.as_str()?.to_owned();
    let new_value = *state.operand_stack.peek()?;
    state.frame_stack.globals.update(global_name, new_value)?;

    Ok(())
}

#[inline(always)]
pub fn eval_object(program: &Program, state: &mut State, class_index: ConstantPoolIndex) -> Result<()> {
    let class_object = program.constant_pool.get(class_index)?;

    let mut field_names = Vec::new();
    let mut methods = IndexMap::new();

    for &member_index in class_object.as_class_members()? {
        let member = program.constant_pool.get(member_index)?;
        match member {
            ProgramObject::Slot { name: field_name_index } => {
                let field_name_object = program.constant_pool.get(*field_name_index)?;
                let field_name = field_name_object.as_str()?;
                field_names.push(field_name);
            }
            ProgramObject::Method(method) => {
                // LATER(kondziu), probably don't need to store methods, tbh, just the class, which would simplify this a lot
                let name_object = program.constant_pool.get(method.name)?;
                let name = name_object.as_str()?;
                let previous = methods.insert(name.to_owned(), member_index);
                ensure!(
                    previous.is_none(),
                    "Member method `{}` has a non-unique name within object.",
                    name
                )
            }
            _ => bail!(
                "Class members must be either Methods or Slots, but found `{}`.",
                member_index
            ),
        }
    }

    let mut fields = IndexMap::new();
    for name in field_names.into_iter().rev() {
        let value = state.operand_stack.pop()?;
        let previous = fields.insert(name.to_owned(), value);
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
        .allocate(&mut state.frame_stack, &mut state.operand_stack, object);
    state.operand_stack.push(Pointer::from(heap_index));

    Ok(())
}

#[inline(always)]
pub fn eval_array(state: &mut State) -> Result<()> {
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

    Ok(())
}

#[inline(always)]
pub fn eval_get_field(program: &Program, state: &mut State, name_index: ConstantPoolIndex) -> Result<()> {
    let name_object = program.constant_pool.get(name_index)?;
    let field_name = name_object.as_str()?;

    let object = state.operand_stack.pop()?;
    let object_pointer = object.into_heap_reference()?;
    let heap_object = state.heap.dereference(&object_pointer)?;
    let object_instance = heap_object.as_object_instance()?;

    let value = object_instance.get_field(field_name)?;
    state.operand_stack.push(*value);

    Ok(())
}

#[inline(always)]
pub fn eval_set_field(program: &Program, state: &mut State, name_index: ConstantPoolIndex) -> Result<()> {
    let name_object = program.constant_pool.get(name_index)?;
    let field_name = name_object.as_str()?;

    let new_value = state.operand_stack.pop()?;

    let object = state.operand_stack.pop()?;
    let object_pointer = object.into_heap_reference()?;
    let heap_object = state.heap.dereference_mut(&object_pointer)?;
    let object_instance = heap_object.as_object_instance_mut()?;

    object_instance.set_field(field_name, new_value)?;
    state.operand_stack.push(new_value);

    Ok(())
}

#[inline(always)]
pub fn eval_call_method(
    program: &Program,
    state: &mut State,
    name_index: ConstantPoolIndex,
    arity: Arity,
) -> Result<(Option<ConstantPoolIndex>, Option<Address>)> {
    ensure!(
        arity.to_usize() > 0,
        "All method calls require at least 1 parameter (receiver)"
    );

    let name_object = program.constant_pool.get(name_index)?;
    let method_name = name_object.as_str()?;

    let arguments = state.operand_stack.pop_sequence(arity.to_usize() - 1)?;
    let receiver = state.operand_stack.pop()?;

    dispatch_method(program, state, receiver, method_name, arguments)
}

fn dispatch_method(
    program: &Program,
    state: &mut State,
    receiver: Pointer,
    method_name: &str,
    arguments: Vec<Pointer>,
) -> Result<(Option<ConstantPoolIndex>, Option<Address>)> {
    match receiver {
        Pointer::Null => {
            dispatch_null_method(method_name, arguments)?.push_onto(&mut state.operand_stack);
            Ok((None, None))
        }
        Pointer::Integer(i) => {
            dispatch_integer_method(&i, method_name, arguments)?.push_onto(&mut state.operand_stack);
            Ok((None, None))
        }
        Pointer::Boolean(b) => {
            dispatch_boolean_method(&b, method_name, arguments)?.push_onto(&mut state.operand_stack);
            Ok((None, None))
        }
        Pointer::Reference(index) => match state.heap.dereference_mut(&index)? {
            HeapObject::Array(array) => {
                dispatch_array_method(array, method_name, arguments)?.push_onto(&mut state.operand_stack);
                Ok((None, None))
            }
            HeapObject::Object(_) => dispatch_object_method(program, state, receiver, method_name, arguments),
        },
    }
}

fn dispatch_null_method(method_name: &str, arguments: Vec<Pointer>) -> Result<Pointer> {
    ensure!(
        arguments.len() == 1,
        "Invalid number of arguments for method `{}` in object `null`",
        method_name
    );

    let argument = arguments.last().unwrap();

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

fn dispatch_integer_method(receiver: &i32, method_name: &str, arguments: Vec<Pointer>) -> Result<Pointer> {
    ensure!(
        arguments.len() == 1,
        "Invalid number of arguments for method `{}` in object `{}`",
        method_name,
        receiver
    );

    let argument_pointer = arguments.last().unwrap();

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

fn dispatch_boolean_method(receiver: &bool, method_name: &str, arguments: Vec<Pointer>) -> Result<Pointer> {
    ensure!(
        arguments.len() == 1,
        "Invalid number of arguments for method `{}` in object `{}`",
        method_name,
        receiver
    );

    let argument_pointer = arguments.last().unwrap();

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

fn dispatch_array_method(array: &mut ArrayInstance, method_name: &str, arguments: Vec<Pointer>) -> Result<Pointer> {
    match method_name {
        "get" => dispatch_array_get_method(array, method_name, arguments),
        "set" => dispatch_array_set_method(array, method_name, arguments),
        // LATER(martin-t) Would be nice to print arguments as well
        _ => bail!("Call method error: no method `{}` in array `{}`", method_name, array),
    }
}

fn dispatch_array_get_method(array: &ArrayInstance, method_name: &str, arguments: Vec<Pointer>) -> Result<Pointer> {
    ensure!(
        arguments.len() == 1,
        "Invalid number of arguments for method `{}` in array `{}`, expecting 1",
        method_name,
        array
    );

    let index_pointer = arguments.first().unwrap();
    let index = index_pointer.as_usize()?;
    array.get_element(index).copied()
}

fn dispatch_array_set_method(array: &mut ArrayInstance, method_name: &str, arguments: Vec<Pointer>) -> Result<Pointer> {
    ensure!(
        arguments.len() == 2,
        "Invalid number of arguments for method `{}` in array `{}`, expecting 2",
        method_name,
        array
    );

    let index_pointer = arguments.first().unwrap();
    let value_pointer = arguments.last().unwrap();
    let index = index_pointer.as_usize()?;
    array.set_element(index, *value_pointer).copied()
}

fn dispatch_object_method(
    program: &Program,
    state: &mut State,
    receiver: Pointer,
    method_name: &str,
    arguments: Vec<Pointer>,
) -> Result<(Option<ConstantPoolIndex>, Option<Address>)> {
    let heap_reference = receiver.into_heap_reference()?; // Should never fail.
    let heap_object = state.heap.dereference_mut(&heap_reference)?; // Should never fail.
    let object_instance = heap_object.as_object_instance_mut()?; // Should never fail.
    let parent = object_instance.parent;

    let method_option = object_instance.methods.get(method_name);

    match method_option {
        Some(&method_index) => {
            let method = program.constant_pool.get(method_index)?.as_method()?;
            let address = eval_call_object_method(program, state, method, method_name, receiver, arguments)?;
            Ok((Some(method_index), address))
        }
        None if object_instance.parent.is_null() => {
            // LATER(martin-t) Would be nice to print arguments as well
            bail!(
                "Call method error: no method `{}` in object `{}`",
                method_name,
                object_instance
            );
        }
        None => dispatch_method(program, state, parent, method_name, arguments),
    }
}

fn eval_call_object_method(
    program: &Program,
    state: &mut State,
    method: &Method,
    method_name: &str,
    receiver: Pointer,
    arguments: Vec<Pointer>,
) -> Result<Option<Address>> {
    ensure!(
        arguments.len() == method.arity.to_usize() - 1,
        "Method `{}` requires {} arguments, but {} were supplied",
        method_name,
        method.arity,
        arguments.len()
    );

    let local_pointers = vec![Pointer::Null; method.locals.to_usize()];

    state.instruction_pointer.bump(program);
    let frame = Frame::from(
        state.instruction_pointer.get(),
        veccat!(vec![receiver], arguments, local_pointers),
    );
    state.frame_stack.push(frame);
    let address = Some(method.code_range.start());
    Ok(address)
}

/// This only executes the call opcode
/// and returns the index of the method the call resolved to.
/// It does not execute the whole function.
#[inline(always)]
pub fn eval_call_function(
    program: &Program,
    state: &mut State,
    name_index: ConstantPoolIndex,
    arity: Arity,
) -> Result<(ConstantPoolIndex, Address)> {
    let name_object = program.constant_pool.get(name_index)?;
    let function_name = name_object.as_str()?;
    let function_index = state.frame_stack.functions.get(function_name)?;
    let function = program.constant_pool.get(function_index)?.as_method()?;

    ensure!(
        arity == function.arity,
        "Function `{}` requires {} arguments, but {} were supplied",
        function_name,
        function.arity,
        arity
    );

    let arguments = state.operand_stack.pop_sequence(arity.to_usize())?;
    let local_pointers = vec![Pointer::Null; function.locals.to_usize()];

    state.instruction_pointer.bump(program);
    let frame = Frame::from(state.instruction_pointer.get(), veccat!(arguments, local_pointers));
    state.frame_stack.push(frame);
    let address = function.code_range.start();
    // state.instruction_pointer.set(Some(*function.code.start()));

    Ok((function_index, address))
}

#[inline(always)]
pub fn eval_print<W>(
    program: &Program,
    state: &mut State,
    output: &mut W,
    format_index: ConstantPoolIndex,
    arity: Arity,
) -> Result<()>
where
    W: Write,
{
    let format_object = program.constant_pool.get(format_index)?;
    let format = format_object.as_str()?;
    let mut arguments = state.operand_stack.pop_reverse_sequence(arity.to_usize())?;

    let mut escaped = false;

    for character in format.chars() {
        match (escaped, character) {
            (true, '~') => {
                output.write_char('~')?;
                escaped = false;
            }
            (true, '\\') => {
                output.write_char('\\')?;
                escaped = false;
            }
            (true, '"') => {
                output.write_char('"')?;
                escaped = false;
            }
            (true, 'n') => {
                output.write_char('\n')?;
                escaped = false;
            }
            (true, 't') => {
                output.write_char('\t')?;
                escaped = false;
            }
            (true, 'r') => {
                output.write_char('\r')?;
                escaped = false;
            }
            (true, chr) => {
                bail!("Unknown control sequence \\{}", chr)
            }
            (false, '\\') => {
                escaped = true;
            }
            (_, '~') => {
                let argument = arguments
                    .pop()
                    .with_context(|| "Not enough arguments for format `{}`")?;
                output.write_str(argument.evaluate_as_string(&state.heap)?.as_str())?
            }
            (_, chr) => output.write_char(chr)?,
        }
    }
    ensure!(
        arguments.is_empty(),
        "{} unused arguments for format `{}`",
        arguments.len(),
        format
    );

    state.operand_stack.push(Pointer::Null);

    Ok(())
}

#[inline(always)]
pub fn eval_jump(program: &Program, label_index: ConstantPoolIndex) -> Result<Address> {
    let label_object = program.constant_pool.get(label_index)?;
    let label_name = label_object.as_str()?;
    let address = *program.labels.get(label_name)?;
    Ok(address)
}

/// Return Some(address) if the condition is truthy, None otherwise.
/// The address is the address of the label to jump to.
/// If the condition is falsy, just bump the instruction pointer.
#[inline(always)]
pub fn eval_branch(program: &Program, state: &mut State, label_index: ConstantPoolIndex) -> Result<Option<Address>> {
    let label_object = program.constant_pool.get(label_index)?;
    let label_name = label_object.as_str()?;
    let value = state.operand_stack.pop()?;
    let truthy = value.evaluate_as_condition();
    if !truthy {
        Ok(None)
    } else {
        let address = *program.labels.get(label_name)?;
        Ok(Some(address))
    }
}

/// Returns the instruction pointer to return to.
///
/// Although this is the same return type as eval_branch,
/// the meaning is different.
/// Always set the instruction pointer to the returned address.
#[inline(always)]
pub fn eval_return(state: &mut State) -> Result<Option<Address>> {
    let frame = state.frame_stack.pop()?;
    Ok(frame.return_address)
}

#[inline(always)]
pub fn eval_drop(state: &mut State) -> Result<()> {
    state.operand_stack.pop()?;
    Ok(())
}
