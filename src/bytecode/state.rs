use std::{
    collections::{
        hash_map::{Values, ValuesMut},
        HashMap, HashSet,
    },
    fmt,
    io::Write,
};

use anyhow::{anyhow, bail, ensure, Context, Result};

use crate::bytecode::{heap::*, program::*};

#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Clone, Copy)]
// LATER(martin-t) It should be possible to avoid the Option here.
pub struct InstructionPointer(Option<Address>);
impl InstructionPointer {
    pub fn new() -> Self {
        InstructionPointer(None)
    }
    pub fn bump(&mut self, program: &Program) {
        if let Some(address) = self.0 {
            self.0 = program.code.next(address)
        }
    }
    pub fn set(&mut self, address: Option<Address>) {
        self.0 = address
    }
    pub fn get(&self) -> Option<Address> {
        self.0
    }
}
impl From<Address> for InstructionPointer {
    fn from(address: Address) -> Self {
        InstructionPointer(Some(address))
    }
}
impl From<&Address> for InstructionPointer {
    fn from(address: &Address) -> Self {
        InstructionPointer(Some(*address))
    }
}
impl From<u32> for InstructionPointer {
    fn from(n: u32) -> Self {
        InstructionPointer(Some(Address::from_u32(n)))
    }
}
impl From<usize> for InstructionPointer {
    fn from(n: usize) -> Self {
        InstructionPointer(Some(Address::from_usize(n)))
    }
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Clone, Default)]
pub struct OperandStack(Vec<Pointer>);
impl OperandStack {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn values(&self) -> &Vec<Pointer> {
        &self.0
    }
    pub fn values_mut(&mut self) -> &mut Vec<Pointer> {
        &mut self.0
    }
    pub fn push(&mut self, pointer: Pointer) {
        self.0.push(pointer)
    }
    pub fn pop(&mut self) -> Result<Pointer> {
        self.0.pop().with_context(|| "Cannot pop from an empty operand stack.")
    }
    pub fn peek(&self) -> Result<&Pointer> {
        self.0
            .last()
            .with_context(|| "Cannot peek from an empty operand stack.")
    }
    #[allow(dead_code)]
    pub fn pop_sequence(&mut self, n: usize) -> Result<Vec<Pointer>> {
        let result = (0..n).map(|_| self.pop()).collect::<Result<Vec<Pointer>>>();
        result.map(|mut sequence| {
            sequence.reverse();
            sequence
        })
    }
    pub fn pop_reverse_sequence(&mut self, n: usize) -> Result<Vec<Pointer>> {
        (0..n).map(|_| self.pop()).collect::<Result<Vec<Pointer>>>()
    }
}

impl From<Vec<Pointer>> for OperandStack {
    fn from(vector: Vec<Pointer>) -> Self {
        OperandStack(vector)
    }
}

#[derive(Eq, PartialEq, Debug)]
pub struct Frame {
    pub return_address: Option<Address>,
    locals: Vec<Pointer>,
}
impl Frame {
    pub fn new() -> Self {
        Frame {
            locals: Vec::new(),
            return_address: None,
        }
    }
    pub fn with_capacity(return_address: Option<Address>, size: usize, initial: Pointer) -> Self {
        Frame {
            locals: (0..size).map(|_| initial).collect(),
            return_address,
        }
    }
    pub fn from(return_address: Option<Address>, locals: Vec<Pointer>) -> Self {
        Frame { locals, return_address }
    }
    pub fn locals(&self) -> &Vec<Pointer> {
        &self.locals
    }
    pub fn locals_mut(&mut self) -> &mut Vec<Pointer> {
        &mut self.locals
    }
    pub fn get(&self, index: &LocalFrameIndex) -> Result<&Pointer> {
        let index = index.value() as usize;
        if index >= self.locals.len() {
            bail!("Local frame index {} out of range (0..{})", index, self.locals.len());
        }
        Ok(&self.locals[index])
    }
    pub fn set(&mut self, index: &LocalFrameIndex, pointer: Pointer) -> Result<()> {
        let index = index.value() as usize;
        if index >= self.locals.len() {
            bail!("Local frame index {} out of range (0..{})", index, self.locals.len());
        }
        self.locals[index] = pointer;
        Ok(())
    }
}

#[derive(Eq, PartialEq, Debug)]
pub struct FrameStack {
    pub globals: GlobalFrame,
    pub functions: GlobalFunctions,
    frames: Vec<Frame>,
}
impl FrameStack {
    pub fn new() -> Self {
        FrameStack {
            globals: GlobalFrame::new(),
            functions: GlobalFunctions::new(),
            frames: Vec::new(),
        }
    }
    pub fn frames(&self) -> &Vec<Frame> {
        &self.frames
    }
    pub fn frames_mut(&mut self) -> &mut Vec<Frame> {
        &mut self.frames
    }
    pub fn pop(&mut self) -> Result<Frame> {
        self.frames
            .pop()
            .with_context(|| "Attempting to pop frame from empty stack.")
    }
    pub fn push(&mut self, frame: Frame) {
        self.frames.push(frame)
    }
    pub fn get_locals(&self) -> Result<&Frame> {
        self.frames
            .last()
            .with_context(|| "Attempting to access frame from empty stack.")
    }
    pub fn get_locals_mut(&mut self) -> Result<&mut Frame> {
        self.frames
            .last_mut()
            .with_context(|| "Attempting to access frame from empty stack.")
    }
}

impl From<(GlobalFrame, GlobalFunctions)> for FrameStack {
    fn from((globals, functions): (GlobalFrame, GlobalFunctions)) -> Self {
        FrameStack {
            globals,
            functions,
            frames: Vec::new(),
        }
    }
}

impl From<Frame> for FrameStack {
    fn from(frame: Frame) -> Self {
        FrameStack {
            globals: GlobalFrame::new(),
            functions: GlobalFunctions::new(),
            frames: vec![frame],
        }
    }
}

#[derive(Eq, PartialEq, Debug)]
pub struct GlobalFunctions(HashMap<String, ConstantPoolIndex>);
impl GlobalFunctions {
    pub fn new() -> Self {
        GlobalFunctions(HashMap::new())
    }
    pub fn get(&self, name: &str) -> Result<&ConstantPoolIndex> {
        self.0
            .get(name)
            .with_context(|| format!("No such function `{}`.", name))
    }
    #[allow(dead_code)]
    pub fn update(&mut self, name: String, index: ConstantPoolIndex) -> Result<()> {
        let result = self.0.insert(name.clone(), index);
        ensure!(result.is_some(), "No such function `{}`.", name);
        Ok(())
    }
    #[allow(dead_code)]
    pub fn define(&mut self, name: String, index: ConstantPoolIndex) -> Result<()> {
        let result = self.0.insert(name.clone(), index);
        ensure!(result.is_none(), "Cannot define function `{}`: already defined.", name);
        Ok(())
    }
    pub fn from(methods: Vec<(String, ConstantPoolIndex)>) -> Result<Self> {
        let mut unique = HashSet::new();
        let functions = methods
            .into_iter()
            .map(|(name, index)| {
                if unique.insert(name.clone()) {
                    Ok((name, index))
                } else {
                    Err(anyhow!("Function is a duplicate: {}", name))
                }
            })
            .collect::<Result<HashMap<String, ConstantPoolIndex>>>()?;
        Ok(GlobalFunctions(functions))
    }
}

#[derive(Eq, PartialEq, Debug)]
pub struct GlobalFrame(HashMap<String, Pointer>);
impl GlobalFrame {
    pub fn new() -> Self {
        GlobalFrame(HashMap::new())
    }
    pub fn get(&self, name: &str) -> Result<&Pointer> {
        self.0.get(name).with_context(|| format!("No such global `{}`.", name))
    }
    pub fn values(&self) -> Values<String, Pointer> {
        self.0.values()
    }
    pub fn values_mut(&mut self) -> ValuesMut<String, Pointer> {
        self.0.values_mut()
    }
    #[allow(dead_code)]
    pub fn update(&mut self, name: String, pointer: Pointer) -> Result<()> {
        let result = self.0.insert(name.clone(), pointer);
        ensure!(result.is_some(), "No such global `{}`.", name);
        Ok(())
    }
    #[allow(dead_code)]
    pub fn define(&mut self, name: String, pointer: Pointer) -> Result<()> {
        let result = self.0.insert(name.clone(), pointer);
        ensure!(result.is_none(), "Cannot define global `{}`: already defined.", name);
        Ok(())
    }
    pub fn from(names: Vec<String>, initial: Pointer) -> Result<Self> {
        let mut unique = HashSet::new();
        let globals = names
            .into_iter()
            .map(|name| {
                if unique.insert(name.clone()) {
                    Ok((name, initial))
                } else {
                    Err(anyhow!("Global is a duplicate: {}", name))
                }
            })
            .collect::<Result<HashMap<String, Pointer>>>()?;
        Ok(GlobalFrame(globals))
    }
}

#[derive(Debug)]
pub struct State {
    pub operand_stack: OperandStack,
    pub frame_stack: FrameStack,
    pub instruction_pointer: InstructionPointer,
    pub heap: Heap,
}

impl State {
    pub fn from(program: &Program) -> Result<Self> {
        // LATER(kondziu) error handling is a right mess here.

        let entry_index = program.entry.get().with_context(|| "Cannot find entry method.")?;
        let entry_method = program
            .constant_pool
            .get(&entry_index)
            .with_context(|| "Cannot find entry method.")?
            .as_method()?;

        let instruction_pointer = if entry_method.code.length() > 0 {
            InstructionPointer::from(*entry_method.code.start())
        } else {
            InstructionPointer::new()
        };

        let global_objects = program
            .globals
            .iter()
            .map(|index| program.constant_pool.get(&index).map(|object| (index, object)))
            .collect::<Result<Vec<(ConstantPoolIndex, &ProgramObject)>>>()?;

        ensure!(
            global_objects
                .iter()
                .all(|(_, object)| object.is_slot() || object.is_method()),
            "Illegal global constant: expecting Method or Slot."
        );

        fn extract_slot(program: &Program, slot: &ProgramObject) -> Result<String> {
            let name_index = slot.as_slot_index()?;
            let name_object = program.constant_pool.get(name_index)?;
            let name = name_object.as_str()?;
            Ok(name.to_owned())
        }

        let globals = global_objects
            .iter()
            .filter(|(_, program_object)| program_object.is_slot())
            .map(|(_, slot)| extract_slot(program, slot))
            .collect::<Result<Vec<String>>>()?;

        fn extract_function(
            program: &Program,
            index: &ConstantPoolIndex,
            method: &ProgramObject,
        ) -> Result<(String, ConstantPoolIndex)> {
            let name_index = method.as_method()?.name;
            let name_object = program.constant_pool.get(&name_index)?;
            let name = name_object.as_str()?;
            Ok((name.to_owned(), *index))
        }

        let functions = global_objects
            .iter()
            .filter(|(_, program_object)| program_object.is_method())
            .map(|(index, method)| extract_function(program, index, method))
            .collect::<Result<Vec<(String, ConstantPoolIndex)>>>()?;

        let global_frame = GlobalFrame::from(globals, Pointer::Null)?;
        let global_functions = GlobalFunctions::from(functions)?;
        let mut frame_stack = FrameStack::from((global_frame, global_functions));
        frame_stack.push(Frame::with_capacity(
            None,
            entry_method.locals.to_usize(),
            Pointer::Null,
        ));

        let operand_stack = OperandStack::new();
        let heap: Heap = Heap::new();

        Ok(State {
            operand_stack,
            frame_stack,
            instruction_pointer,
            heap,
        })
    }

    #[allow(dead_code)]
    pub fn new() -> Self {
        State {
            operand_stack: OperandStack::new(),
            frame_stack: FrameStack::new(),
            instruction_pointer: InstructionPointer::new(),
            heap: Heap::new(),
        }
    }

    #[allow(dead_code)]
    pub fn minimal() -> Self {
        State {
            operand_stack: OperandStack::new(),
            frame_stack: FrameStack::from(Frame::new()),
            instruction_pointer: InstructionPointer::from(Address::from_usize(0)),
            heap: Heap::new(),
        }
    }
}

pub struct Output();

impl Output {
    pub fn new() -> Self {
        Output()
    }
}

impl fmt::Write for Output {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        match std::io::stdout().write_all(s.as_bytes()) {
            Ok(_) => Ok(()),
            Err(_) => Err(std::fmt::Error),
        }
    }
}
