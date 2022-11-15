use std::{
    collections::HashMap,
    fmt::{self, Display, Formatter},
    io::{Read, Write},
};

use anyhow::{anyhow, ensure, Context, Error, Result};

use crate::bytecode::{
    opcodes::OpCode,
    serializable::{self, Serializable, SerializableWithContext},
};

#[derive(Eq, PartialEq, Debug)]
pub struct Program {
    pub constant_pool: ConstantPool,
    pub globals: Globals,
    pub entry: Entry,
    pub code: Code,
    pub labels: Labels,
}

#[derive(Eq, PartialEq, Debug)]
pub struct ConstantPool(Vec<ProgramObject>);

#[derive(Eq, PartialEq, Debug)]
pub struct Globals(Vec<ConstantPoolIndex>);

// LATER(martin-t) This probably shouldn't be an Option
#[derive(Eq, PartialEq, Debug)]
pub struct Entry(Option<ConstantPoolIndex>);

#[derive(Eq, PartialEq, Debug)]
pub struct Code(Vec<OpCode>);

#[derive(Eq, PartialEq, Debug)]
pub struct Labels {
    names: HashMap<String, Address>,
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Clone)]
pub enum ProgramObject {
    /**
     * Represents a 32 bit integer. Used by the `Literal` instruction.
     *
     * Serialized with tag `0x00`.
     */
    Integer(i32),

    /**
     * Represents a boolean. Used by the `Literal` instruction.
     *
     * Serialized with tag `0x06`.
     */
    Boolean(bool),

    /**
     * Represents the unit value. Used by the `Literal` instruction.
     *
     * Serialized with tag `0x01`.
     */
    Null,

    /**
     * Represents a character string. Strings are used to:
     *   - represent the names of functions, slots, methods, and labels,
     *   - as format strings in the `Print`.
     *
     * Serialized with tag `0x02`.
     */
    String(String),

    /**
     * Represents one of two things:
     *   - a field member (aka slot) of an object when it is referred to from a `Class` object, or
     *   - a global variable when referred to from the list of Global slots.
     *
     * Contains an index that refers to a `ProgramObject::String` object. The string object
     * represents this slot's name.
     *
     * Serialized with tag `0x04`.
     */
    Slot { name: ConstantPoolIndex },

    /**
     * Represents one of two things:
     *   - a method member of an object, or
     *   - a global function.
     *
     * Contains:
     *   - `name`: an index that refers to a `ProgramObject::String` object, which represents this
     *             method's name,
     *   - `arguments`: the number of arguments this function takes,
     *   - `locals`: the number of local variables defined in this method,
     *   - `code`: a vector containing all the instructions in this method.
     *
     * Serialized with tag `0x03`.
     */
    Method(Method),

    /**
     * Represents an object structure consisting of field (aka slot) and method members for each
     * type of object created by `object`.
     *
     * It contains a vector containing indices to all the slots in the objects. Each index refers to
     * either:
     *   - a `ProgramObject::Slot` object representing a member field, or
     *   - a `ProgramObject::Method` object representing a member method.
     *
     * Serialized with tag `0x05`.
     */
    Class(Vec<ConstantPoolIndex>),
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Clone)]
pub struct Method {
    pub name: ConstantPoolIndex,
    pub parameters: Arity,
    pub locals: Size,
    pub code: AddressRange,
}

#[derive(PartialEq, Debug, Copy, Clone, Eq, PartialOrd, Ord, Hash)]
pub struct ConstantPoolIndex(u16);
#[derive(PartialEq, Debug, Copy, Clone, Eq, PartialOrd, Ord, Hash)]
pub struct LocalFrameIndex(u16);
#[derive(PartialEq, Debug, Copy, Clone, Eq, PartialOrd, Ord, Hash)]
pub struct Arity(u8);
#[derive(PartialEq, Debug, Copy, Clone, Eq, PartialOrd, Ord, Hash)]
pub struct Size(u16);
#[derive(PartialEq, Debug, Copy, Clone, Eq, PartialOrd, Ord, Hash)]
pub struct Address(u32);
#[derive(PartialEq, Debug, Copy, Clone, Eq, PartialOrd, Ord, Hash)]
pub struct AddressRange {
    start: Address,
    length: usize,
}

impl Program {
    #[allow(dead_code)]
    pub fn from(code: Code, constant_pool: ConstantPool, globals: Globals, entry: Entry) -> Result<Self> {
        let label_names = code.labels();
        let label_constants = constant_pool.get_all(label_names)?.into_iter();
        let label_addresses = code.label_addresses().into_iter();
        let labels = Labels::from(label_constants.zip(label_addresses)).unwrap();

        Ok(Program {
            constant_pool,
            globals,
            entry,
            code,
            labels,
        })
    }
}

impl ConstantPool {
    pub fn new() -> Self {
        ConstantPool(Vec::new())
    }
    pub fn get(&self, index: &ConstantPoolIndex) -> Result<&ProgramObject> {
        self.0
            .get(index.as_usize())
            .with_context(|| format!("Cannot dereference object from the constant pool at index: `{}`", index))
    }
    pub fn get_all(&self, indices: Vec<&ConstantPoolIndex>) -> Result<Vec<&ProgramObject>> {
        indices.iter().map(|index| self.get(index)).collect()
    }
    pub fn push(&mut self, program_object: ProgramObject) -> ConstantPoolIndex {
        self.0.push(program_object);
        ConstantPoolIndex::from_usize(self.0.len() - 1)
    }
    pub fn position(&self, program_object: &ProgramObject) -> Option<ConstantPoolIndex> {
        self.0
            .iter()
            .position(|c| c == program_object)
            .map(ConstantPoolIndex::from_usize)
    }
    pub fn register(&mut self, program_object: ProgramObject) -> ConstantPoolIndex {
        let index = self.position(&program_object);
        match index {
            Some(index) => index,
            None => self.push(program_object),
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = &ProgramObject> {
        self.0.iter()
    }
    #[allow(dead_code)]
    pub fn size(&self) -> usize {
        self.0.len()
    }
}

impl From<Vec<ProgramObject>> for ConstantPool {
    fn from(vector: Vec<ProgramObject>) -> Self {
        ConstantPool(vector)
    }
}

impl From<Vec<i32>> for ConstantPool {
    fn from(vector: Vec<i32>) -> Self {
        ConstantPool(vector.into_iter().map(ProgramObject::from_i32).collect())
    }
}

impl From<Vec<&str>> for ConstantPool {
    fn from(vector: Vec<&str>) -> Self {
        ConstantPool(vector.into_iter().map(ProgramObject::from_str).collect())
    }
}

impl From<Vec<bool>> for ConstantPool {
    fn from(vector: Vec<bool>) -> Self {
        ConstantPool(vector.into_iter().map(ProgramObject::from_bool).collect())
    }
}

impl Globals {
    pub fn new() -> Self {
        Globals(Vec::new())
    }
    pub fn register(&mut self, name_index: ConstantPoolIndex) -> Result<()> {
        ensure!(
            !self.0.contains(&name_index),
            "Cannot register global `{}`, index is already registered as a global.",
            name_index
        );
        self.0.push(name_index);
        Ok(())
    }
    pub fn iter(&self) -> impl Iterator<Item = ConstantPoolIndex> + '_ {
        self.0.iter().copied()
    }
}

impl From<Vec<ConstantPoolIndex>> for Globals {
    fn from(vector: Vec<ConstantPoolIndex>) -> Self {
        Globals(vector)
    }
}

impl Entry {
    pub fn new() -> Self {
        Entry(None)
    }
    pub fn get(&self) -> Result<ConstantPoolIndex> {
        ensure!(self.0.is_some(), "Entry point was read, but it was not set yet.");
        Ok(self.0.unwrap())
    }
    pub fn set(&mut self, index: ConstantPoolIndex) {
        self.0 = Some(index)
    }
}

impl From<ConstantPoolIndex> for Entry {
    fn from(index: ConstantPoolIndex) -> Self {
        Entry(Some(index))
    }
}

impl From<u16> for Entry {
    fn from(index: u16) -> Self {
        Entry(Some(ConstantPoolIndex::from(index)))
    }
}

impl Code {
    pub fn new() -> Self {
        Code(Vec::new())
    }
    pub fn upcoming_address(&self) -> Address {
        Address::from_usize(self.0.len())
    }
    pub fn extend(&mut self, code: Code) -> (Address, usize) {
        let first = self.upcoming_address();
        let length = code.length();
        // println!("code {} {} {:?}", first, length, code.0);
        self.0.extend(code.0.into_iter());
        (first, length)
    }
    pub fn emit(&mut self, opcode: OpCode) {
        self.0.push(opcode)
    }
    #[allow(dead_code)]
    pub fn emit_if(&mut self, opcode: OpCode, condition: bool) {
        if condition {
            self.emit(opcode)
        }
    }
    pub fn emit_unless(&mut self, opcode: OpCode, condition: bool) {
        if !condition {
            self.emit(opcode)
        }
    }
    #[allow(dead_code)]
    pub fn length(&self) -> usize {
        self.0.len()
    }
    pub fn materialize(&self, range: &AddressRange) -> Result<Vec<&OpCode>> {
        let start = range.start().value_usize();
        let end = start + range.length();

        //println!("start: {}, end: {}", start, end);

        ensure!(
            end <= self.0.len(),
            "Address range exceeds code size: {} + {} >= {}.",
            start,
            range.length,
            self.0.len()
        );

        Ok((start..end).map(|index| &self.0[index]).collect())
    }
    pub fn append(&mut self, opcodes: Vec<OpCode>) -> AddressRange {
        let start = self.0.len();
        let length = opcodes.len();
        self.0.extend(opcodes);
        AddressRange::new(Address::from_usize(start), length)
    }
    pub fn labels(&self) -> Vec<&ConstantPoolIndex> {
        self.0
            .iter()
            .flat_map(|opcode| match opcode {
                OpCode::Label { name } => Some(name),
                _ => None,
            })
            .collect()
    }
    pub fn label_addresses(&self) -> Vec<Address> {
        self.0
            .iter()
            .enumerate()
            .flat_map(|(address, opcode)| match opcode {
                OpCode::Label { .. } => Some(Address::from_usize(address)),
                _ => None,
            })
            .collect()
    }
    pub fn next(&self, address: Address) -> Option<Address> {
        let index = address.value_usize() + 1;
        if index < self.0.len() {
            Some(Address::from_usize(index))
        } else {
            None
        }
    }
    pub fn get(&self, address: Address) -> Result<&OpCode> {
        let index = address.value_usize();
        if index < self.0.len() {
            Ok(&self.0[index])
        } else {
            Err(anyhow!("Code address {} out of bounds (0..{})", address, self.0.len()))
        }
    }
}

impl From<Vec<OpCode>> for Code {
    fn from(vector: Vec<OpCode>) -> Self {
        Code(vector)
    }
}

// LATER(kondziu,fixme) clean up
impl Labels {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Labels { names: HashMap::new() }
    }
    pub fn get(&self, label: &str) -> Result<&Address> {
        self.names
            .get(label)
            .with_context(|| format!("Label `{}` was not previously register.", label))
    }
    pub fn from<'a, I>(labels: I) -> Result<Self>
    where
        I: IntoIterator<Item = (&'a ProgramObject, Address)>,
    {
        let names = labels
            .into_iter()
            .map(|(program_object, address)| program_object.as_str().map(|name| (name.to_owned(), address)))
            .collect::<Result<HashMap<String, Address>>>()?;
        Ok(Labels { names })
    }
}

impl ProgramObject {
    #[allow(dead_code)]
    pub fn is_literal(&self) -> bool {
        matches!(
            self,
            ProgramObject::Null | ProgramObject::Boolean(_) | ProgramObject::Integer(_)
        )
    }
    pub fn as_str(&self) -> anyhow::Result<&str> {
        match self {
            ProgramObject::String(string) => Ok(string),
            _ => anyhow::bail!("Expecting a program object representing a String, found `{}`", self),
        }
    }
    pub fn as_class_definition(&self) -> anyhow::Result<&Vec<ConstantPoolIndex>> {
        match self {
            ProgramObject::Class(members) => Ok(members),
            _ => anyhow::bail!("Expecting a program object representing a Class, found `{}`", self),
        }
    }
    pub fn is_slot(&self) -> bool {
        matches!(self, ProgramObject::Slot { .. })
    }
    pub fn as_method(&self) -> anyhow::Result<&Method> {
        match self {
            ProgramObject::Method(method) => Ok(method),
            _ => anyhow::bail!("Expecting a program object representing a Method, found `{}`", self),
        }
    }
    pub fn is_method(&self) -> bool {
        matches!(self, ProgramObject::Method { .. })
    }
    pub fn as_slot_index(&self) -> anyhow::Result<&ConstantPoolIndex> {
        match self {
            ProgramObject::Slot { name } => Ok(name),
            _ => anyhow::bail!("Expecting a program object representing a Slot, found `{}`", self),
        }
    }

    #[rustfmt::skip]
    fn tag(&self) -> u8 {
        match &self {
            ProgramObject::Integer(_)    => 0x00,
            ProgramObject::Null          => 0x01,
            ProgramObject::String(_)     => 0x02,
            ProgramObject::Method(_)     => 0x03,
            ProgramObject::Slot {name:_} => 0x04,
            ProgramObject::Class(_)      => 0x05,
            ProgramObject::Boolean(_)    => 0x06,
        }
    }

    #[allow(dead_code)]
    pub fn null() -> Self {
        ProgramObject::Null
    }

    #[allow(dead_code)]
    pub fn from_bool(b: bool) -> Self {
        ProgramObject::Boolean(b)
    }

    pub fn from_str(string: &str) -> Self {
        ProgramObject::String(string.to_string())
    }

    #[allow(dead_code)]
    pub fn from_string(string: String) -> Self {
        ProgramObject::String(string)
    }

    #[allow(dead_code)]
    pub fn from_i32(n: i32) -> Self {
        ProgramObject::Integer(n)
    }

    #[allow(dead_code)]
    pub fn from_usize(n: usize) -> Self {
        ProgramObject::Integer(n as i32)
    }

    pub fn slot_from_index(index: ConstantPoolIndex) -> Self {
        ProgramObject::Slot { name: index }
    }

    #[allow(dead_code)]
    pub fn slot_from_u16(index: u16) -> Self {
        ProgramObject::Slot {
            name: ConstantPoolIndex::new(index),
        }
    }

    #[allow(dead_code)]
    pub fn class_from_vec(indices: Vec<u16>) -> Self {
        ProgramObject::Class(indices.iter().map(|n| ConstantPoolIndex::new(*n)).collect())
    }
}

impl ConstantPoolIndex {
    pub fn new(value: u16) -> ConstantPoolIndex {
        ConstantPoolIndex(value)
    }

    pub fn value(&self) -> u16 {
        self.0
    }

    pub fn from_usize(value: usize) -> ConstantPoolIndex {
        assert!(value <= 65535usize);
        ConstantPoolIndex(value as u16)
    }

    pub fn as_usize(&self) -> usize {
        self.value() as usize
    }

    pub fn read_cpi_vector<R: Read>(input: &mut R) -> Vec<ConstantPoolIndex> {
        serializable::read_u16_vector(input)
            .into_iter()
            .map(ConstantPoolIndex::new)
            .collect()
    }

    pub fn write_cpi_vector<R: Write>(sink: &mut R, vector: &[ConstantPoolIndex]) -> anyhow::Result<()> {
        let vector_of_u16s: Vec<u16> = vector.iter().map(|cpi| cpi.0).collect();
        serializable::write_u16_vector(sink, &vector_of_u16s)
    }
}

impl From<u16> for ConstantPoolIndex {
    fn from(n: u16) -> Self {
        ConstantPoolIndex(n)
    }
}

impl From<usize> for ConstantPoolIndex {
    fn from(n: usize) -> Self {
        assert!(n <= 65535usize);
        ConstantPoolIndex(n as u16)
    }
}

impl LocalFrameIndex {
    #[allow(dead_code)]
    pub fn new(value: u16) -> LocalFrameIndex {
        LocalFrameIndex(value)
    }

    pub fn value(&self) -> u16 {
        self.0
    }

    pub fn from_usize(value: usize) -> LocalFrameIndex {
        assert!(value <= 65535usize);
        LocalFrameIndex(value as u16)
    }
}

impl Arity {
    pub fn new(value: u8) -> Arity {
        Arity(value)
    }

    pub fn value(&self) -> u8 {
        self.0
    }

    pub fn from_usize(value: usize) -> Arity {
        assert!(value <= 255usize);
        Arity(value as u8)
    }

    pub fn to_usize(self) -> usize {
        self.0 as usize
    }
}

impl Size {
    #[allow(dead_code)]
    pub fn new(value: u16) -> Size {
        Size(value)
    }

    pub fn value(&self) -> u16 {
        self.0
    }

    pub fn from_usize(value: usize) -> Size {
        assert!(value <= 65535usize);
        Size(value as u16)
    }

    pub fn to_usize(self) -> usize {
        self.0 as usize
    }
}

impl Address {
    #[allow(dead_code)]
    pub fn from_u32(value: u32) -> Address {
        Address(value)
    }

    pub fn from_usize(value: usize) -> Address {
        assert!(value <= 4_294_967_295usize);
        Address(value as u32)
    }

    pub fn value_u32(&self) -> u32 {
        self.0
    }

    pub fn value_usize(&self) -> usize {
        self.0 as usize
    }

    #[allow(dead_code)]
    pub fn offset(&self, n: usize) -> Self {
        Address::from_usize(self.value_usize() + n)
    }
}

impl AddressRange {
    pub fn new(start: Address, length: usize) -> Self {
        AddressRange { start, length }
    }

    #[allow(dead_code)]
    pub fn from(start: usize, length: usize) -> Self {
        AddressRange {
            start: Address::from_usize(start),
            length,
        }
    }

    #[allow(dead_code)]
    pub fn from_addresses(start: Address, end: Address) -> Self {
        AddressRange {
            start,
            length: end.value_usize() - start.value_usize() + 1,
        }
    }

    pub fn start(&self) -> &Address {
        &self.start
    }

    pub fn length(&self) -> usize {
        self.length
    }
}

impl Serializable for Program {
    fn serialize<W: Write>(&self, sink: &mut W) -> anyhow::Result<()> {
        self.constant_pool.serialize(sink, &self.code)?;
        self.globals.serialize(sink)?;
        self.entry.serialize(sink)
    }

    fn from_bytes<R: Read>(input: &mut R) -> Self {
        let mut code = Code::new();

        let constant_pool = ConstantPool::from_bytes(input, &mut code);
        let globals = Globals::from_bytes(input);
        let entry = Entry::from_bytes(input);

        let label_names = code.labels();
        let label_constants = constant_pool.get_all(label_names).unwrap().into_iter(); // LATER(kondziu) error handling
        let label_addresses = code.label_addresses().into_iter();
        let labels = Labels::from(label_constants.zip(label_addresses)).unwrap();

        Program {
            constant_pool,
            labels,
            code,
            globals,
            entry,
        }
    }
}

impl SerializableWithContext for ConstantPool {
    fn serialize<W: Write>(&self, sink: &mut W, code: &Code) -> Result<()> {
        serializable::write_usize_as_u16(sink, self.0.len())?;
        self.0
            .iter()
            .try_for_each(|program_object| program_object.serialize(sink, code))
    }

    fn from_bytes<R: Read>(input: &mut R, code: &mut Code) -> Self {
        let size = serializable::read_u16_as_usize(input);
        let constants: Vec<ProgramObject> = (0..size).map(|_| ProgramObject::from_bytes(input, code)).collect();

        ConstantPool(constants)
    }
}

impl Serializable for Globals {
    fn serialize<W: Write>(&self, sink: &mut W) -> Result<(), Error> {
        ConstantPoolIndex::write_cpi_vector(sink, &self.0)
    }

    fn from_bytes<R: Read>(input: &mut R) -> Self {
        Globals(ConstantPoolIndex::read_cpi_vector(input))
    }
}

impl Serializable for Entry {
    fn serialize<W: Write>(&self, sink: &mut W) -> Result<(), Error> {
        self.0.expect("Cannot serialize an empty entry point.").serialize(sink)
    }

    fn from_bytes<R: Read>(input: &mut R) -> Self {
        Entry(Some(ConstantPoolIndex::from_bytes(input)))
    }
}

impl SerializableWithContext for ProgramObject {
    fn serialize<W: Write>(&self, sink: &mut W, code: &Code) -> anyhow::Result<()> {
        serializable::write_u8(sink, self.tag())?;
        match &self {
            ProgramObject::Null => Ok(()),
            ProgramObject::Integer(n) => serializable::write_i32(sink, *n),
            ProgramObject::Boolean(b) => serializable::write_bool(sink, *b),
            ProgramObject::String(s) => serializable::write_utf8(sink, s),
            ProgramObject::Class(v) => ConstantPoolIndex::write_cpi_vector(sink, v),
            ProgramObject::Slot { name } => name.serialize(sink),

            ProgramObject::Method(method) => {
                method.name.serialize(sink)?;
                method.parameters.serialize(sink)?;
                method.locals.serialize(sink)?;
                OpCode::write_opcode_vector(sink, &code.materialize(&method.code)?)
            }
        }
    }

    fn from_bytes<R: Read>(input: &mut R, code: &mut Code) -> Self {
        // LATER(kondziu) error handling
        let tag = serializable::read_u8(input);
        match tag {
            0x00 => ProgramObject::Integer(serializable::read_i32(input)),
            0x01 => ProgramObject::Null,
            0x02 => ProgramObject::String(serializable::read_utf8(input)),
            0x03 => ProgramObject::Method(Method {
                name: ConstantPoolIndex::from_bytes(input),
                parameters: Arity::from_bytes(input),
                locals: Size::from_bytes(input),
                code: code.append(OpCode::read_opcode_vector(input)),
            }),
            0x04 => ProgramObject::Slot {
                name: ConstantPoolIndex::from_bytes(input),
            },
            0x05 => ProgramObject::Class(ConstantPoolIndex::read_cpi_vector(input)),
            0x06 => ProgramObject::Boolean(serializable::read_bool(input)),
            _ => panic!("Cannot deserialize value: unrecognized value tag: {}", tag),
        }
    }
}

impl Serializable for ConstantPoolIndex {
    fn serialize<W: Write>(&self, sink: &mut W) -> anyhow::Result<()> {
        serializable::write_u16(sink, self.0)
    }
    fn from_bytes<R: Read>(input: &mut R) -> Self {
        ConstantPoolIndex(serializable::read_u16(input))
    }
}

impl Serializable for LocalFrameIndex {
    fn serialize<W: Write>(&self, sink: &mut W) -> anyhow::Result<()> {
        serializable::write_u16(sink, self.0)
    }
    fn from_bytes<R: Read>(input: &mut R) -> Self {
        LocalFrameIndex(serializable::read_u16(input))
    }
}

impl Serializable for Arity {
    fn serialize<W: Write>(&self, sink: &mut W) -> anyhow::Result<()> {
        serializable::write_u8(sink, self.0)
    }

    fn from_bytes<R: Read>(input: &mut R) -> Self {
        Arity(serializable::read_u8(input))
    }
}

impl Serializable for Size {
    fn serialize<W: Write>(&self, sink: &mut W) -> anyhow::Result<()> {
        serializable::write_u16(sink, self.0)
    }

    fn from_bytes<R: Read>(input: &mut R) -> Self {
        Size(serializable::read_u16(input))
    }
}

impl Serializable for Address {
    fn serialize<W: Write>(&self, sink: &mut W) -> anyhow::Result<()> {
        serializable::write_u32(sink, self.0)
    }
    fn from_bytes<R: Read>(input: &mut R) -> Self {
        Address(serializable::read_u32(input))
    }
}

impl Display for Program {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Constant Pool:")?;
        write!(f, "{}", self.constant_pool)?;
        writeln!(f, "Globals:")?;
        write!(f, "{}", self.globals)?;
        writeln!(f, "Entry: {}", self.entry)?;
        writeln!(f, "Code:")?;
        write!(f, "{}", self.code)?;
        Ok(())
    }
}

impl Display for ConstantPool {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        for (i, program_object) in self.0.iter().enumerate() {
            writeln!(f, "{}: {}", i, program_object)?;
        }
        Ok(())
    }
}

impl Display for Globals {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        for (i, global) in self.0.iter().enumerate() {
            writeln!(f, "{}: {}", i, global)?;
        }
        Ok(())
    }
}

impl Display for Entry {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.0.as_ref().map_or(Ok(()), |index| write!(f, "{}", index))
    }
}

impl Display for Code {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (i, opcode) in self.0.iter().enumerate() {
            writeln!(f, "{}: {}", i, opcode)?;
        }
        Ok(())
    }
}

impl Display for ProgramObject {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ProgramObject::Integer(n) => write!(f, "{}", n),
            ProgramObject::Boolean(b) => write!(f, "{}", b),
            ProgramObject::Null => write!(f, "null"),
            ProgramObject::String(s) => write!(f, "\"{}\"", s),
            ProgramObject::Slot { name } => write!(f, "slot {}", name),
            ProgramObject::Method(method) => {
                write!(
                    f,
                    "method {} args:{} locals:{} {}",
                    method.name, method.parameters, method.locals, method.code
                )
            }
            ProgramObject::Class(members) => {
                let members = members.iter().map(|i| i.to_string()).collect::<Vec<String>>().join(",");
                write!(f, "class {}", members)
            }
        }
    }
}

impl Display for ConstantPoolIndex {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

impl Display for LocalFrameIndex {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "::{}", self.0)
    }
}

impl Display for Arity {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Display for Size {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Display for Address {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{number:>0width$}", number = self.0, width = 4)
    }
}

impl Display for AddressRange {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.length == 0 {
            write!(f, "{}-∅", self.start)
        } else {
            write!(
                f,
                "{}-{}",
                self.start,
                Address::from_usize(self.start.value_usize() + self.length - 1)
            )
        }
    }
}
