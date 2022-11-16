use std::fmt::{self, Display, Formatter};
use std::fs::{self, File};
use std::io::Write;
use std::mem;
use std::path::PathBuf;
use std::time::SystemTime;

use anyhow::{anyhow, bail, ensure, Context, Result};
use indexmap::IndexMap;

use crate::bytecode::program::{AddressRange, Arity, ConstantPoolIndex, ProgramObject, Size};
use crate::bytecode::state::OperandStack;

#[derive(Debug, Default)]
pub struct Heap {
    gc_size: Option<usize>,
    size: usize,
    log: Option<File>,
    memory: Vec<HeapObject>,
}

impl Eq for Heap {}
impl PartialEq for Heap {
    fn eq(&self, other: &Self) -> bool {
        self.memory.eq(&other.memory)
    }
}

impl Heap {
    pub fn new() -> Self {
        Heap::default()
    }

    pub fn set_gc_size(&mut self, size: Option<usize>) {
        self.gc_size = size;
    }

    pub fn set_log(&mut self, path: PathBuf) {
        let mut dir = path.clone();
        dir.pop();
        fs::create_dir_all(dir).unwrap();

        let mut file = File::create(path).unwrap();
        writeln!(file, "timestamp,event,heap").unwrap();

        self.log = Some(file);
        self.log_start();
    }

    fn log_start(&mut self) {
        if let Some(file) = &mut self.log {
            let timestamp = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            write!(file, "{},S,0\n", timestamp).unwrap();
        }
    }

    fn log_allocate(&mut self, memory: usize) {
        if let Some(file) = &mut self.log {
            let timestamp = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            write!(file, "{},A,{}\n", timestamp, memory).unwrap();
        }
    }

    #[allow(dead_code)]
    fn log_gc(&mut self, memory: usize) {
        if let Some(file) = &mut self.log {
            let timestamp = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            write!(file, "{},G,{}\n", timestamp, memory).unwrap();
        }
    }

    pub fn allocate(&mut self, object: HeapObject) -> HeapIndex {
        self.size += object.size();
        //dbg!(object.size(), &object); // LATER(martin-t) Remove
        self.log_allocate(self.size);
        let index = HeapIndex::from(self.memory.len());
        self.memory.push(object);
        index
    }

    pub fn dereference(&self, index: &HeapIndex) -> Result<&HeapObject> {
        self.memory
            .get(index.as_usize())
            .with_context(|| format!("Cannot dereference object from the heap at index: `{}`", index))
    }

    pub fn dereference_mut(&mut self, index: &HeapIndex) -> Result<&mut HeapObject> {
        self.memory
            .get_mut(index.as_usize())
            .with_context(|| format!("Cannot dereference object from the heap at index: `{}`", index))
    }
}

impl From<Vec<HeapObject>> for Heap {
    fn from(objects: Vec<HeapObject>) -> Self {
        Heap {
            gc_size: None,
            size: objects.iter().map(|o| o.size()).sum(),
            log: None,
            memory: objects,
        }
    }
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub enum HeapObject {
    Array(ArrayInstance),
    Object(ObjectInstance),
}

impl HeapObject {
    #[allow(dead_code)]
    pub fn new_object(
        parent: Pointer,
        fields: IndexMap<String, Pointer>,
        methods: IndexMap<String, ProgramObject>,
    ) -> Self {
        HeapObject::Object(ObjectInstance {
            parent,
            fields,
            methods,
        })
    }
    pub fn as_object_instance(&self) -> Result<&ObjectInstance> {
        match self {
            HeapObject::Object(instance) => Ok(instance),
            array => Err(anyhow!("Attempt to cast an array as an object instance `{}`.", array)),
        }
    }
    pub fn as_object_instance_mut(&mut self) -> Result<&mut ObjectInstance> {
        match self {
            HeapObject::Object(instance) => Ok(instance),
            array => Err(anyhow!("Attempt to cast an array as an object instance `{}`.", array)),
        }
    }
    #[allow(dead_code)]
    pub fn empty_object() -> Self {
        HeapObject::Object(ObjectInstance::new())
    }
    #[allow(dead_code)]
    pub fn empty_array() -> Self {
        HeapObject::Array(ArrayInstance::new())
    }
    pub fn from_pointers(v: Vec<Pointer>) -> Self {
        HeapObject::Array(ArrayInstance::from(v))
    }
    #[allow(dead_code)]
    pub fn from(parent: Pointer, fields: IndexMap<String, Pointer>, methods: IndexMap<String, ProgramObject>) -> Self {
        HeapObject::Object(ObjectInstance {
            parent,
            fields,
            methods,
        })
    }
    pub fn evaluate_as_string(&self, heap: &Heap) -> Result<String> {
        match self {
            HeapObject::Array(array) => array.evaluate_as_string(heap),
            HeapObject::Object(object) => object.evaluate_as_string(heap),
        }
    }
    pub fn size(&self) -> usize {
        match self {
            HeapObject::Array(array) => mem::size_of::<ArrayInstance>() + array.length() * mem::size_of::<Pointer>(),
            HeapObject::Object(object) => {
                let header = mem::size_of::<ObjectInstance>();
                let fields: usize = object
                    .fields
                    .iter()
                    .map(|(string, _pointer)| string.len() + mem::size_of::<Pointer>())
                    .sum();
                let methods: usize = object
                    .methods
                    .iter()
                    .map(|(string, program_object)| {
                        string.len()
                            + match program_object {
                                // TODO Maybe this should store just const pool index?
                                // TODO Maybe simplify - check variant and do mem::size_of on program_object?
                                ProgramObject::Method { .. } => {
                                    mem::size_of::<ConstantPoolIndex>()
                                        + mem::size_of::<Arity>()
                                        + mem::size_of::<Size>()
                                        + mem::size_of::<AddressRange>()
                                }
                                _ => unreachable!(),
                            }
                    })
                    .sum();
                header + fields + methods
            }
        }
    }
}

impl Display for HeapObject {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            HeapObject::Array(array) => write!(f, "{}", array),
            HeapObject::Object(object) => write!(f, "{}", object),
        }
    }
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Clone, Default)]
pub struct ArrayInstance(Vec<Pointer>);

impl ArrayInstance {
    #[allow(dead_code)]
    pub fn new() -> Self {
        ArrayInstance(vec![])
    }
    #[allow(dead_code)]
    pub fn iter(&self) -> impl Iterator<Item = &Pointer> {
        self.0.iter()
    }
    #[allow(dead_code)]
    pub fn length(&self) -> usize {
        self.0.len()
    }
    pub fn get_element(&self, index: usize) -> Result<&Pointer> {
        let length = self.0.len();
        ensure!(
            index < length,
            "Index out of range {} for array `{}` with length {}",
            index,
            self,
            length
        );
        Ok(&self.0[index])
    }
    pub fn set_element(&mut self, index: usize, value_pointer: Pointer) -> Result<&Pointer> {
        let length = self.0.len();
        ensure!(
            index < length,
            "Index out of range {} for array `{}` with length {}",
            index,
            self,
            length
        );
        self.0[index] = value_pointer;
        Ok(&self.0[index])
    }
    pub fn evaluate_as_string(&self, heap: &Heap) -> Result<String> {
        let elements = self
            .0
            .iter()
            .map(|element| element.evaluate_as_string(heap))
            .collect::<Result<Vec<String>>>()?;
        Ok(format!("[{}]", elements.join(", ")))
    }
}

impl From<Vec<Pointer>> for ArrayInstance {
    fn from(v: Vec<Pointer>) -> Self {
        ArrayInstance(v)
    }
}

impl Display for ArrayInstance {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.0
                .iter()
                .map(|pointer| format!("{}", pointer))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

#[derive(Eq, PartialEq, Debug, Clone, Default)]
pub struct ObjectInstance {
    pub parent: Pointer,
    pub fields: IndexMap<String, Pointer>,        // LATER(kondziu) make private
    pub methods: IndexMap<String, ProgramObject>, // LATER(kondziu) make private
}

impl ObjectInstance {
    #[allow(dead_code)]
    pub fn new() -> Self {
        ObjectInstance::default()
    }
    pub fn get_field(&self, name: &str) -> Result<&Pointer> {
        self.fields
            .get(name)
            .with_context(|| format!("There is no field named `{}` in object `{}`", name, self))
    }
    pub fn set_field(&mut self, name: &str, pointer: Pointer) -> Result<Pointer> {
        self.fields
            .insert(name.to_owned(), pointer)
            .with_context(|| format!("There is no field named `{}` in object `{}`", name, self))
    }
    pub fn evaluate_as_string(&self, heap: &Heap) -> Result<String> {
        let parent = match self.parent {
            Pointer::Null => None,
            parent => Some(parent.evaluate_as_string(heap)?),
        };

        // Sort fields in lexographical order
        let mut sorted_fields: Vec<(&String, &Pointer)> = self.fields.iter().collect();
        sorted_fields.sort_by_key(|(name, _)| *name);

        let fields = sorted_fields
            .into_iter()
            .map(|(name, value)| {
                value
                    .evaluate_as_string(heap)
                    .map(|value| format!("{}={}", name, value))
            })
            .collect::<Result<Vec<String>>>()?;

        match parent {
            Some(parent) if fields.is_empty() => Ok(format!("object(..={})", parent)),
            Some(parent) => Ok(format!("object(..={}, {})", parent, fields.join(", "))),
            None => Ok(format!("object({})", fields.join(", "))),
        }
    }
}

impl Display for ObjectInstance {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let parent = match self.parent {
            Pointer::Null => None,
            parent => Some(parent.to_string()),
        };

        let fields = self
            .fields
            .iter()
            .map(|(name, value)| format!("{}={}", name, value))
            .collect::<Vec<String>>();

        match parent {
            Some(parent) => write!(f, "object(..={}, {})", parent, fields.join(", ")),
            None => write!(f, "object({})", fields.join(", ")),
        }
    }
}

#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Ord, PartialOrd)]
pub struct HeapIndex(usize);

impl From<usize> for HeapIndex {
    fn from(n: usize) -> Self {
        HeapIndex(n)
    }
}

impl From<&Pointer> for HeapIndex {
    fn from(p: &Pointer) -> Self {
        match p {
            Pointer::Reference(p) => *p,
            Pointer::Null => panic!("Cannot create heap reference from a null-tagged pointer"),
            Pointer::Integer(_) => panic!("Cannot create heap reference from an integer-tagged pointer"),
            Pointer::Boolean(_) => panic!("Cannot create heap reference from a boolean-tagged pointer"),
        }
    }
}

impl From<Pointer> for HeapIndex {
    fn from(p: Pointer) -> Self {
        match p {
            Pointer::Reference(p) => p,
            Pointer::Null => panic!("Cannot create heap reference from a null-tagged pointer"),
            Pointer::Integer(_) => panic!("Cannot create heap reference from an integer-tagged pointer"),
            Pointer::Boolean(_) => panic!("Cannot create heap reference from a boolean-tagged pointer"),
        }
    }
}

impl HeapIndex {
    pub fn as_usize(&self) -> usize {
        self.0
    }
}

impl Display for HeapIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:x>8}", self.0)
    }
}

#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Ord, PartialOrd, Default)]
pub enum Pointer {
    #[default]
    Null,
    Integer(i32),
    Boolean(bool),
    Reference(HeapIndex),
}

impl Pointer {
    pub fn push_onto(self, stack: &mut OperandStack) {
        stack.push(self);
    }
}

impl Pointer {
    pub fn from_literal(program_object: &ProgramObject) -> Result<Pointer> {
        match program_object {
            ProgramObject::Null => Ok(Self::Null),
            ProgramObject::Integer(value) => Ok(Self::Integer(*value)),
            ProgramObject::Boolean(value) => Ok(Self::Boolean(*value)),
            _ => bail!(
                "Expecting either a null, an integer, or a boolean, but found `{}`.",
                program_object
            ),
        }
    }
    #[allow(dead_code)]
    pub fn is_heap_reference(&self) -> bool {
        matches!(self, Pointer::Reference(_))
    }
    #[allow(dead_code)]
    pub fn as_heap_reference(&self) -> Option<&HeapIndex> {
        match self {
            Pointer::Reference(reference) => Some(reference),
            _ => None,
        }
    }
    pub fn into_heap_reference(self) -> Result<HeapIndex> {
        match self {
            Pointer::Reference(reference) => Ok(reference),
            pointer => Err(anyhow!("Expecting a heap reference, but found `{}`.", pointer)),
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Pointer::Null)
    }
    #[allow(dead_code)]
    pub fn as_null(&self) -> Option<()> {
        match self {
            Pointer::Null => Some(()),
            _ => None,
        }
    }
    #[allow(dead_code)]
    pub fn is_i32(&self) -> bool {
        matches!(self, Pointer::Integer(_))
    }
    pub fn as_i32(&self) -> Result<i32> {
        match self {
            Pointer::Integer(i) => Ok(*i),
            pointer => Err(anyhow!("Expecting an integer, but found `{}`", pointer)),
        }
    }
    pub fn as_usize(&self) -> Result<usize> {
        match self {
            Pointer::Integer(i) if *i >= 0 => Ok(*i as usize),
            pointer => Err(anyhow!("Expecting a positive integer, but found `{}`", pointer)),
        }
    }
    #[allow(dead_code)]
    pub fn is_bool(&self) -> bool {
        matches!(self, Pointer::Boolean(_))
    }
    #[allow(dead_code)]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Pointer::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    pub fn evaluate_as_condition(&self) -> bool {
        match self {
            Pointer::Null => false,
            Pointer::Integer(_) => true,
            Pointer::Boolean(b) => *b,
            Pointer::Reference(_) => true,
        }
    }

    pub fn evaluate_as_string(&self, heap: &Heap) -> Result<String> {
        // LATER(kondziu) trait candidate
        match self {
            Pointer::Null => Ok("null".to_owned()),
            Pointer::Integer(i) => Ok(i.to_string()),
            Pointer::Boolean(b) => Ok(b.to_string()),
            Pointer::Reference(index) => heap.dereference(index)?.evaluate_as_string(heap),
        }
    }
}

impl From<Pointer> for bool {
    fn from(val: Pointer) -> Self {
        match val {
            Pointer::Boolean(b) => b,
            p => panic!("Cannot cast `{}` into a boolean pointer.", p),
        }
    }
}

impl From<Pointer> for i32 {
    fn from(val: Pointer) -> Self {
        match val {
            Pointer::Integer(i) => i,
            p => panic!("Cannot cast `{}` into an integer pointer.", p),
        }
    }
}

impl From<&ProgramObject> for Pointer {
    fn from(constant: &ProgramObject) -> Self {
        match constant {
            ProgramObject::Null => Self::Null,
            ProgramObject::Integer(value) => Self::Integer(*value),
            ProgramObject::Boolean(value) => Self::Boolean(*value),
            _ => unimplemented!(),
        }
    }
}

impl From<ProgramObject> for Pointer {
    fn from(constant: ProgramObject) -> Self {
        match constant {
            ProgramObject::Null => Self::Null,
            ProgramObject::Integer(value) => Self::Integer(value),
            ProgramObject::Boolean(value) => Self::Boolean(value),
            _ => unimplemented!(),
        }
    }
}

impl From<&HeapIndex> for Pointer {
    fn from(p: &HeapIndex) -> Self {
        Pointer::Reference(*p)
    }
}

impl From<HeapIndex> for Pointer {
    fn from(p: HeapIndex) -> Self {
        Pointer::Reference(p)
    }
}

impl From<usize> for Pointer {
    fn from(n: usize) -> Self {
        Pointer::from(HeapIndex::from(n))
    }
}

impl From<i32> for Pointer {
    fn from(i: i32) -> Self {
        Pointer::Integer(i)
    }
}

impl From<bool> for Pointer {
    fn from(b: bool) -> Self {
        Pointer::Boolean(b)
    }
}

impl Display for Pointer {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Pointer::Null => write!(f, "null"),
            Pointer::Integer(i) => write!(f, "{}", i),
            Pointer::Boolean(b) => write!(f, "{}", b),
            Pointer::Reference(p) => write!(f, "{}", p),
        }
    }
}
