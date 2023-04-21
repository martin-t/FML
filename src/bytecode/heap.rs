use std::{
    fmt::{self, Display, Formatter},
    fs::{self, File},
    io::Write,
    mem,
    path::PathBuf,
    time::SystemTime,
};

use anyhow::{anyhow, bail, ensure, Context, Result};
use indexmap::IndexMap;

use crate::bytecode::{
    program::{ConstantPoolIndex, ProgramObject},
    state::{FrameStack, OperandStack},
};

#[derive(Debug, Default)]
pub struct Heap {
    gc_size: Option<usize>,
    size: usize,
    log: Option<File>,
    memory: Vec<HeapObject>,
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub enum HeapObject {
    Array(ArrayInstance),
    Object(ObjectInstance),
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Clone, Default)]
pub struct ArrayInstance(Vec<Pointer>);

#[derive(Eq, PartialEq, Debug, Clone, Default)]
pub struct ObjectInstance {
    pub parent: Pointer,
    pub fields: IndexMap<String, Pointer>,            // LATER(kondziu) make private
    pub methods: IndexMap<String, ConstantPoolIndex>, // LATER(kondziu) make private
}

/// This should really be called a Value but the reference calls it a Pointer.
#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Ord, PartialOrd, Default)]
pub enum Pointer {
    #[default]
    Null,
    Integer(i32),
    Boolean(bool),
    Reference(HeapIndex),
}

#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy, Ord, PartialOrd)]
pub struct HeapIndex(usize);

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
            writeln!(file, "{timestamp},S,0").unwrap();
        }
    }

    fn log_allocate(&mut self) {
        if let Some(file) = &mut self.log {
            let timestamp = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            writeln!(file, "{},A,{}", timestamp, self.size).unwrap();
        }
    }

    fn log_gc(&mut self) {
        if let Some(file) = &mut self.log {
            let timestamp = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            writeln!(file, "{},G,{}", timestamp, self.size).unwrap();
        }
    }

    pub fn allocate(
        &mut self,
        frame_stack: &mut FrameStack,
        operand_stack: &mut OperandStack,
        object: HeapObject,
    ) -> HeapIndex {
        //dbg!(object.size(), &object); // LATER(martin-t) Remove

        if let Some(gc_size) = self.gc_size {
            if self.size + object.size() > gc_size {
                self.gc(frame_stack, operand_stack);
                self.log_gc();
            }
        }

        self.size += object.size();
        self.log_allocate();
        let index = HeapIndex::from(self.memory.len());
        self.memory.push(object);
        index
    }

    fn gc(&mut self, frame_stack: &mut FrameStack, operand_stack: &mut OperandStack) {
        #![allow(clippy::needless_range_loop)] // I just find this easier to read

        use std::collections::VecDeque;

        #[derive(Debug)]
        struct Gc {
            marks: Vec<bool>,
            to_visit: VecDeque<HeapIndex>,
        }

        impl Gc {
            fn enque(&mut self, pointer: Pointer) {
                if let Pointer::Reference(index) = pointer {
                    if !self.marks[index.0] {
                        self.to_visit.push_back(index);
                    }
                    self.marks[index.0] = true;
                }
            }
        }

        let mut gc = Gc {
            marks: vec![false; self.memory.len()],
            to_visit: VecDeque::new(),
        };

        // Mark roots
        for &global in frame_stack.globals.values() {
            gc.enque(global);
        }
        for frame in frame_stack.frames() {
            for &local in frame.locals() {
                gc.enque(local);
            }
        }
        for &operand in operand_stack.values() {
            gc.enque(operand);
        }

        // Mark everything reachable using BFS
        // LATER(martin-t) Why BFS? DFS could be marginally faster?
        while let Some(index) = gc.to_visit.pop_front() {
            let heap_object = &self.memory[index.0];
            match heap_object {
                HeapObject::Array(array) => {
                    for &element in &array.0 {
                        gc.enque(element);
                    }
                }
                HeapObject::Object(object) => {
                    gc.enque(object.parent);
                    for &field in object.fields.values() {
                        gc.enque(field);
                    }
                }
            }
        }

        // Nuke everything unreachable so it crashes if there's a bug
        // LATER(martin-t) Make optional: --debug flag?
        for index in 0..gc.marks.len() {
            if !gc.marks[index] {
                self.memory[index] = HeapObject::Object(ObjectInstance {
                    parent: Pointer::Integer(666),
                    fields: IndexMap::new(),
                    methods: IndexMap::new(),
                });
            }
        }

        // Sweep - compute forwarding pointers
        let mut forwarding = vec![0; self.memory.len()];
        let mut new_index = 0;
        for old_index in 0..gc.marks.len() {
            if gc.marks[old_index] {
                forwarding[old_index] = new_index;
                new_index += 1;
            }
        }
        let new_length = new_index;

        // Sweep - update pointers
        let rewrite_pointer = |pointer: &mut Pointer| {
            if let Pointer::Reference(index) = pointer {
                let new_index = forwarding[index.0];
                index.0 = new_index;
            }
        };
        for global in frame_stack.globals.values_mut() {
            rewrite_pointer(global)
        }
        for frame in frame_stack.frames_mut() {
            for local in frame.locals_mut() {
                rewrite_pointer(local);
            }
        }
        for operand in operand_stack.values_mut() {
            rewrite_pointer(operand);
        }
        for index in 0..gc.marks.len() {
            if !gc.marks[index] {
                continue;
            }

            let heap_object = &mut self.memory[index];
            match heap_object {
                HeapObject::Array(array) => {
                    for element in &mut array.0 {
                        rewrite_pointer(element);
                    }
                }
                HeapObject::Object(object) => {
                    rewrite_pointer(&mut object.parent);
                    for field in object.fields.values_mut() {
                        rewrite_pointer(field);
                    }
                }
            }
        }

        // Sweep - compaction
        for old_index in 0..gc.marks.len() {
            if !gc.marks[old_index] {
                continue;
            }

            let new_index = forwarding[old_index];
            // LATER(martin-t) Perf - avoid clone? mem::swap?
            self.memory[new_index] = self.memory[old_index].clone();
        }
        self.memory.truncate(new_length);

        // Recalculate size
        let mut size = 0;
        for heap_object in &self.memory {
            size += heap_object.size();
        }
        self.size = size;
    }

    pub fn dereference(&self, index: &HeapIndex) -> Result<&HeapObject> {
        self.memory
            .get(index.as_usize())
            .with_context(|| format!("Cannot dereference object from the heap at index: `{index}`"))
    }

    pub fn dereference_mut(&mut self, index: &HeapIndex) -> Result<&mut HeapObject> {
        self.memory
            .get_mut(index.as_usize())
            .with_context(|| format!("Cannot dereference object from the heap at index: `{index}`"))
    }
}

impl PartialEq for Heap {
    fn eq(&self, other: &Self) -> bool {
        self.memory.eq(&other.memory)
    }
}

impl Eq for Heap {}

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

impl HeapObject {
    #[allow(dead_code)]
    pub fn new_object(
        parent: Pointer,
        fields: IndexMap<String, Pointer>,
        methods: IndexMap<String, ConstantPoolIndex>,
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
    pub fn from(
        parent: Pointer,
        fields: IndexMap<String, Pointer>,
        methods: IndexMap<String, ConstantPoolIndex>,
    ) -> Self {
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
                    .map(|(name, pointer)| name.len() + mem::size_of_val(pointer))
                    .sum();
                let methods: usize = object
                    .methods
                    .iter()
                    .map(|(name, index)| name.len() + mem::size_of_val(index))
                    .sum();
                header + fields + methods
            }
        }
    }
}

impl Display for HeapObject {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            HeapObject::Array(array) => write!(f, "{array}"),
            HeapObject::Object(object) => write!(f, "{object}"),
        }
    }
}

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
                .map(|pointer| format!("{pointer}"))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

impl ObjectInstance {
    #[allow(dead_code)]
    pub fn new() -> Self {
        ObjectInstance::default()
    }
    pub fn get_field(&self, name: &str) -> Result<&Pointer> {
        self.fields
            .get(name)
            .with_context(|| format!("There is no field named `{name}` in object `{self}`"))
    }
    pub fn set_field(&mut self, name: &str, pointer: Pointer) -> Result<Pointer> {
        self.fields
            .insert(name.to_owned(), pointer)
            .with_context(|| format!("There is no field named `{name}` in object `{self}`"))
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
            .map(|(name, value)| value.evaluate_as_string(heap).map(|value| format!("{name}={value}")))
            .collect::<Result<Vec<String>>>()?;

        match parent {
            Some(parent) if fields.is_empty() => Ok(format!("object(..={parent})")),
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
            .map(|(name, value)| format!("{name}={value}"))
            .collect::<Vec<String>>();

        match parent {
            Some(parent) => write!(f, "object(..={}, {})", parent, fields.join(", ")),
            None => write!(f, "object({})", fields.join(", ")),
        }
    }
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
            p => panic!("Cannot cast `{p}` into a boolean pointer."),
        }
    }
}

impl From<Pointer> for i32 {
    fn from(val: Pointer) -> Self {
        match val {
            Pointer::Integer(i) => i,
            p => panic!("Cannot cast `{p}` into an integer pointer."),
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
            Pointer::Integer(i) => write!(f, "{i}"),
            Pointer::Boolean(b) => write!(f, "{b}"),
            Pointer::Reference(p) => write!(f, "{p}"),
        }
    }
}

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
