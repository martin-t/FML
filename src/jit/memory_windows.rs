pub struct JitMemory {
    pub code: *mut u8,
    size: usize,
}

impl JitMemory {
    #[allow(unused_variables)] // LATER(martin-t) Remove
    pub fn new(data: &[u8]) -> Self {
        unimplemented!("Windows is currently not supported")
    }
}
