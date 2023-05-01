//! Completely untested executable memory for Windows.
//!
//! LATER(martin-t) Test on Windows.

use std::mem::ManuallyDrop;

use windows_sys::Win32::System::Memory::{VirtualProtect, PAGE_EXECUTE_READWRITE, PAGE_READWRITE};

pub struct JitMemory {
    pub code: *mut u8,
    len: usize,
    cap: usize,
}

impl JitMemory {
    pub fn new(data: &[u8]) -> Self {
        // Inspiration: https://github.com/mrowqa/brainfuck-jit/blob/master/src/vm/jitmem.rs

        // LATER(martin-t) Allocate this in a separate page so we can enforce W^X.
        //  Maybe VirtualAlloc?
        let vec = data.to_vec();
        let mut vec = ManuallyDrop::new(vec);
        let code = vec.as_mut_ptr();
        let len = vec.len();
        let cap = vec.capacity();

        // Currently we have to use PAGE_EXECUTE_READWRITE because
        // the rest of the page might contain data we need to write.
        // LATER(martin-t) Use PAGE_EXECUTE_READ
        let mut old_prot = 0u32;
        let ret = unsafe { VirtualProtect(code as *mut _, len, PAGE_EXECUTE_READWRITE, &mut old_prot as *mut _) };
        // Yes, 0 really means failure and anything else means success.
        assert_ne!(ret, 0);
        Self { code, len, cap }
    }
}

impl Drop for JitMemory {
    fn drop(&mut self) {
        // Undo VirtualProtext.
        let mut old_prot = 0u32;
        let ret = unsafe { VirtualProtect(self.code as *mut _, self.len, PAGE_READWRITE, &mut old_prot as *mut _) };
        assert_ne!(ret, 0);

        // Reconstruct and drop the Vec.
        unsafe {
            Vec::from_raw_parts(self.code, self.len, self.cap);
        }
    }
}
