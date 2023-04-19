//! Completely untested executable memory for Windows.
//!
//! LATER(martin-t) Test on Windows.

use windows_sys::Win32::System::Memory::{VirtualProtect, PAGE_EXECUTE_READWRITE};

pub struct JitMemory {
    pub code: *mut u8,
}

impl JitMemory {
    pub fn new(data: &[u8]) -> Self {
        let vec = data.to_vec();
        let size = vec.len();
        let code = vec.leak().as_mut_ptr();
        let mut old_prot = 0u32;
        let ret = unsafe { VirtualProtect(code as *mut _, size, PAGE_EXECUTE_READWRITE, &mut old_prot as *mut _) };
        assert_ne!(ret, 0);
        Self { code }
    }
}

impl Drop for JitMemory {
    fn drop(&mut self) {
        // LATER(martin-t) Currently we're leaking the data.
        //  Undo VirtualProtext, then drop the Vec.
        //  Will probably need to use from_raw_parts which is not stable yet.
    }
}
