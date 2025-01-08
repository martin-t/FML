//! Executable memory implementation for Windows with proper page allocation and W^X protection.

use std::ptr::null_mut;
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, VirtualProtect, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_READWRITE,
};

/// Note this can't be tested by miri
/// because miri doesn't support some of the functions used.
pub struct JitMemory {
    pub code: *mut u8,
    size: usize,
}

const PAGE_SIZE: usize = 4096;
const INT3: u8 = 0xCC;

impl JitMemory {
    pub fn new(data: &[u8]) -> Self {
        // Match Unix behavior: allocate extra page to catch overruns with INT3
        let num_pages = data.len() / PAGE_SIZE + 1;
        let size = num_pages * PAGE_SIZE;

        unsafe {
            // Allocate with initial RW protection
            let memptr = VirtualAlloc(null_mut(), size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            assert!(!memptr.is_null());

            // Fill with INT3 to catch invalid execution
            std::ptr::write_bytes(memptr as *mut u8, INT3, size);

            // Copy the actual code
            std::ptr::copy_nonoverlapping(data.as_ptr(), memptr as *mut u8, data.len());

            // W^X: Remove write, add execute
            let mut old_prot = 0u32;
            let ret = VirtualProtect(memptr, size, PAGE_EXECUTE_READ, &mut old_prot);
            assert_ne!(ret, 0);

            Self {
                code: memptr as *mut u8,
                size,
            }
        }
    }
}

impl Drop for JitMemory {
    fn drop(&mut self) {
        unsafe {
            // W^X: Remove execute before freeing
            let mut old_prot = 0u32;
            let ret = VirtualProtect(self.code as *mut _, self.size, PAGE_READWRITE, &mut old_prot);
            assert_ne!(ret, 0);

            // Free the allocated memory
            let ret = VirtualFree(self.code as *mut _, 0, MEM_RELEASE);
            assert_ne!(ret, 0);
        }
    }
}
