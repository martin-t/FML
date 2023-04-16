use libc::{c_int, c_void, PROT_EXEC, PROT_READ, PROT_WRITE};

/// Note this can't be tested by miri
/// because it doesn't support some of the functions used.
///
/// LATER Windows support?
pub struct JitMemory {
    pub code: *mut u8,
    size: usize,
}

const PAGE_SIZE: usize = 4096;
const INT3: c_int = 0xCC;

impl JitMemory {
    pub fn new(data: &[u8]) -> Self {
        // Sooooo, if we need exactly a multiple of PAGE_SIZE,
        // this allocates an extra page. Which is ugly, yes.
        // But it also means if we accidentally keep executing code
        // beyond the end, we'll _always_ reach an INT3 instruction
        // and get SIGTRAP instead of SIGSEGV which is nice(r).
        // So I'll keep it this way.
        // LATER Actually fix this or make it optional.
        //       We could even add a dummy page full of INT3 in front.
        let num_pages = data.len() / PAGE_SIZE + 1;
        let size = num_pages * PAGE_SIZE;

        let mut memptr: *mut c_void = std::ptr::null_mut();

        // SAFETY: TODO
        unsafe {
            // Despite the name, this allocates.
            // OS X requires alignment: https://www.jntrnr.com/building-a-simple-jit-in-rust/
            let ret = libc::posix_memalign(&mut memptr, PAGE_SIZE, size);
            assert_eq!(ret, 0);

            // Executing INT3 causes a crash (trace trap) so we know
            // we got into parts of memory without valid instructions.
            libc::memset(memptr, INT3, size);

            libc::memcpy(memptr, data.as_ptr() as *const c_void, data.len());

            // W^X: Remove write, add execute.
            // Addr must be aligned to a page boundary (probably elsehwere too).
            let ret = libc::mprotect(memptr, size, PROT_EXEC | PROT_READ);
            assert_eq!(ret, 0);
        }

        Self {
            code: memptr as *mut u8,
            size,
        }
    }
}

impl Drop for JitMemory {
    fn drop(&mut self) {
        unsafe {
            let code = self.code as *mut c_void;

            // W^X: Remove execute, add write.
            // Apparently we need write permission to free the memory.
            let ret = libc::mprotect(code, self.size, PROT_READ | PROT_WRITE);
            assert_eq!(ret, 0);

            libc::free(self.code as *mut c_void)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem;

    use crate::jit::asm_encoding::*;
    use crate::jit::asm_repr::*;
    use Instr::*;
    use Reg::*;

    use super::*;

    #[test]
    fn test_fn_void() {
        let instrs = [Ret];
        let code = compile(&instrs);
        let jit = JitMemory::new(&code);
        let f: fn() = unsafe { mem::transmute(jit.code) };
        f();
        f();
    }

    #[test]
    fn test_fn_int() {
        let instrs = [MovRI(Rax, 1337), Ret];
        let code = compile(&instrs);
        let jit = JitMemory::new(&code);
        let f: fn() -> i32 = unsafe { mem::transmute(jit.code) };
        let ret = f();
        assert_eq!(ret, 1337);
        let ret = f();
        assert_eq!(ret, 1337);
    }
}
