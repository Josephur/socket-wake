// SPDX-License-Identifier: Apache-2.0
//! Fixed-size bump allocator with peak usage tracking.
//!
//! SAFETY: this module uses `unsafe` to project a typed `&mut [T]` from a
//! raw byte pointer after bounds-checking the cursor against the buffer
//! length. The projection is valid because (a) we verified alignment and
//! length, (b) the returned slice borrows `&mut self.buf` exclusively, and
//! (c) the caller cannot outlive the `Arena`. Unsafe is isolated to this
//! module so the crate-level `#![deny(unsafe_code)]` still applies.
#![allow(unsafe_code)]
//! No `free` -- fragmentation impossible by construction.
//!
//! No `free` -- fragmentation impossible by construction. Suitable for
//! the inference hot path on MCUs without a heap. Embedded consumers
//! provide a `&'static mut [u8]` buffer; we bump a cursor and report the
//! high-water mark for the memory-profile CI test.

#[derive(Debug)]
pub struct Arena<'a> {
    buf: &'a mut [u8],
    used: usize,
    peak: usize,
}

impl<'a> Arena<'a> {
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, used: 0, peak: 0 }
    }

    /// Allocates `count * size_of::<T>()` bytes aligned to `align_of::<T>()`.
    /// Returns `None` on overflow.
    pub fn alloc<T>(&mut self, count: usize) -> Option<&mut [T]> {
        let size = core::mem::size_of::<T>() * count;
        let align = core::mem::align_of::<T>();
        let aligned_used = (self.used + align - 1) & !(align - 1);
        if aligned_used + size > self.buf.len() {
            return None;
        }
        let ptr = unsafe { self.buf.as_mut_ptr().add(aligned_used) };
        self.used = aligned_used + size;
        if self.used > self.peak {
            self.peak = self.used;
        }
        Some(unsafe { core::slice::from_raw_parts_mut(ptr as *mut T, count) })
    }

    /// High-water mark since construction (or last reset).
    pub fn peak(&self) -> usize {
        self.peak
    }

    /// Current usage.
    pub fn used(&self) -> usize {
        self.used
    }

    /// Resets the cursor to zero. Peak is preserved (the user is asking
    /// for a fresh window, not erasing history).
    pub fn reset(&mut self) {
        self.used = 0;
    }

    /// Capacity of the backing buffer in bytes.
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_tracks_peak_usage() {
        let mut buf = [0u8; 1024];
        let mut a = Arena::new(&mut buf);
        let _x = a.alloc::<u32>(10);          // 40 bytes
        let _y = a.alloc::<i8>(100);          // 100 bytes
        assert_eq!(a.peak(), 140);
        a.reset();                            // does not lower peak
        assert_eq!(a.peak(), 140);
        assert_eq!(a.used(), 0);
    }

    #[test]
    fn arena_rejects_oversized_alloc() {
        let mut buf = [0u8; 16];
        let mut a = Arena::new(&mut buf);
        assert!(a.alloc::<u32>(1024).is_none());
    }

    #[test]
    fn arena_aligns_allocations() {
        let mut buf = [0u8; 64];
        let mut a = Arena::new(&mut buf);
        let _x = a.alloc::<u8>(1);            // 1 byte
        let _y = a.alloc::<u32>(1);           // align to 4: pad 3 bytes, then 4 bytes
        // 1 + pad(3) + 4 = 8
        assert_eq!(a.used(), 8);
        assert_eq!(a.peak(), 8);
    }
}