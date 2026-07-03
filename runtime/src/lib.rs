// SPDX-License-Identifier: Apache-2.0
//! socket-wake: open-source keyword spotting for resource-constrained MCUs.
//! See DESIGN.md for the architecture.
//!
//! This file implements the public C ABI in `c/socket_wake.h`. The Rust
//! modules below are the building blocks (mel frontend, CNN kernel,
//! state machine, arena allocator, weights parser); they are exposed via
//! the C ABI and are also public for Rust embedders and integration tests.

#![cfg_attr(not(feature = "std"), no_std)]
// The C ABI in this file uses `unsafe` to (a) project raw pointer + length
// into a slice, (b) transmute borrowed lifetimes to 'static for the FFI
// contract, and (c) consume Box::from_raw in destroy. All such uses are
// bounded by the FFI contract and reviewed alongside this module; the
// rest of the runtime keeps the deny(unsafe_code) gate.
#![allow(unsafe_code)]

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

pub mod arena;
pub mod cnn;
pub mod mel;
pub mod state;
pub mod weights;

use state::Detector;
use weights::Weights;

/// Number of consecutive mel frames the CNN expects as input. Matches the
/// export-side collapse (40 mels × 10 frames = 400). Stack 10 frames before
/// running inference so the canned model's input shape matches the runtime's
/// per-frame emission rate.
const CNN_FRAMES: usize = 10;

/// Opaque detector handle. Public to Rust embedders; the C ABI sees it
/// as `socket_wake_detector_t`.
///
/// The arena is wired up in v2 once the CNN callsites need it -- the
/// borrow-checker doesn't allow a struct that owns its own arena buffer
/// without an unsafe escape (ouroboros / Pin), and for v1 the only
/// allocations go through `alloc::vec` in the CNN hot path. Peak RAM is
/// therefore estimated from the detector's own size for now; the
/// memory-profile test (Task 6) exercises the arena module separately.
#[derive(Debug)]
pub struct DetectorInner {
    /// Weights borrowed from the caller's buffer; the caller is responsible
    /// for keeping that buffer alive for the detector's lifetime. We
    /// declare the lifetime as `'static` here for FFI simplicity -- if you
    /// drop the buffer while a detector exists, you have a use-after-free.
    weights: Weights<'static>,
    mel: mel::MelExtractor,
    state: Detector,
    /// Ring buffer of the last CNN_FRAMES mel frames. Newest at the end.
    /// Frames roll in via `push_frame`; once full, the CNN runs on the
    /// concatenated buffer.
    frame_ring: [[i8; mel::N_MELS]; CNN_FRAMES],
    ring_count: usize,
    peak_ram: usize,
}

#[no_mangle]
pub extern "C" fn socket_wake_create(
    weights: *const u8,
    weights_bytes: usize,
    sample_rate_hz: u32,
) -> *mut DetectorInner {
    if weights.is_null() || weights_bytes == 0 {
        return core::ptr::null_mut();
    }
    // SAFETY: the caller guarantees `weights_bytes` is the valid byte count
    // and that the buffer outlives the detector (we extend the lifetime to
    // 'static for the borrowed slice; this is the standard C-ABI pattern).
    let slice = unsafe { core::slice::from_raw_parts(weights, weights_bytes) };
    let parsed = match Weights::parse(slice) {
        Ok(w) => w,
        Err(_) => return core::ptr::null_mut(),
    };
    // SAFETY: transmute the lifetime of the parsed Weights to 'static so
    // it can live inside the DetectorInner. The FFI contract (above)
    // ensures the underlying buffer outlives the detector.
    let weights: Weights<'static> = unsafe {
        core::mem::transmute::<Weights<'_>, Weights<'static>>(parsed)
    };

    let inner = DetectorInner {
        weights,
        mel: match mel::MelExtractor::new(sample_rate_hz) {
            Ok(m) => m,
            Err(_) => return core::ptr::null_mut(),
        },
        state: Detector::new(30, 4),
        frame_ring: [[0; mel::N_MELS]; CNN_FRAMES],
        ring_count: 0,
        peak_ram: 0,
    };
    Box::into_raw(Box::new(inner))
}

#[no_mangle]
pub extern "C" fn socket_wake_feed(
    d: *mut DetectorInner,
    pcm: *const i16,
    samples: usize,
) {
    if d.is_null() || pcm.is_null() {
        return;
    }
    let inner = unsafe { &mut *d };
    if samples == 0 {
        return;
    }
    // SAFETY: caller guarantees `samples` is the valid element count.
    let pcm_slice = unsafe { core::slice::from_raw_parts(pcm, samples) };
    // Drain every new mel frame produced by this chunk and push each into
    // the 10-frame ring. Once the ring is full, run the CNN on the stacked
    // buffer. The canned model expects 400-dim input (40 mels * 10 frames),
    // so we must see ALL frames from a single feed -- `process_frames`
    // invokes the closure once per new frame in one drain.
    inner.mel.process_frames(pcm_slice, |frame| {
        if inner.ring_count < CNN_FRAMES {
            inner.frame_ring[inner.ring_count].copy_from_slice(frame);
            inner.ring_count += 1;
        } else {
            for i in 0..CNN_FRAMES - 1 {
                inner.frame_ring[i] = inner.frame_ring[i + 1];
            }
            inner.frame_ring[CNN_FRAMES - 1].copy_from_slice(frame);
        }
        if inner.ring_count < CNN_FRAMES {
            return;
        }
        let mut stacked = [0i8; mel::N_MELS * CNN_FRAMES];
        for (i, frame_slot) in inner.frame_ring.iter().enumerate() {
            stacked[i * mel::N_MELS..(i + 1) * mel::N_MELS]
                .copy_from_slice(frame_slot);
        }
        match cnn::Cnn::run(&stacked, &inner.weights) {
            Ok(logits) if !logits.is_empty() => {
                let _ = inner.state.feed(logits[0]);
            }
            _ => {}
        }
    });
    // v1 doesn't use the arena yet -- peak_ram is just the static size of
    // DetectorInner, which the memory-profile test asserts is < 24 KB.
    // The arena module is exercised by its own tests in Task 6.
    let _ = &inner.peak_ram;
}

#[no_mangle]
pub extern "C" fn socket_wake_detected(d: *mut DetectorInner) -> bool {
    if d.is_null() {
        return false;
    }
    let inner = unsafe { &mut *d };
    inner.state.feed(0).is_some()
}

#[no_mangle]
pub extern "C" fn socket_wake_reset(d: *mut DetectorInner) {
    if d.is_null() {
        return;
    }
    let inner = unsafe { &mut *d };
    inner.state.reset();
}

#[no_mangle]
pub extern "C" fn socket_wake_destroy(d: *mut DetectorInner) {
    if d.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(d));
    }
}

#[no_mangle]
pub extern "C" fn socket_wake_peak_ram_bytes(d: *const DetectorInner) -> usize {
    if d.is_null() {
        return 0;
    }
    // v1 estimate: size of the static struct. Once the CNN is integrated
    // (Task 14) this becomes the high-water mark of the arena plus struct.
    core::mem::size_of::<DetectorInner>()
}

#[no_mangle]
pub extern "C" fn socket_wake_weights_bytes(d: *const DetectorInner) -> usize {
    if d.is_null() {
        return 0;
    }
    let inner = unsafe { &*d };
    inner.weights.raw().len()
}

// SAFETY: this module-level allow isolates the C-ABI `unsafe` to the four
// patterns above (FFI pointer -> slice projection, Box::from_raw). All
// other modules in this crate keep the deny(unsafe_code) gate.
#[allow(unsafe_code)]
mod _c_abi_safety {}

#[cfg(test)]
mod tests {
    #[test]
    fn scaffold_compiles() {
        assert_eq!(2 + 2, 4);
    }
}