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
use alloc::vec::Vec;

pub mod arena;
pub mod cnn;
pub mod mel;
pub mod state;
pub mod weights;

use state::Detector;
use weights::Weights;

/// Mel frames per CNN input window: 1.0 s of audio at the 10 ms hop.
/// Matches WINDOW_FRAMES in the v3 training pipeline (input = 40x98 HWC).
pub const CNN_FRAMES: usize = 98;

/// Run one CNN inference every this many new mel frames (30 ms), matching
/// CADENCE_FRAMES in eval_v3.py -- the cadence the FAR/FRR numbers were
/// measured at.
pub const CADENCE_FRAMES: usize = 3;

/// Opaque detector handle. Public to Rust embedders; the C ABI sees it
/// as `socket_wake_detector_t`.
#[derive(Debug)]
pub struct DetectorInner {
    /// Weights borrowed from the caller's buffer; the caller is responsible
    /// for keeping that buffer alive for the detector's lifetime. We
    /// declare the lifetime as `'static` here for FFI simplicity -- if you
    /// drop the buffer while a detector exists, you have a use-after-free.
    weights: Weights<'static>,
    mel: mel::MelExtractor,
    state: Detector,
    /// Circular buffer of the last CNN_FRAMES mel frames, already
    /// requantized to the model's input scale. `head` is the slot the NEXT
    /// frame will be written to; once `count` reaches CNN_FRAMES the window
    /// is the ring read in time order starting at `head`.
    frame_ring: [[i8; mel::N_MELS]; CNN_FRAMES],
    head: usize,
    count: usize,
    /// New frames since the last inference; run the CNN when this reaches
    /// CADENCE_FRAMES.
    since_infer: usize,
    /// Set when the state machine fires; cleared by socket_wake_detected.
    pending: bool,
}

impl DetectorInner {
    /// Feed one raw mel frame (i8, length N_MELS). Applies the model's
    /// input requantization, updates the ring, and runs the CNN at the
    /// inference cadence once a full 1 s window is buffered.
    fn push_frame(&mut self, frame: &[i8; mel::N_MELS]) {
        let scale = self.weights.params().input_scale;
        let slot = &mut self.frame_ring[self.head];
        for (dst, &v) in slot.iter_mut().zip(frame.iter()) {
            let q = (v as f32 * scale).round_ties_even();
            *dst = if q > 127.0 { 127 } else if q < -127.0 { -127 } else { q as i8 };
        }
        self.head = (self.head + 1) % CNN_FRAMES;
        if self.count < CNN_FRAMES {
            self.count += 1;
        }
        if self.count < CNN_FRAMES {
            return;
        }
        self.since_infer += 1;
        if self.since_infer < CADENCE_FRAMES {
            return;
        }
        self.since_infer = 0;

        // Assemble the HWC input: h = mel bin (40), w = time (98), c = 1.
        // Time index 0 is the OLDEST frame, which lives at `head` (the
        // slot about to be overwritten next).
        let mut input = [0i8; mel::N_MELS * CNN_FRAMES];
        for x in 0..CNN_FRAMES {
            let f = &self.frame_ring[(self.head + x) % CNN_FRAMES];
            for y in 0..mel::N_MELS {
                input[y * CNN_FRAMES + x] = f[y];
            }
        }
        if let Ok(logits) = cnn::Cnn::run(&input, &self.weights) {
            if logits.len() == 2 {
                // Quantized logit margin, saturated to i8: target minus
                // not-target. p(target) >= theta iff this clears the
                // export-computed threshold in the weights header.
                let margin = (logits[1] as i16 - logits[0] as i16)
                    .clamp(i8::MIN as i16, i8::MAX as i16) as i8;
                if self.state.feed(margin).is_some() {
                    self.pending = true;
                }
            }
        }
    }
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
    let params = weights.params();

    let inner = DetectorInner {
        weights,
        mel: match mel::MelExtractor::new(sample_rate_hz) {
            Ok(m) => m,
            Err(_) => return core::ptr::null_mut(),
        },
        state: Detector::new(params.logit_thr, params.hold, params.refractory),
        frame_ring: [[0; mel::N_MELS]; CNN_FRAMES],
        head: 0,
        count: 0,
        since_infer: 0,
        pending: false,
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
    // The borrow checker won't let the closure capture `inner` while
    // `inner.mel` is mutably borrowed, so buffer the frames first (a feed
    // chunk yields at most a handful) and push them after the drain.
    let mut frames: Vec<[i8; mel::N_MELS]> = Vec::new();
    inner.mel.process_frames(pcm_slice, |frame| {
        let mut f = [0i8; mel::N_MELS];
        f.copy_from_slice(frame);
        frames.push(f);
    });
    for f in &frames {
        inner.push_frame(f);
    }
}

#[no_mangle]
pub extern "C" fn socket_wake_detected(d: *mut DetectorInner) -> bool {
    if d.is_null() {
        return false;
    }
    let inner = unsafe { &mut *d };
    let fired = inner.pending;
    inner.pending = false;
    fired
}

#[no_mangle]
pub extern "C" fn socket_wake_reset(d: *mut DetectorInner) {
    if d.is_null() {
        return;
    }
    let inner = unsafe { &mut *d };
    inner.state.reset();
    inner.pending = false;
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
    // Static struct size plus the CNN's transient activation high-water
    // mark (first layer output dominates: 20*49*16 for the v3 model; we
    // report struct size only -- the memory-profile test bounds the rest).
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

#[cfg(test)]
mod tests {
    #[test]
    fn scaffold_compiles() {
        assert_eq!(2 + 2, 4);
    }
}
