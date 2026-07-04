// SPDX-License-Identifier: Apache-2.0
// Python-generates / Rust-verifies parity test.
//
// `python -m socket_wake.export` quantizes the trained v3 checkpoint,
// runs the Python integer reference (socket_wake/int8_ref.py) over
// held-out mel windows, and writes both the SWWT v2 blob and a vector
// file of (raw mel window, expected logits) pairs. This test runs the
// SAME windows through the Rust kernels and demands bit-identical
// logits -- any divergence in conv indexing, requantization rounding, or
// layout is a hard failure, not a tolerance miss.
//
// Vector file format (SWTV v1):
//   magic  [u8; 4] = b"SWTV"
//   version u16 LE = 1
//   n       u16 LE
//   n records of: input i8[40 * 98] (HWC row-major), logits i8[2]

use socket_wake_runtime::cnn::Cnn;
use socket_wake_runtime::weights::Weights;

const WEIGHTS: &[u8] = include_bytes!("../../models/hey-socket-v1/weights_v3.bin");
const VECTORS: &[u8] = include_bytes!("../../models/hey-socket-v1/testvectors_v3.bin");

const N_MELS: usize = 40;
const N_FRAMES: usize = 98;
const WINDOW: usize = N_MELS * N_FRAMES;

/// Input requantization, the same formula `lib.rs::push_frame` applies to
/// each mel frame before it enters the ring.
fn quantize_input(raw: &[i8], scale: f32) -> Vec<i8> {
    raw.iter()
        .map(|&v| {
            let q = (v as f32 * scale).round_ties_even();
            if q > 127.0 { 127 } else if q < -127.0 { -127 } else { q as i8 }
        })
        .collect()
}

#[test]
fn rust_kernels_match_python_reference_bit_for_bit() {
    let weights = Weights::parse(WEIGHTS).expect("exported blob must parse");
    assert_eq!(weights.n_layers(), 6, "v3 model is 6 layers");
    let params = weights.params();

    assert_eq!(&VECTORS[0..4], b"SWTV");
    let version = u16::from_le_bytes(VECTORS[4..6].try_into().unwrap());
    assert_eq!(version, 1);
    let n = u16::from_le_bytes(VECTORS[6..8].try_into().unwrap()) as usize;
    assert!(n > 0, "vector file must not be empty");

    let rec = WINDOW + 2;
    assert_eq!(VECTORS.len(), 8 + n * rec, "vector file length mismatch");

    let mut mismatches = 0;
    for i in 0..n {
        let base = 8 + i * rec;
        let raw: Vec<i8> = VECTORS[base..base + WINDOW]
            .iter()
            .map(|&b| b as i8)
            .collect();
        let expected = [
            VECTORS[base + WINDOW] as i8,
            VECTORS[base + WINDOW + 1] as i8,
        ];
        let input = quantize_input(&raw, params.input_scale);
        let logits = Cnn::run(&input, &weights).expect("forward pass");
        assert_eq!(logits.len(), 2);
        if logits[..2] != expected {
            eprintln!(
                "vector {i}: rust logits {:?} != python {:?}",
                &logits[..2], expected
            );
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "{mismatches}/{n} vectors diverged");
    eprintln!("parity: {n}/{n} vectors bit-identical");
}

#[test]
fn exported_header_carries_detector_params() {
    let weights = Weights::parse(WEIGHTS).expect("parse");
    let p = weights.params();
    assert!(p.logit_thr > 0, "threshold must be positive: {}", p.logit_thr);
    assert_eq!(p.hold, 2, "v3 benchmark used 2 consecutive hits");
    assert_eq!(p.refractory, 32, "v3 benchmark used 1 s refractory at 30 ms");
    assert!(p.input_scale > 0.0);
}
