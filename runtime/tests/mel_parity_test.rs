// SPDX-License-Identifier: Apache-2.0
// Mel-frontend parity against the Python pipeline the model was trained
// on. `python -m socket_wake.gen_mel_ref` produces deterministic PCM and
// the Python INT8 mel frames; this test streams the same PCM through the
// Rust extractor.
//
// Exact float equality with numpy's FFT is not achievable, so the
// contract is: every mel value within 1 quantization step, and the
// overwhelming majority bit-identical. A systematic offset (wrong window,
// wrong filterbank bins, truncation instead of rounding) trips this
// immediately.

use socket_wake_runtime::mel::{MelExtractor, N_MELS};

const REF: &[u8] = include_bytes!("../../models/hey-socket-v1/mel_ref.bin");

#[test]
fn mel_frontend_matches_python_within_one_step() {
    assert_eq!(&REF[0..4], b"SWMR");
    let version = u16::from_le_bytes(REF[4..6].try_into().unwrap());
    assert_eq!(version, 1);
    let n_samples = u32::from_le_bytes(REF[6..10].try_into().unwrap()) as usize;
    let n_frames = u16::from_le_bytes(REF[10..12].try_into().unwrap()) as usize;
    let pcm_end = 12 + n_samples * 2;
    assert_eq!(REF.len(), pcm_end + n_frames * N_MELS);

    let pcm: Vec<i16> = REF[12..pcm_end]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let expected = &REF[pcm_end..];

    let mut extractor = MelExtractor::new(16_000).expect("extractor");
    let mut got: Vec<i8> = Vec::with_capacity(n_frames * N_MELS);
    extractor.process_frames(&pcm, |frame| got.extend_from_slice(frame));
    assert_eq!(got.len(), n_frames * N_MELS, "frame count mismatch");

    let mut exact = 0usize;
    let mut max_diff = 0i32;
    for (i, (&g, &e)) in got.iter().zip(expected.iter().map(|&b| b as i8).map(|v| v).collect::<Vec<_>>().iter()).enumerate() {
        let d = (g as i32 - e as i32).abs();
        if d == 0 {
            exact += 1;
        }
        if d > max_diff {
            max_diff = d;
        }
        assert!(
            d <= 1,
            "frame {} mel {}: rust {} vs python {} (diff {})",
            i / N_MELS, i % N_MELS, g, e, d
        );
    }
    let total = got.len();
    let pct = 100.0 * exact as f64 / total as f64;
    eprintln!("mel parity: {exact}/{total} exact ({pct:.2}%), max diff {max_diff}");
    assert!(pct > 95.0, "too many off-by-one mels: {pct:.2}% exact");
}
