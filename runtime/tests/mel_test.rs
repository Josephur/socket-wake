// SPDX-License-Identifier: Apache-2.0
use socket_wake_runtime::mel::{MelError, MelExtractor};

const SAMPLE_RATE: u32 = 16_000;
const WINDOW_SAMPLES: usize = 480;
const N_MELS: usize = 40;

#[test]
fn mel_extracts_silence_to_zero() {
    // Silence: energy = 0, log10(0 + 1e-6) ≈ -6, scaled to INT8 ≈ -95.
    // All mels must be near that floor.
    let mut m = MelExtractor::new(SAMPLE_RATE).unwrap();
    let frame = vec![0i16; WINDOW_SAMPLES];
    let out = m.process_frame(&frame);
    assert_eq!(out.len(), N_MELS);
    for &v in out {
        assert!((-127..=-80).contains(&v),
            "silence should be at the log-floor (~-95): got {}", v);
    }
}

#[test]
fn mel_extracts_dc_to_positive() {
    // DC: bin 0 dominates; should exceed the silence floor.
    let mut m = MelExtractor::new(SAMPLE_RATE).unwrap();
    let frame: Vec<i16> = vec![1000i16; WINDOW_SAMPLES];
    let out = m.process_frame(&frame);
    assert!(out[0] > -50,
        "DC must exceed silence floor at mel[0]: got {}", out[0]);
    // Higher bins should remain at or below bin 0.
    assert!(out[0] >= out[1],
        "mel[0] (DC energy) should be >= mel[1]: {} vs {}", out[0], out[1]);
}

#[test]
fn mel_rejects_wrong_sample_rate() {
    let r = MelExtractor::new(8_000);
    assert!(matches!(r, Err(MelError::UnsupportedSampleRate(8_000))));
}

#[test]
fn mel_emits_one_frame_per_window_of_pcm() {
    let mut m = MelExtractor::new(SAMPLE_RATE).unwrap();
    // Feed exactly one window: expect one frame back.
    let one_window = vec![0i16; WINDOW_SAMPLES];
    let out = m.process_frame(&one_window);
    assert_eq!(out.len(), N_MELS);
    // Second call should be partial (not yet enough for another full window).
    let partial = vec![0i16; 100];
    let out2 = m.process_frame(&partial);
    assert!(out2.is_empty());
}