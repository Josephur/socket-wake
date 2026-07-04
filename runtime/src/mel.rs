// SPDX-License-Identifier: Apache-2.0
//! Mel-filterbank feature extractor: 16 kHz, 30 ms window, 10 ms hop,
//! 40-mel filterbank, INT8 output.
//!
//! Mirrors `python/socket_wake/features/mel.py` -- the pipeline the model
//! was trained on -- step for step:
//!
//!   1. window = pcm/32768 * hann(480)   (hann = 0.5*(1-cos(2*pi*i/479)))
//!   2. spec   = |rfft(window, n=512)|^2  (257 bins)
//!   3. mel    = log10(spec @ fb.T + 1e-6)
//!      with fb = Slaney-style triangles, bin = floor(513 * hz / 16000)
//!   4. q      = clip(round(mel * 127 / 8), -127, 127) as i8 (ties-to-even)
//!
//! Exact float equality with numpy's pocketfft is not guaranteed, but the
//! INT8 quantization grid absorbs the tiny FFT ordering differences;
//! `tests/mel_parity_test.rs` bounds the divergence at <= 1 quantization
//! step against Python-generated reference frames.

extern crate alloc;
use alloc::vec::Vec;

/// Allowed sample rate. v1 supports 16 kHz only.
const SAMPLE_RATE_HZ: u32 = 16_000;
const WINDOW_SAMPLES: usize = 480;   // 30 ms @ 16 kHz
const HOP_SAMPLES: usize = 160;      // 10 ms @ 16 kHz
const N_FFT: usize = 512;
const N_BINS: usize = N_FFT / 2 + 1; // 257
pub const N_MELS: usize = 40;
/// Maps log-mel energy to INT8: `q = round(log_mel * 127 / SCALE)`.
const INT8_SCALE: f32 = 8.0;

#[derive(Debug, PartialEq)]
pub enum MelError {
    UnsupportedSampleRate(u32),
}

/// One triangular mel filter stored sparsely: weights for FFT bins
/// `lo..lo+weights.len()`.
#[derive(Debug, PartialEq)]
struct MelFilter {
    lo: usize,
    weights: Vec<f32>,
}

/// Slaney-style mel filterbank, matching `features/mel.py` bit-for-bit:
/// mel points linspaced in f64, hz points via 10^x, bins floored with the
/// (n_fft + 1) numerator, triangle slopes computed in f32 from the
/// integer bins.
fn mel_filterbank() -> Vec<MelFilter> {
    let sr = SAMPLE_RATE_HZ as f64;
    let mel_max = 2595.0 * libm::log10(1.0 + (sr / 2.0) / 700.0);

    let mut bin_pts = [0usize; N_MELS + 2];
    for (i, b) in bin_pts.iter_mut().enumerate() {
        let mel = mel_max * i as f64 / (N_MELS + 1) as f64;
        let hz = 700.0 * (libm::pow(10.0, mel / 2595.0) - 1.0);
        let bin = libm::floor((N_FFT as f64 + 1.0) * hz / sr) as usize;
        *b = bin.min(N_FFT / 2);
    }

    let mut fb = Vec::with_capacity(N_MELS);
    for m in 0..N_MELS {
        let (lo, mid, hi) = (bin_pts[m], bin_pts[m + 1], bin_pts[m + 2]);
        let mut weights = Vec::with_capacity(hi.saturating_sub(lo));
        for k in lo..mid {
            weights.push((k - lo) as f32 / (mid - lo) as f32);
        }
        for k in mid..hi {
            weights.push((hi - k) as f32 / (hi - mid) as f32);
        }
        fb.push(MelFilter { lo, weights });
    }
    fb
}

#[derive(Debug, PartialEq)]
pub struct MelExtractor {
    ring: [i16; WINDOW_SAMPLES],
    ring_len: usize,
    out: [i8; N_MELS],
    hann: [f32; WINDOW_SAMPLES],
    fb: Vec<MelFilter>,
}

impl MelExtractor {
    pub fn new(sample_rate_hz: u32) -> Result<Self, MelError> {
        if sample_rate_hz != SAMPLE_RATE_HZ {
            return Err(MelError::UnsupportedSampleRate(sample_rate_hz));
        }
        let mut hann = [0.0f32; WINDOW_SAMPLES];
        for (i, h) in hann.iter_mut().enumerate() {
            let t = 2.0 * core::f64::consts::PI * i as f64
                / (WINDOW_SAMPLES - 1) as f64;
            *h = (0.5 * (1.0 - libm::cos(t))) as f32;
        }
        Ok(Self {
            ring: [0i16; WINDOW_SAMPLES],
            ring_len: 0,
            out: [0i8; N_MELS],
            hann,
            fb: mel_filterbank(),
        })
    }

    /// Appends samples to an internal ring; when a full window has arrived,
    /// computes one mel frame and stores it in `out`. The returned slice is
    /// valid until the next call. Returns an empty slice if no full frame yet.
    pub fn process_frame(&mut self, pcm: &[i16]) -> &[i8] {
        for &s in pcm {
            if self.ring_len < WINDOW_SAMPLES {
                self.ring[self.ring_len] = s;
                self.ring_len += 1;
            }
            if self.ring_len == WINDOW_SAMPLES {
                self.compute_frame_into_out();
                // Slide by HOP_SAMPLES: drop the oldest HOP, leaving WINDOW-HOP.
                self.ring.copy_within(HOP_SAMPLES..WINDOW_SAMPLES, 0);
                self.ring_len = WINDOW_SAMPLES - HOP_SAMPLES;
                return &self.out;
            }
        }
        &[]
    }

    /// Drains `pcm` and invokes `on_frame` for every new mel frame produced.
    /// Returns the number of frames emitted. Use this when the caller needs
    /// every frame (e.g. to drive a downstream model that expects a stack of
    /// consecutive frames); the simpler `process_frame` only exposes the last.
    pub fn process_frames<F: FnMut(&[i8])>(&mut self, pcm: &[i16], mut on_frame: F) -> usize {
        let mut n = 0;
        for &s in pcm {
            if self.ring_len < WINDOW_SAMPLES {
                self.ring[self.ring_len] = s;
                self.ring_len += 1;
            }
            if self.ring_len == WINDOW_SAMPLES {
                self.compute_frame_into_out();
                self.ring.copy_within(HOP_SAMPLES..WINDOW_SAMPLES, 0);
                self.ring_len = WINDOW_SAMPLES - HOP_SAMPLES;
                on_frame(&self.out);
                n += 1;
            }
        }
        n
    }

    fn compute_frame_into_out(&mut self) {
        // Windowed samples, zero-padded to the FFT size.
        let mut buf = [0.0f32; N_FFT];
        for i in 0..WINDOW_SAMPLES {
            buf[i] = (self.ring[i] as f32 / 32768.0) * self.hann[i];
        }
        // Real FFT. microfft packs the (real) Nyquist bin into the
        // imaginary part of bin 0, so unpack to the numpy layout of 257
        // power bins.
        let spectrum = microfft::real::rfft_512(&mut buf);
        let mut spec = [0.0f32; N_BINS];
        spec[0] = spectrum[0].re * spectrum[0].re;
        spec[N_FFT / 2] = spectrum[0].im * spectrum[0].im;
        for k in 1..N_FFT / 2 {
            spec[k] = spectrum[k].re * spectrum[k].re
                + spectrum[k].im * spectrum[k].im;
        }

        for (m, filter) in self.fb.iter().enumerate() {
            let mut energy = 0.0f32;
            for (j, &w) in filter.weights.iter().enumerate() {
                energy += w * spec[filter.lo + j];
            }
            let log_e = libm::log10f(energy + 1e-6);
            let q = libm::rintf(log_e * 127.0 / INT8_SCALE);
            self.out[m] = q.clamp(-127.0, 127.0) as i8;
        }
    }

    pub fn window_samples(&self) -> usize { WINDOW_SAMPLES }
    pub fn hop_samples(&self) -> usize { HOP_SAMPLES }
    pub fn n_mels(&self) -> usize { N_MELS }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_is_at_log_floor() {
        // Silence: energy = 0, log10(0 + 1e-6) ~= -6, scaled to INT8 ~= -95.
        // All mels must be near that floor.
        let mut m = MelExtractor::new(SAMPLE_RATE_HZ).unwrap();
        let silent = vec![0i16; WINDOW_SAMPLES];
        let out = m.process_frame(&silent);
        for &v in out {
            assert!((-127..=-80).contains(&v),
                "silence should be at the log-floor (~-95): got {}", v);
        }
    }

    #[test]
    fn rejects_wrong_sample_rate() {
        assert_eq!(
            MelExtractor::new(8_000),
            Err(MelError::UnsupportedSampleRate(8_000))
        );
    }

    #[test]
    fn dc_energy_lands_in_bin_zero() {
        let mut m = MelExtractor::new(SAMPLE_RATE_HZ).unwrap();
        let pcm = vec![1000i16; WINDOW_SAMPLES];
        let out = m.process_frame(&pcm);
        assert!(out[0] > -50, "DC should exceed the silence floor: {}", out[0]);
    }
}
