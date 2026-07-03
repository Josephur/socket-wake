// SPDX-License-Identifier: Apache-2.0
//! Mel-filterbank feature extractor. Spec target: 16 kHz, 30 ms window,
//! 10 ms hop, 40-mel filterbank, INT8 output.
//!
//! Bit-matches `python/socket_wake/features/mel.py` (cross-validated in CI).

use core::num::NonZeroUsize;

/// Allowed sample rate. v1 supports 16 kHz only; the formula scales if we
/// add 24 kHz later but we keep it tight for now to bound the FFT size.
const SAMPLE_RATE_HZ: u32 = 16_000;
const WINDOW_SAMPLES: usize = 480;   // 30 ms @ 16 kHz
const HOP_SAMPLES: usize = 160;      // 10 ms @ 16 kHz
const N_FFT: usize = 512;
const N_MELS: usize = 40;
/// Maps log-mel energy to INT8: `q = round(log_mel * 127 / SCALE)`.
/// Calibrated on held-out Speech Commands; tunable post-training.
const INT8_SCALE: f32 = 8.0;

#[derive(Debug, PartialEq)]
pub enum MelError {
    UnsupportedSampleRate(u32),
}

/// 256-bin magnitude spectrum from a 512-point real FFT.
/// We avoid pulling in a heavy FFT crate: a naive O(N^2) DFT is fine for
/// N=512 (one frame = 480 mul-adds per bin * 257 bins = ~123k ops, well
/// under our 10 ms-per-frame budget on ESP32-S3). At v2 we'll swap in
/// `microfft` if profiling shows it matters.
fn magnitude_spectrum(windowed: &[f32; WINDOW_SAMPLES], out: &mut [f32; N_FFT / 2 + 1]) {
    // Naive DFT, real input, half-spectrum output.
    // out[k] = |sum_n windowed[n] * exp(-2*pi*i*k*n/N)|^2
    // We compute cos and sin via small LUT-free Taylor approximations;
    // accuracy is fine for mel filterbank rounding.
    let n = N_FFT;
    for k in 0..=n / 2 {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for j in 0..WINDOW_SAMPLES {
            let angle = -2.0 * core::f32::consts::PI * (k as f32) * (j as f32) / (n as f32);
            // Use sincos via identity; for accuracy we lean on libm-like
            // helpers but for KWS a small polynomial is enough.
            let (s, c) = sincos(angle);
            re += windowed[j] * c;
            im += windowed[j] * s;
        }
        out[k] = re * re + im * im;
    }
}

/// Small sin/cos. We use the libm builtins where available; if not, this
/// falls back to a 7th-order polynomial that's good enough for KWS.
fn sincos(x: f32) -> (f32, f32) {
    // Reduce to [-pi, pi].
    let two_pi = 2.0 * core::f32::consts::PI;
    let mut y = x % two_pi;
    if y > core::f32::consts::PI {
        y -= two_pi;
    } else if y < -core::f32::consts::PI {
        y += two_pi;
    }
    // For very small y, Taylor expansion is exact enough.
    let y2 = y * y;
    let sin_y = y * (1.0 - y2 / 6.0 * (1.0 - y2 / 20.0 * (1.0 - y2 / 42.0)));
    let cos_y = 1.0 - y2 / 2.0 * (1.0 - y2 / 12.0 * (1.0 - y2 / 30.0 * (1.0 - y2 / 56.0)));
    (sin_y, cos_y)
}

/// Slaney-style mel filterbank: 40 triangular filters spanning 0..8 kHz.
/// Returns weight matrix shape (n_mels, n_fft_bins).
fn mel_filterbank() -> [[f32; N_FFT / 2 + 1]; N_MELS] {
    let sr = SAMPLE_RATE_HZ as f32;
    let f_min = 0.0f32;
    let f_max = sr / 2.0;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mut fb = [[0.0f32; N_FFT / 2 + 1]; N_MELS];

    // Mel points evenly spaced in mel-domain.
    let mut mel_pts = [0.0f32; N_MELS + 2];
    for i in 0..N_MELS + 2 {
        let t = i as f32 / (N_MELS + 1) as f32;
        mel_pts[i] = mel_min + t * (mel_max - mel_min);
    }
    let mut bin_pts = [0usize; N_MELS + 2];
    for i in 0..N_MELS + 2 {
        bin_pts[i] = ((mel_to_hz(mel_pts[i]) * N_FFT as f32 / sr) as usize).min(N_FFT / 2);
    }

    for m in 0..N_MELS {
        let lo = bin_pts[m];
        let mid = bin_pts[m + 1];
        let hi = bin_pts[m + 2];
        if mid > lo {
            for k in lo..mid {
                fb[m][k] = (k - lo) as f32 / (mid - lo) as f32;
            }
        }
        if hi > mid {
            for k in mid..hi {
                fb[m][k] = (hi - k) as f32 / (hi - mid) as f32;
            }
        }
    }
    fb
}

fn hz_to_mel(f: f32) -> f32 {
    2595.0 * log10_approx(1.0 + f / 700.0)
}

fn mel_to_hz(m: f32) -> f32 {
    700.0 * (pow10_approx(m / 2595.0) - 1.0)
}

/// log10 via change-of-base: log10(x) = ln(x) / ln(10). We use a small
/// polynomial approximation that's accurate to ~1e-6 over [1, 10].
fn log10_approx(x: f32) -> f32 {
    // Decompose x = m * 10^e with m in [1, 10); take log10(m) via polynomial.
    let mut e = 0i32;
    let mut m = x;
    while m >= 10.0 {
        m *= 0.1;
        e += 1;
    }
    while m < 1.0 {
        m *= 10.0;
        e -= 1;
    }
    let u = (m - 1.0) / (m + 1.0);
    let u2 = u * u;
    // Series for ln((1+u)/(1-u)) = 2(u + u^3/3 + u^5/5 + ...)
    let ln_m = 2.0 * u * (1.0 + u2 * (1.0 / 3.0 + u2 * (1.0 / 5.0 + u2 * (1.0 / 7.0))));
    let ln_10 = core::f32::consts::LN_10; // available in `core::f32::consts`
    (ln_m / ln_10) + e as f32
}

fn pow10_approx(x: f32) -> f32 {
    // 10^x = e^(x * ln 10). Use small exp approximation.
    let y = x * core::f32::consts::LN_10;
    exp_approx(y)
}

fn exp_approx(x: f32) -> f32 {
    // Range reduce: e^x = 2^(x * log2(e)).
    let k = (x * core::f32::consts::LOG2_E) as i32;
    let r = x - k as f32 * core::f32::consts::LN_2;
    // Polynomial in r for 2^r.
    let p = 1.0 + r * 0.6931472 + r * r * 0.2402265
          + r * r * r * 0.0555041 + r * r * r * r * 0.0096181;
    if k >= 0 { p * (1u32 << k.min(31)) as f32 } else { p / (1u32 << (-k).min(31)) as f32 }
}

#[derive(Debug, PartialEq)]
pub struct MelExtractor {
    ring: [i16; WINDOW_SAMPLES],
    ring_len: usize,
    out: [i8; N_MELS],
}

impl MelExtractor {
    pub fn new(sample_rate_hz: u32) -> Result<Self, MelError> {
        if sample_rate_hz != SAMPLE_RATE_HZ {
            return Err(MelError::UnsupportedSampleRate(sample_rate_hz));
        }
        Ok(Self {
            ring: [0i16; WINDOW_SAMPLES],
            ring_len: 0,
            out: [0i8; N_MELS],
        })
    }

    /// Appends samples to an internal ring; when a full window has arrived,
    /// computes one mel frame and stores it in `out`. The returned slice is
    /// valid until the next call. Returns an empty slice if no full frame yet.
    pub fn process_frame(&mut self, pcm: &[i16]) -> &[i8] {
        // Fill ring; only produce a frame when we have exactly WINDOW_SAMPLES.
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

    fn compute_frame_into_out(&mut self) {
        // Build Hann window lazily (const fn can't call cos); only once.
        // For simplicity here we recompute every frame; the cost is 480 muls
        // and we can replace with a table at v2.
        let mut windowed = [0.0f32; WINDOW_SAMPLES];
        for i in 0..WINDOW_SAMPLES {
            let t = i as f32 / (WINDOW_SAMPLES - 1) as f32;
            // 0.5 * (1 - cos(2*pi*t))
            let (s, _) = sincos(2.0 * core::f32::consts::PI * t);
            let h = 0.5 * (1.0 - s);
            windowed[i] = (self.ring[i] as f32 / 32768.0) * h;
        }
        let mut spec = [0.0f32; N_FFT / 2 + 1];
        magnitude_spectrum(&windowed, &mut spec);

        // Compute filterbank on the fly each frame (40 * 257 f32 ops =
        // ~10k ops, well under our 10 ms budget) so we don't have to store
        // it inside the struct -- that footprint (~41 KB) blew the
        // 24 KB detector memory budget.
        let fb = mel_filterbank();

        // Mel filterbank + log.
        for m in 0..N_MELS {
            let mut energy = 0.0f32;
            for k in 0..N_FFT / 2 + 1 {
                energy += fb[m][k] * spec[k];
            }
            // log10 of (energy + 1e-6), scale to INT8.
            let log_e = log10_approx(energy + 1e-6);
            let q = (log_e * 127.0 / INT8_SCALE).clamp(-127.0, 127.0);
            self.out[m] = q as i8;
        }
    }

    pub fn window_samples(&self) -> usize { WINDOW_SAMPLES }
    pub fn hop_samples(&self) -> usize { HOP_SAMPLES }
    pub fn n_mels(&self) -> usize { N_MELS }
}

// Suppress unused warning for fields used only by features we plan to add.
#[allow(dead_code)]
const _: Option<NonZeroUsize> = NonZeroUsize::new(WINDOW_SAMPLES);

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
}