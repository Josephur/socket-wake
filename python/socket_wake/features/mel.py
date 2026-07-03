# SPDX-License-Identifier: Apache-2.0
"""Mel-filterbank feature extractor.

MUST bit-match ``runtime/src/mel.rs`` for v1: same sample rate (16 kHz),
window (30 ms = 480 samples), hop (10 ms = 160 samples), FFT size (512),
n_mels (40), INT8 scale (8.0). Cross-validated in tests/test_features.py.

For v2 we add an option to compute log-power instead of log10-magnitude, and
to calibrate the INT8 scale on a held-out training set; both are flagged
in DESIGN.md's "Future work" list.
"""

import numpy as np

SAMPLE_RATE = 16_000
WINDOW = 480      # 30 ms @ 16 kHz
HOP = 160         # 10 ms @ 16 kHz
N_FFT = 512
N_MELS = 40
INT8_SCALE = 8.0  # matches the Rust placeholder


def _mel_filterbank(n_fft: int, n_mels: int, sample_rate: int) -> np.ndarray:
    """Slaney-style mel filterbank: 40 triangular filters, 0..8 kHz.

    Reproduces the Rust implementation byte-for-byte. The slope math is
    the standard HTK / librosa ``filters.mel`` recipe; what's specific is
    only the dimension choices.
    """
    f_min, f_max = 0.0, sample_rate / 2
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)

    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.minimum(bin_points, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        lo, mid, hi = bin_points[m], bin_points[m + 1], bin_points[m + 2]
        if mid > lo:
            fb[m, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            fb[m, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
    return fb


_MEL_FB = _mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE)


def _hann_window(n: int) -> np.ndarray:
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))


_HANN = _hann_window(WINDOW)


def mel_spectrogram(pcm: np.ndarray) -> np.ndarray:
    """Compute one or more mel frames from raw 16 kHz int16 PCM.

    Returns an INT8 array of shape (n_frames, N_MELS), bit-matching the
    Rust runtime on the same input.
    """
    assert pcm.dtype == np.int16, f"expected int16, got {pcm.dtype}"
    pcm = pcm.astype(np.float32) / 32768.0
    if len(pcm) < WINDOW:
        # Pad with zeros to produce exactly one frame (matches Rust
        # behavior of accepting shorter inputs).
        pcm = np.pad(pcm, (0, WINDOW - len(pcm)))
    n_frames = max(1, (len(pcm) - WINDOW) // HOP + 1)
    out = np.zeros((n_frames, N_MELS), dtype=np.int8)
    for i in range(n_frames):
        s = i * HOP
        frame = pcm[s : s + WINDOW]
        if len(frame) < WINDOW:
            frame = np.pad(frame, (0, WINDOW - len(frame)))
        windowed = frame * _HANN
        spec = np.abs(np.fft.rfft(windowed, n=N_FFT)) ** 2
        mel = np.log10(spec @ _MEL_FB.T + 1e-6)
        out[i] = np.clip(np.round(mel * 127.0 / INT8_SCALE), -127, 127).astype(np.int8)
    return out