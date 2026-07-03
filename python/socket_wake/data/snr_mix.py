# SPDX-License-Identifier: Apache-2.0
"""SNR-mixing augmentation: blend a wake-word signal with real noise clips.

Unlike ``augment.py`` (synthetic Gaussian noise), this module mixes in real
recorded noise (MUSAN or similar) at a controlled signal-to-noise ratio, so
the trained model learns what "wake word + fan noise" or "wake word +
traffic" actually looks like -- not just "wake word + white noise."
"""

from __future__ import annotations

import numpy as np


def mix_at_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Mix ``signal`` (wake-word audio) with ``noise`` at ``snr_db`` dB SNR.

    Both inputs may be int16 or float; the result is always float32, same
    length as ``signal``. If ``noise`` is shorter than ``signal`` it is
    looped (tiled) to cover the length; if longer, a random contiguous crop
    is taken. Noise power is scaled so that the ratio of signal power to
    noise power equals ``10 ** (snr_db / 10)``.
    """
    rng = rng or np.random.default_rng()

    sig = signal.astype(np.float32)
    noi = noise.astype(np.float32)

    n = sig.shape[0]
    if noi.shape[0] == 0:
        # Degenerate: no noise available, return the signal unchanged.
        return sig.copy()

    if noi.shape[0] < n:
        reps = int(np.ceil(n / noi.shape[0]))
        noi = np.tile(noi, reps)
    if noi.shape[0] > n:
        max_start = noi.shape[0] - n
        start = int(rng.integers(0, max_start + 1))
        noi = noi[start : start + n]
    else:
        noi = noi[:n]

    signal_power = float(np.mean(sig ** 2)) + 1e-9
    noise_power = float(np.mean(noi ** 2)) + 1e-9

    # Scale noise so that signal_power / (scale^2 * noise_power) == 10^(snr/10)
    target_ratio = 10.0 ** (snr_db / 10.0)
    scale = np.sqrt(signal_power / (target_ratio * noise_power))

    mixed = sig + noi * scale
    return mixed.astype(np.float32)
