# SPDX-License-Identifier: Apache-2.0
"""Audio augmentations for keyword-spotting training.

Plain numpy only -- no external audio deps. The augmenter applies a
random chain (Gaussian noise, gain jitter) on a per-call basis; for
v2 we add SpecAugment in the frequency domain.
"""

import numpy as np


def augment(wav: np.ndarray, sample_rate: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply light augmentations to a mono float32 waveform.

    Returns the same shape array. Each call draws fresh RNG so the same
    input produces different output across epochs.
    """
    rng = rng or np.random.default_rng()
    out = wav.astype(np.float32, copy=True)

    # Gaussian noise at a random SNR.
    if rng.random() < 0.8:
        snr_db = float(rng.uniform(5.0, 30.0))
        signal_power = float(np.mean(out ** 2)) + 1e-9
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        out = out + rng.normal(0.0, np.sqrt(noise_power), out.shape).astype(np.float32)

    # Random gain.
    out = out * float(rng.uniform(0.6, 1.4))
    return out