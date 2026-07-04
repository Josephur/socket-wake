# SPDX-License-Identifier: Apache-2.0
"""Generate mel-frontend reference vectors for the Rust parity test.

Writes ``models/hey-socket-v1/mel_ref.bin``: deterministic synthetic PCM
(tones + seeded noise + a speech-like chirp) plus the INT8 mel frames the
Python pipeline (the one the model was trained on) produces for it.
``runtime/tests/mel_parity_test.rs`` runs the same PCM through the Rust
extractor and bounds the divergence.

Format:
    magic   [u8; 4] = b"SWMR"
    version u16 LE  = 1
    n_samples u32 LE
    n_frames  u16 LE
    pcm     [i16 LE; n_samples]
    mel     [i8; n_frames * 40]

Usage:
    python -m socket_wake.gen_mel_ref
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from socket_wake.data.build_v3_dataset import MODEL_DIR
from socket_wake.features.mel import HOP, N_MELS, WINDOW, mel_spectrogram

SR = 16_000


def make_pcm() -> np.ndarray:
    rng = np.random.default_rng(20260704)
    t = np.arange(2 * SR) / SR
    sig = (
        3000.0 * np.sin(2 * np.pi * 440.0 * t)
        + 2000.0 * np.sin(2 * np.pi * 1200.0 * t)
        + 1000.0 * np.sin(2 * np.pi * (300.0 + 2000.0 * t) * t)  # chirp
        + rng.normal(0.0, 800.0, t.size)
    )
    # Amplitude modulation so frames span quiet and loud regimes.
    sig *= 0.2 + 0.8 * (0.5 * (1 + np.sin(2 * np.pi * 1.3 * t)))
    return np.clip(sig, -32767, 32767).astype(np.int16)


def main() -> None:
    pcm = make_pcm()
    mel = mel_spectrogram(pcm)
    # The streaming Rust extractor emits one frame per hop once the window
    # is full; Python's framing is identical for exact multiples.
    n_frames = (pcm.size - WINDOW) // HOP + 1
    mel = mel[:n_frames]

    out = bytearray()
    out.extend(b"SWMR")
    out.extend(struct.pack("<HIH", 1, pcm.size, n_frames))
    out.extend(pcm.astype("<i2").tobytes())
    out.extend(mel.astype(np.int8).tobytes())
    path = Path(MODEL_DIR) / "mel_ref.bin"
    path.write_bytes(bytes(out))
    print(f"[gen_mel_ref] wrote {path}: {pcm.size} samples, "
          f"{n_frames} frames x {N_MELS} mels")


if __name__ == "__main__":
    main()
