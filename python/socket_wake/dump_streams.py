# SPDX-License-Identifier: Apache-2.0
"""Dump held-out evaluation streams as raw 16 kHz s16le PCM files.

These feed the Rust-side streaming check (``cargo run --example
stream_check``), which drives the real runtime (mel frontend -> INT8 CNN
-> state machine) over the same audio eval_v3.py scored in simulation:

  - ``noise.raw``     held-out MUSAN noise (should produce ~0 fires)
  - ``speech.raw``    held-out hard-negative utterances (adversarial)
  - ``pos_NN.raw``    held-out positives embedded in noise at 20 dB SNR
                      (each should produce exactly 1 fire)

Usage:
    python -m socket_wake.dump_streams [--out DIR] [--max-noise-s 120]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from socket_wake.data.build_v3_dataset import MODEL_DIR, place_in_window, trim_active
from socket_wake.data.build_v2_dataset import _to_int16
from socket_wake.data.snr_mix import mix_at_snr
from socket_wake.eval_v3 import RNG_SEED, load_test_clips

SNR_DB = 20.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=MODEL_DIR / "train_dataset_v3.pt")
    p.add_argument("--out", type=Path, default=MODEL_DIR / "streams")
    p.add_argument("--max-noise-s", type=float, default=120.0)
    p.add_argument("--n-pos", type=int, default=8)
    args = p.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    payload = torch.load(args.data, weights_only=False)
    test = load_test_clips(payload)
    args.out.mkdir(parents=True, exist_ok=True)

    noise = np.concatenate(test["noise"])[: int(args.max_noise_s * 16_000)]
    (args.out / "noise.raw").write_bytes(noise.astype("<i2").tobytes())

    gap = np.zeros(8_000, dtype=np.int16)
    speech_parts: list[np.ndarray] = []
    for c in test["neg_tts"]:
        speech_parts.extend((c, gap))
    speech = np.concatenate(speech_parts) if speech_parts else gap
    (args.out / "speech.raw").write_bytes(speech.astype("<i2").tobytes())

    for i, clip in enumerate(test["pos_tts"][: args.n_pos]):
        active = trim_active(clip)
        placed = place_in_window(active, rng)
        bg_src = test["noise"][int(rng.integers(0, len(test["noise"])))]
        bg = np.tile(bg_src, int(np.ceil(48_000 / bg_src.size)))[:48_000]
        trial = bg.astype(np.float32).copy()
        trial[16_000:32_000] = mix_at_snr(placed, bg[16_000:32_000], SNR_DB, rng)
        (args.out / f"pos_{i:02d}.raw").write_bytes(
            _to_int16(trial).astype("<i2").tobytes())

    print(f"[dump_streams] wrote noise ({noise.size / 16_000:.0f}s), "
          f"speech ({speech.size / 16_000:.0f}s), "
          f"{min(args.n_pos, len(test['pos_tts']))} positive trials -> {args.out}")


if __name__ == "__main__":
    main()
