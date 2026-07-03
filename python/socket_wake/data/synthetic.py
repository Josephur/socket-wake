"""Synthetic mel-feature dataset for the v1 canned "hey socket" model.

This is *not* a deployment dataset — it's a reproducible stand-in that
exercises the training/export pipeline end-to-end. Real audio for v2
comes from Speech Commands + recorded utterances + Piper synthesis.

The two classes are distinguished by *where* energy sits on the mel
axis (40 mels, ~0-2 kHz ~= mels 0-20):

    target:     random PCM whose magnitude concentrates on mels 0-20
    not-target: random PCM whose magnitude is spread across all mels

Each example is a (1, 40, 10) mel tensor — 10 stacked time frames so the
DS-CNN-L stem sees a real-shaped input.
"""

from pathlib import Path

import numpy as np


N_PER_CLASS = 200
N_MELS = 40
N_FRAMES = 10


def _random_pcm(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform [-1, 1] PCM."""
    return rng.uniform(-1.0, 1.0, size=n_samples).astype(np.float32)


def _pcm_to_mel(pcm: np.ndarray, target: bool, rng: np.random.Generator) -> np.ndarray:
    """Synthesize a (N_MELS, N_FRAMES) mel tensor from random PCM.

    The mel transform is approximated as a per-band gain envelope — we
    don't need a faithful STFT here, just a per-mel magnitude profile
    that matches the class label.
    """
    pcm = pcm[: N_MELS * N_FRAMES]
    if pcm.size < N_MELS * N_FRAMES:
        pcm = np.pad(pcm, (0, N_MELS * N_FRAMES - pcm.size))

    pcm = pcm.reshape(N_MELS, N_FRAMES)
    base = np.abs(pcm) + 1e-3
    if target:
        # Concentrate energy in mels 0-20 (0-2 kHz).
        gains = np.zeros(N_MELS, dtype=np.float32)
        gains[:20] = rng.uniform(0.6, 1.0, size=20).astype(np.float32)
        gains[20:] = rng.uniform(0.0, 0.2, size=20).astype(np.float32)
    else:
        # Spread energy across all mels.
        gains = rng.uniform(0.4, 1.0, size=N_MELS).astype(np.float32)

    mel = base * gains[:, None]
    # Light temporal jitter so consecutive frames differ.
    mel = mel + rng.normal(0.0, 0.02, size=mel.shape).astype(np.float32)
    return mel.astype(np.float32)


def synthesize(n_per_class: int = N_PER_CLASS, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for label, target in (("target", True), ("not_target", False)):
        arr = np.zeros((n_per_class, 1, N_MELS, N_FRAMES), dtype=np.float32)
        for i in range(n_per_class):
            pcm = _random_pcm(N_MELS * N_FRAMES * 4, rng)
            mel = _pcm_to_mel(pcm, target=target, rng=rng)
            arr[i, 0] = mel
        out[label] = arr
    return out


def main(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = synthesize()
    np.savez(out_path, **data)
    print(
        f"wrote {out_path}  "
        f"target={data['target'].shape}  not_target={data['not_target'].shape}"
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate synthetic mel training data.")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("models/hey-socket-v1/train_data.npz"),
    )
    args = p.parse_args()
    main(args.out)