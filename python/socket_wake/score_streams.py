# SPDX-License-Identifier: Apache-2.0
"""Score raw PCM stream files with Int8Sim + the eval_v3 streaming
detector -- the simulation-side counterpart of the Rust stream_check
example, for A/B-ing the runtime against the validated simulation on the
exact same audio.

Usage:
    python -m socket_wake.score_streams models/hey-socket-v1/streams/*.raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from socket_wake.data.build_v3_dataset import MODEL_DIR
from socket_wake.eval_v3 import count_fires, stream_scores
from socket_wake.model.kws_cnn import Int8Sim, KWSConvNet, fold_batchnorm

THRESHOLD = 0.95


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--ckpt", type=Path, default=MODEL_DIR / "checkpoint_v3.pt")
    p.add_argument("--data", type=Path, default=MODEL_DIR / "train_dataset_v3.pt")
    args = p.parse_args()

    state = torch.load(args.ckpt, weights_only=False)
    model = KWSConvNet(n_classes=state.get("n_classes", 2))
    model.load_state_dict(state["model"])
    folded = fold_batchnorm(model.eval())
    payload = torch.load(args.data, weights_only=False)
    x_all, split = payload["x"], payload["split"].bool()
    calib = x_all[split][:512].float() / 127.0
    sim = Int8Sim(folded, calib)

    total = 0
    for f in args.files:
        pcm = np.frombuffer(f.read_bytes(), dtype="<i2")
        probs = stream_scores(pcm, sim.forward)
        fires = count_fires(probs, THRESHOLD)
        total += fires
        peak = probs.max() if probs.size else 0.0
        print(f"{f.name:<40} {pcm.size / 16_000:>7.1f}s  fires={fires}  "
              f"max_p={peak:.3f}")
    print(f"total fires: {total}")


if __name__ == "__main__":
    main()
