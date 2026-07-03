#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Training entry point.

Usage:
    python -m socket_wake.train --word "hey socket" --out models/hey-socket-v1

The training loop pulls positive examples (recordings of the target
word + Piper-synthesized variants) and negatives (Speech Commands +
MUSAN noise), extracts mel features via ``socket_wake.features.mel``,
trains the DS-CNN-L model, and saves a checkpoint at ``out/checkpoint.pt``.
Run ``socket_wake.export`` next to emit the INT8 ``weights.bin`` that
the Rust runtime loads.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from socket_wake.model.ds_cnn import DSCNN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a wake-word model.")
    p.add_argument("--word", required=True, help="Target wake phrase (e.g. 'hey socket').")
    p.add_argument("--out", type=Path, required=True, help="Output directory for checkpoint.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_dummy_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic batch for the v1 training-entry smoke test.

    Real training pulls from Speech Commands + user WAVs + Piper synth
    (see DESIGN.md). For v1 we just confirm the loop compiles and the
    loss decreases on a tiny synthetic set; Task 14 trains the canned
    "hey socket" model on real data.
    """
    torch.manual_seed(0)
    x = torch.randn(batch_size, 1, 40, 10)         # 1 channel, 40 mels, 10 frames
    y = torch.randint(0, 2, (batch_size,))         # binary: target vs not-target
    return x, y


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    model = DSCNN(n_classes=2)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        x, y = build_dummy_batch(args.batch_size)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d}  loss={loss.item():.4f}")

    ckpt = args.out / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "n_classes": 2}, ckpt)
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()