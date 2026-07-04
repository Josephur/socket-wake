# SPDX-License-Identifier: Apache-2.0
"""Train KWSConvNet on the v3 (1 s window, clip-split) dataset.

Usage:
    python -m socket_wake.train_v3 [--epochs 20] [--data ...] [--out ...]

Reports held-out window metrics at several thresholds (NOT just argmax --
the deployment operating point is a tuned threshold plus the streaming
detector's consecutive-hit requirement; see eval_v3.py for the streaming
FAR/FRR benchmark that is the real acceptance gate).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from socket_wake.model.kws_cnn import KWSConvNet, count_params

MODEL_DIR = Path("models/hey-socket-v1")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=MODEL_DIR / "train_dataset_v3.pt")
    p.add_argument("--out", type=Path, default=MODEL_DIR / "checkpoint_v3.pt")
    p.add_argument("--epochs", type=int, default=20)
    args = p.parse_args()

    payload = torch.load(args.data, weights_only=False)
    x = payload["x"].float() / 127.0
    y = payload["y"].long()
    split = payload["split"].bool()
    x_train, y_train = x[split], y[split]
    x_test, y_test = x[~split], y[~split]
    print(f"train={len(y_train)} test={len(y_test)} "
          f"(pos_train={int(y_train.sum())}, pos_test={int(y_test.sum())})")

    torch.manual_seed(0)
    model = KWSConvNet(n_classes=2)
    print(f"params={count_params(model)}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    counts = torch.bincount(y_train, minlength=2).float()
    weights = counts.sum() / (counts.clamp_min(1) * 2)
    print(f"class_counts={counts.tolist()} class_weights={weights.tolist()}")
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64,
                        shuffle=True)
    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
            seen += xb.size(0)
        print(f"epoch {epoch} loss={running / seen:.4f}")

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x_test), dim=1)[:, 1].numpy()
    yt = y_test.numpy()

    print("\nheld-out window metrics by threshold:")
    print(f"{'thr':>6} {'recall':>7} {'precision':>9} {'FA-rate':>8}")
    for thr in (0.5, 0.9, 0.95, 0.99, 0.995, 0.999):
        pred = probs >= thr
        tp = int((pred & (yt == 1)).sum())
        fp = int((pred & (yt == 0)).sum())
        fn = int((~pred & (yt == 1)).sum())
        rec = tp / max(1, tp + fn)
        prec = tp / max(1, tp + fp)
        far = fp / max(1, int((yt == 0).sum()))
        print(f"{thr:>6} {rec:>7.3f} {prec:>9.3f} {far:>8.4f}")

    torch.save({"model": model.state_dict(), "n_classes": 2,
                "arch": "KWSConvNet", "window_frames": 98}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
