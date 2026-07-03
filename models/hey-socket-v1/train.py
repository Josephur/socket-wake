"""Train KWSClassifier wake-word on the v2 noise-augmented dataset.

Reads train_dataset_v2.pt (x (N,1,40,10) int8, y (N,) int64, split (N,)
bool -- see docs/training.md "Reproducing the v2 (noise-augmented)
dataset" for exactly how this file is built), trains
KWSClassifier(n_classes=2) -- a 400->128->2 MLP that matches the
runtime's flat (40 mels * 10 frames) stacked buffer input -- with Adam +
class-weighted CrossEntropyLoss on CPU, and writes checkpoint.pt for
socket_wake.export to consume.

Class weighting: the v2 dataset is ~4.8:1 negative:positive (a real
improvement over an earlier, buggy 86:1 imbalance -- see
docs/training.md for that story), but still skewed enough that
unweighted cross-entropy would bias the model toward predicting
not-target. Weights are set inversely proportional to class frequency
in the training split.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from socket_wake.model.ds_cnn import KWSClassifier


HERE = Path(__file__).resolve().parent


def main() -> None:
    payload = torch.load(HERE / "train_dataset_v2.pt", weights_only=False)
    x = payload["x"].float() / 127.0
    y = payload["y"].long()
    split = payload["split"].bool()
    x_train, y_train = x[split], y[split]
    x_test, y_test = x[~split], y[~split]

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)

    model = KWSClassifier(n_classes=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    class_counts = torch.bincount(y_train, minlength=2).float()
    class_weights = class_counts.sum() / (class_counts.clamp_min(1) * len(class_counts))
    print(f"class_counts={class_counts.tolist()} class_weights={class_weights.tolist()}")
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(10):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
            seen += xb.size(0)
        print(f"epoch {epoch} loss={running / seen:.4f}")

    model.eval()
    with torch.no_grad():
        out = model(x_test)
        acc = (out.argmax(1) == y_test).float().mean().item()
    print(f"test_acc={acc:.4f}")

    torch.save({"model": model.state_dict(), "n_classes": 2}, HERE / "checkpoint.pt")
    print("saved checkpoint.pt")


if __name__ == "__main__":
    main()