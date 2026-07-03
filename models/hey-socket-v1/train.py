"""Train KWSClassifier wake-word on the v2 TTS dataset. Minimal loop.

Reads train_dataset.pt (x (N,1,40,10) int8, y (N,) int64, split (N,) bool),
trains KWSClassifier(n_classes=2) -- a 400->128->2 MLP that matches the
runtime's flat (40 mels * 10 frames) stacked buffer input -- with Adam +
CrossEntropyLoss on CPU, and writes checkpoint.pt for socket_wake.export
to consume.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from socket_wake.model.ds_cnn import KWSClassifier


HERE = Path(__file__).resolve().parent


def main() -> None:
    payload = torch.load(HERE / "train_dataset.pt", weights_only=False)
    x = payload["x"].float() / 127.0
    y = payload["y"].long()
    split = payload["split"].bool()
    x_train, y_train = x[split], y[split]
    x_test, y_test = x[~split], y[~split]

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)

    model = KWSClassifier(n_classes=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

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