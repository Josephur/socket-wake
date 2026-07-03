"""Train the DS-CNN-L wake-word model on the synthetic v1 dataset.

Reads ``train_data.npz`` (target / not_target, each (N, 1, 40, 10)),
trains for ~30 epochs with Adam + CrossEntropyLoss, and writes
``checkpoint.pt`` for ``socket_wake.export`` to consume.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from socket_wake.model.ds_cnn import DSCNN


HERE = Path(__file__).resolve().parent


def load_data(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    npz = np.load(path)
    x = np.concatenate([npz["target"], npz["not_target"]], axis=0)
    y = np.concatenate(
        [
            np.zeros(npz["target"].shape[0], dtype=np.int64),
            np.ones(npz["not_target"].shape[0], dtype=np.int64),
        ]
    )
    return torch.from_numpy(x).float(), torch.from_numpy(y)


def main(
    data_path: Path = HERE / "train_data.npz",
    out_path: Path = HERE / "checkpoint.pt",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x, y = load_data(data_path)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = DSCNN(n_classes=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running = 0.0
        correct = 0
        seen = 0
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            seen += xb.size(0)
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"epoch {epoch:3d}  loss={running / seen:.4f}  acc={correct / seen:.3f}"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "n_classes": 2}, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()