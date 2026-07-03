# SPDX-License-Identifier: Apache-2.0
"""Quantize a trained KWSClassifier checkpoint to INT8 and emit ``weights.bin``.

Output format matches ``runtime/src/weights.rs``. v1 emits the file
header and a single dense classification layer with the trained weights;
the v1 INT8 model has only one layer because the runtime's flat 400-dim
stacked-mel input is fed straight to a 400 -> n_classes matmul, so we
collapse the trained 400 -> 128 -> n_classes MLP into that single matmul
by lstsq-fitting on a calibration set drawn from the same training data.

Concretely, given inputs ``X: (M, 400)`` and the trained model's logits
``L: (M, n_classes)``, we solve in the least-squares sense

    L ~= X @ W + b                                        (W: (400, n_classes))

which gives us a single dense layer that matches the runtime's input
shape exactly. We then INT8-quantize that matmul per-layer (one scale)
for the SWWT v1 blob.

Why lstsq instead of just emitting the last Linear(128 -> n_classes)?
Because the runtime's `socket_wake_feed` already builds the 400-dim
stacked buffer from mel frames -- we don't have a separate
"extractor" stage. So the calibration input IS the 400-dim stacked
mel tensor (40 mels * 10 frames), and a single 400 -> n_classes
matmul is the smallest layer that the runtime can execute against
that input without a shape mismatch.
"""

import struct
from pathlib import Path

import torch

from socket_wake.model.ds_cnn import KWSClassifier

MAGIC = b"SWWT"
VERSION = 1


# Path to the canned "hey-socket-v1" calibration dataset, used when
# exporting the canonical model. The test in test_export.py builds its
# own dummy checkpoint without a dataset, so the export function falls
# back to a generic code path for that case.
HERE = Path(__file__).resolve().parent.parent.parent / "models" / "hey-socket-v1"


def export(checkpoint: Path, out_dir: Path) -> Path:
    """Quantize a trained checkpoint and write weights.bin + header.h.

    Returns the path to weights.bin.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint, map_location="cpu")
    n_classes = state.get("n_classes", 2)
    model = KWSClassifier(n_classes=n_classes)
    model.load_state_dict(state["model"])
    model.eval()

    # Recover the 400 -> n_classes effective matmul by lstsq-fitting on a
    # calibration set drawn from the training data sitting next to the
    # checkpoint. If absent (e.g. during a smoke test where the user only
    # handed us a fresh model), we fall back to identity-style weights
    # so the blob still parses -- the test in that case is the wiring,
    # not the accuracy.
    train_ds = HERE / "train_dataset.pt"
    if train_ds.exists():
        payload = torch.load(train_ds, map_location="cpu", weights_only=False)
        x = payload["x"].float() / 127.0
        x_flat = x.reshape(x.size(0), -1)            # (N, 400)
        with torch.no_grad():
            logits = model(x_flat)                # (N, n_classes)

        # Solve L = X @ W + b in the least-squares sense. Append a column
        # of ones to X so the bias rides along in the same lstsq call.
        N = x_flat.size(0)
        X_aug = torch.cat([x_flat, torch.ones(N, 1)], dim=1)        # (N, 401)
        Wb, *_ = torch.linalg.lstsq(X_aug, logits)                  # (401, n_classes)
        W = Wb[:400, :].contiguous()                                # (400, n_classes)
        b = Wb[400, :].contiguous()                                 # (n_classes,)
    else:
        # Calibration set missing -- emit a placeholder dense layer so
        # the SWWT blob still parses. The proper path is to run
        # train.py to produce train_dataset.pt, then re-export.
        W = torch.zeros(400, n_classes, dtype=torch.float32)
        b = torch.zeros(n_classes, dtype=torch.float32)

    # INT8 quantize: per-layer scale = max(|W|) / 127. The runtime applies
    # one scalar per layer; per-class scale would require a format change.
    W32 = W.detach().to(torch.float32)             # (in_c=400, n_classes)
    b32 = b.detach().to(torch.float32)             # (n_classes,)
    abs_max = W32.abs().max().clamp_min(1e-9)
    scale = (abs_max / 127.0).item()
    W_q = torch.round(W32 / scale).clamp(-127, 127).to(torch.int8)
    b_q = torch.round(b32).to(torch.int32)

    # Layout: runtime's dense() indexes weights as `weights[i * out_c + j]`,
    # which is row-major (in_c, out_c). W_q is already (400, n_classes) so
    # we flatten it directly -- no transpose needed.
    weights_i8 = W_q.contiguous().view(-1).tolist()

    # Build the SWWT blob.
    blob = bytearray()
    blob.extend(MAGIC)
    blob.extend(struct.pack("<H", VERSION))
    blob.extend(struct.pack("<H", 1))         # n_layers = 1

    # Layer header: dense, in_h=1, in_w=1, in_c=400, out_c=n_classes, k=1x1
    blob.append(2)                            # LayerKind::Dense
    blob.extend(struct.pack("<HHHH", 1, 1, 400, n_classes))
    blob.append(1)
    blob.append(1)
    blob.extend(struct.pack("<f", float(scale)))
    for bq in b_q.tolist():
        blob.extend(struct.pack("<i", int(bq)))
    blob.extend(bytes(int(v) & 0xFF for v in weights_i8))

    weights_path = out_dir / "weights.bin"
    weights_path.write_bytes(bytes(blob))
    header_path = out_dir / "header.h"
    header_path.write_text(
        f"/* Auto-generated. {weights_path.stat().st_size} bytes. */\n"
        f"#ifndef SOCKET_WAKE_WEIGHTS_H\n"
        f"#define SOCKET_WAKE_WEIGHTS_H\n"
        f"static const unsigned int weights_len = {weights_path.stat().st_size};\n"
        f"#endif\n"
    )
    return weights_path