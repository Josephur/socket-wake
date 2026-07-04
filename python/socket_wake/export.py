# SPDX-License-Identifier: Apache-2.0
"""Export a trained KWSConvNet checkpoint to an SWWT v2 weights blob.

This replaces the v1 lstsq single-layer collapse: every layer of the
BN-folded network is quantized per-tensor symmetric INT8 and emitted as
its own layer record, so the runtime executes the same graph that
``Int8Sim`` validated (see docs/training.md "Measured v3 results").

Quantization arithmetic
-----------------------
``Int8Sim`` fake-quantizes weights (scale = max|w|/127) and activations
(calibrated scales), keeps biases in float, and requantizes after ReLU.
The integer equivalent executed by the runtime is, per layer::

    acc   = sum(x_int * w_int)                    (INT32)
    out   = clamp(rint(acc * M + B[o]), lo, 127)  (INT8)
    M     = s_in * s_w / s_out                    (the layer's `scale`)
    B     = bias_float / s_out                    (the layer's f32 bias)

The final dense layer folds global-average-pooling into ``M`` (divide by
h*w) and quantizes the float logits at an export-chosen ``s_logit``. The
state-machine threshold in the header converts the eval_v3 probability
threshold into a quantized logit margin: p >= theta iff
(logit_target - logit_not) >= ln(theta/(1-theta)), so
``logit_thr = ceil(ln(theta/(1-theta)) / s_logit)``.

Bit-exactness is defined against ``socket_wake.int8_ref`` (the Python twin
of the Rust kernels); this module also emits parity vectors that
``runtime/tests/parity_test.rs`` checks against the Rust implementation,
and verifies the integer pipeline agrees with ``Int8Sim`` to within a
small probability tolerance on real windows.

Usage:
    python -m socket_wake.export          # models/hey-socket-v1/checkpoint_v3.pt
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import torch

from socket_wake import int8_ref
from socket_wake.int8_ref import (
    KIND_CONV2D,
    KIND_DENSE,
    KIND_DEPTHWISE,
    KIND_POINTWISE,
    QLayer,
    QModel,
)
from socket_wake.model.kws_cnn import Int8Sim, KWSConvNet, fold_batchnorm

MAGIC = b"SWWT"
VERSION = 2
TV_MAGIC = b"SWTV"
TV_VERSION = 1

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "hey-socket-v1"

# Streaming detector parameters validated in eval_v3.py: fire at
# p(target) >= 0.95 for 2 consecutive inferences, 1 s refractory at the
# 30 ms inference cadence (98 frames / 3 frames-per-inference).
PROB_THRESHOLD = 0.95
HOLD = 2
REFRACTORY = 32


def _wq(w: torch.Tensor) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor INT8 quantization, identical to Int8Sim."""
    s = max(w.abs().max().item() / 127.0, 1e-9)
    q = torch.round(w / s).clamp(-127, 127).to(torch.int8)
    return q.numpy(), s


def quantize(folded: KWSConvNet, calib: torch.Tensor) -> QModel:
    """Build the integer model from a BN-folded float model.

    `calib` is a batch of float windows (N, 40, 98) already scaled by
    1/127 -- the same tensor Int8Sim calibrates on.
    """
    sim = Int8Sim(folded, calib)
    s = sim.act_scales
    m = folded

    # Weight quantization (per-tensor symmetric, torch round = ties-even).
    conv1_q, s_conv1w = _wq(m.conv1.weight.data)     # (16, 1, 3, 3) OIHW
    dw2_q, s_dw2w = _wq(m.dw2.weight.data)           # (16, 1, 3, 3)
    pw2_q, s_pw2w = _wq(m.pw2.weight.data)           # (32, 16, 1, 1)
    dw3_q, s_dw3w = _wq(m.dw3.weight.data)           # (32, 1, 3, 3)
    pw3_q, s_pw3w = _wq(m.pw3.weight.data)           # (32, 32, 1, 1)
    fc_q, s_fcw = _wq(m.fc.weight.data)              # (2, 32)

    # Logit scale: cover the calibration logit range with ~25% headroom so
    # real-data logits don't clip at +/-127 near the decision boundary.
    with torch.no_grad():
        calib_logits = sim.forward(calib)
    margin = math.log(PROB_THRESHOLD / (1 - PROB_THRESHOLD))
    # Floor the scale so the threshold always fits in i8 (<= 110) even for
    # low-confidence models; the floor only coarsens quantization when the
    # logit range is tiny.
    s_logit = max(calib_logits.abs().max().item() * 1.25 / 127.0, margin / 110.0)
    logit_thr = math.ceil(margin / s_logit)

    def bias(conv: torch.nn.Module) -> np.ndarray:
        return conv.bias.data.numpy().astype(np.float64)

    layers = [
        # conv1: (40, 98, 1) -> (20, 49, 16). OIHW -> OHWI.
        QLayer(KIND_CONV2D, 40, 98, 1, 16, 3, 2, True,
               np.float32(s["input"] * s_conv1w / s["conv1"]),
               (bias(m.conv1) / s["conv1"]).astype(np.float32),
               np.transpose(conv1_q, (0, 2, 3, 1))),
        # dw2: (20, 49, 16) -> (10, 25, 16). (c, 1, k, k) -> (c, k, k).
        QLayer(KIND_DEPTHWISE, 20, 49, 16, 16, 3, 2, True,
               np.float32(s["conv1"] * s_dw2w / s["dw2"]),
               (bias(m.dw2) / s["dw2"]).astype(np.float32),
               dw2_q[:, 0, :, :]),
        # pw2: (10, 25, 16) -> (10, 25, 32). (out, in, 1, 1) -> (in, out).
        QLayer(KIND_POINTWISE, 10, 25, 16, 32, 1, 1, True,
               np.float32(s["dw2"] * s_pw2w / s["pw2"]),
               (bias(m.pw2) / s["pw2"]).astype(np.float32),
               pw2_q[:, :, 0, 0].T.copy()),
        # dw3: (10, 25, 32) -> (5, 13, 32).
        QLayer(KIND_DEPTHWISE, 10, 25, 32, 32, 3, 2, True,
               np.float32(s["pw2"] * s_dw3w / s["dw3"]),
               (bias(m.dw3) / s["dw3"]).astype(np.float32),
               dw3_q[:, 0, :, :]),
        # pw3: (5, 13, 32) -> (5, 13, 32).
        QLayer(KIND_POINTWISE, 5, 13, 32, 32, 1, 1, True,
               np.float32(s["dw3"] * s_pw3w / s["pw3"]),
               (bias(m.pw3) / s["pw3"]).astype(np.float32),
               pw3_q[:, :, 0, 0].T.copy()),
        # fc with GAP folded: (5, 13, 32) -> 2 logits. (out, in) -> (in, out).
        QLayer(KIND_DENSE, 5, 13, 32, 2, 1, 1, False,
               np.float32(s["pw3"] * s_fcw / (65.0 * s_logit)),
               (m.fc.bias.data.numpy().astype(np.float64) / s_logit).astype(np.float32),
               fc_q.T.copy()),
    ]
    # Raw mel i8 -> Int8Sim input grid: windows enter Int8Sim as mel/127
    # and are re-quantized at s["input"], so one multiply covers both.
    input_scale = 1.0 / (127.0 * s["input"])
    model = QModel(np.float32(input_scale), logit_thr, HOLD, REFRACTORY, layers)
    model.s_logit = s_logit          # for callers that map logits back to probs
    return model


def serialize(model: QModel) -> bytes:
    blob = bytearray()
    blob.extend(MAGIC)
    blob.extend(struct.pack("<HH", VERSION, len(model.layers)))
    blob.extend(struct.pack("<f", float(model.input_scale)))
    blob.extend(struct.pack("<bBH", model.logit_thr, model.hold, model.refractory))
    for l in model.layers:
        blob.append(l.kind)
        blob.extend(struct.pack("<HHHH", l.in_h, l.in_w, l.in_c, l.out_c))
        blob.extend(struct.pack("<BBBB", l.k, l.k, l.stride, int(l.relu)))
        blob.extend(struct.pack("<f", float(l.scale)))
        blob.extend(l.bias.astype("<f4").tobytes())
        blob.extend(l.weights.astype(np.int8).reshape(-1).tobytes())
    return bytes(blob)


def make_parity_vectors(model: QModel, windows: np.ndarray) -> bytes:
    """Emit raw mel windows + expected logits for the Rust parity test."""
    blob = bytearray()
    blob.extend(TV_MAGIC)
    blob.extend(struct.pack("<HH", TV_VERSION, windows.shape[0]))
    for win in windows:
        logits = int8_ref.forward(model, win)
        blob.extend(win.astype(np.int8).reshape(-1).tobytes())   # (40, 98) row-major
        blob.extend(logits.astype(np.int8).tobytes())            # 2 bytes
    return bytes(blob)


def check_against_sim(model: QModel, folded: KWSConvNet, calib: torch.Tensor,
                      windows: np.ndarray) -> float:
    """Max |p_target| gap between Int8Sim (float sim) and the integer ref."""
    sim = Int8Sim(folded, calib)
    x = torch.from_numpy(windows).float().unsqueeze(1) / 127.0
    with torch.no_grad():
        p_sim = torch.softmax(sim.forward(x), dim=1)[:, 1].numpy()
    p_ref = []
    for win in windows:
        logits = int8_ref.forward(model, win).astype(np.float64) * model.s_logit
        e = np.exp(logits - logits.max())
        p_ref.append((e / e.sum())[1])
    return float(np.abs(p_sim - np.asarray(p_ref)).max())


def export(checkpoint: Path = MODEL_DIR / "checkpoint_v3.pt",
           dataset: Path = MODEL_DIR / "train_dataset_v3.pt",
           out_dir: Path = MODEL_DIR,
           n_vectors: int = 24) -> Path:
    """Quantize a trained v3 checkpoint and write weights_v3.bin,
    testvectors_v3.bin and header.h. Returns the weights path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net = KWSConvNet(n_classes=state.get("n_classes", 2))
    net.load_state_dict(state["model"])
    folded = fold_batchnorm(net)

    payload = torch.load(dataset, map_location="cpu", weights_only=False)
    x_all, y_all, split = payload["x"], payload["y"], payload["split"].bool()
    calib = x_all[split][:512].float() / 127.0        # same slice as eval_v3

    model = quantize(folded, calib)

    # Parity vectors from held-out windows, both classes represented.
    test_x, test_y = x_all[~split], y_all[~split]
    pos = test_x[test_y == 1][: n_vectors // 2]
    neg = test_x[test_y == 0][: n_vectors - len(pos)]
    # (N, 1, 40, 98) -> (N, 40, 98): raw mel windows, channel squeezed.
    windows = torch.cat([pos, neg]).squeeze(1).numpy().astype(np.int8)

    gap = check_against_sim(model, folded, calib, windows)
    print(f"[export] integer-ref vs Int8Sim: max |p_target| gap = {gap:.5f} "
          f"on {len(windows)} held-out windows")
    if gap > 0.02:
        raise AssertionError(
            f"integer pipeline diverges from Int8Sim (gap {gap:.4f} > 0.02); "
            "the exported model would not match the validated metrics")

    blob = serialize(model)
    weights_path = out_dir / "weights_v3.bin"
    weights_path.write_bytes(blob)
    (out_dir / "testvectors_v3.bin").write_bytes(make_parity_vectors(model, windows))
    (out_dir / "header.h").write_text(
        f"/* Auto-generated. {len(blob)} bytes. */\n"
        f"#ifndef SOCKET_WAKE_WEIGHTS_H\n"
        f"#define SOCKET_WAKE_WEIGHTS_H\n"
        f"static const unsigned int weights_len = {len(blob)};\n"
        f"#endif\n"
    )
    print(f"[export] wrote {weights_path} ({len(blob)} bytes), "
          f"logit_thr={model.logit_thr} (theta={PROB_THRESHOLD}), "
          f"s_logit={model.s_logit:.5f}, input_scale={model.input_scale:.6f}")
    return weights_path


if __name__ == "__main__":
    export()
