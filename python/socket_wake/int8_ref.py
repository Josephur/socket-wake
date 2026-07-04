# SPDX-License-Identifier: Apache-2.0
"""Integer reference implementation of the SWWT v2 runtime kernels.

This is the Python twin of ``runtime/src/cnn.rs`` and MUST stay
bit-identical to it: the exporter uses it to generate the parity vectors
that ``runtime/tests/parity_test.rs`` verifies. The contract:

  - activations and weights are INT8, accumulation is exact integer math
  - requantization: ``out = clamp(rint(acc_f32 * scale + bias[o]), lo, 127)``
    where ``rint`` rounds half to even, ``scale``/``bias`` are f32, the
    multiply and add each round to f32 (matching Rust f32 ops), and
    ``lo = 0`` when the layer has ReLU else ``-127``
  - conv padding is `same` ((k-1)/2 both sides), stride 1 or 2
  - dense layers with a spatial input sum over all positions with shared
    weights (global-average-pool folded into ``scale``)

Layouts match the weights file: activations HWC; conv2d weights OHWI;
depthwise (c, k, k); pointwise and dense (in_c, out_c).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KIND_DEPTHWISE = 0
KIND_POINTWISE = 1
KIND_DENSE = 2
KIND_CONV2D = 3


@dataclass
class QLayer:
    kind: int
    in_h: int
    in_w: int
    in_c: int
    out_c: int
    k: int
    stride: int
    relu: bool
    scale: float                 # requant multiplier M (stored as f32)
    bias: np.ndarray             # f32 (out_c,), already divided by s_out
    weights: np.ndarray          # int8, runtime layout (see module docstring)


@dataclass
class QModel:
    input_scale: float           # mel i8 -> conv input requant multiplier
    logit_thr: int               # state-machine threshold on quantized margin
    hold: int
    refractory: int
    layers: list[QLayer] = field(default_factory=list)


def _requant(acc: np.ndarray, scale: float, bias: np.ndarray, relu: bool) -> np.ndarray:
    """Bit-exact twin of cnn.rs::requant (f32 mul, f32 add, rint, clamp)."""
    q = acc.astype(np.float32) * np.float32(scale) + bias.astype(np.float32)
    q = np.rint(q)
    lo = 0.0 if relu else -127.0
    return np.clip(q, lo, 127.0).astype(np.int8)


def quantize_input(x: np.ndarray, input_scale: float) -> np.ndarray:
    """Raw mel i8 -> model input i8 (twin of lib.rs::push_frame requant)."""
    q = np.rint(x.astype(np.float32) * np.float32(input_scale))
    return np.clip(q, -127.0, 127.0).astype(np.int8)


def _conv2d(x: np.ndarray, l: QLayer) -> np.ndarray:
    """General conv, weights (out_c, k, k, in_c); x is (h, w, in_c) int8."""
    h, w, in_c = x.shape
    k, s = l.k, l.stride
    pad = (k - 1) // 2
    out_h = (h + 2 * pad - k) // s + 1
    out_w = (w + 2 * pad - k) // s + 1
    xp = np.zeros((h + 2 * pad, w + 2 * pad, in_c), dtype=np.int64)
    xp[pad : pad + h, pad : pad + w, :] = x
    wgt = l.weights.astype(np.int64)                       # (out_c, k, k, in_c)
    acc = np.zeros((out_h, out_w, l.out_c), dtype=np.int64)
    for ky in range(k):
        for kx in range(k):
            patch = xp[ky : ky + s * (out_h - 1) + 1 : s,
                       kx : kx + s * (out_w - 1) + 1 : s, :]
            acc += patch @ wgt[:, ky, kx, :].T             # (in_c, out_c) via .T
    return _requant(acc, l.scale, l.bias, l.relu)


def _depthwise(x: np.ndarray, l: QLayer) -> np.ndarray:
    """Depthwise conv, weights (c, k, k); x is (h, w, c) int8."""
    h, w, c = x.shape
    k, s = l.k, l.stride
    pad = (k - 1) // 2
    out_h = (h + 2 * pad - k) // s + 1
    out_w = (w + 2 * pad - k) // s + 1
    xp = np.zeros((h + 2 * pad, w + 2 * pad, c), dtype=np.int64)
    xp[pad : pad + h, pad : pad + w, :] = x
    wgt = l.weights.astype(np.int64)                       # (c, k, k)
    acc = np.zeros((out_h, out_w, c), dtype=np.int64)
    for ky in range(k):
        for kx in range(k):
            patch = xp[ky : ky + s * (out_h - 1) + 1 : s,
                       kx : kx + s * (out_w - 1) + 1 : s, :]
            acc += patch * wgt[:, ky, kx]                  # broadcast over hw
    return _requant(acc, l.scale, l.bias, l.relu)


def _pointwise(x: np.ndarray, l: QLayer) -> np.ndarray:
    """Shared-weight 1x1 conv, weights (in_c, out_c); x is (h, w, in_c)."""
    h, w, in_c = x.shape
    acc = x.reshape(h * w, in_c).astype(np.int64) @ l.weights.astype(np.int64)
    return _requant(acc.reshape(h, w, l.out_c), l.scale, l.bias, l.relu)


def _dense(x: np.ndarray, l: QLayer) -> np.ndarray:
    """Dense over (h, w, in_c) or (in_c,): sums spatial positions (GAP is
    folded into `scale` by the exporter)."""
    flat = x.reshape(-1, l.in_c).astype(np.int64).sum(axis=0)
    acc = flat @ l.weights.astype(np.int64)                # (out_c,)
    return _requant(acc, l.scale, l.bias, l.relu)


_APPLY = {
    KIND_CONV2D: _conv2d,
    KIND_DEPTHWISE: _depthwise,
    KIND_POINTWISE: _pointwise,
    KIND_DENSE: _dense,
}


def forward(model: QModel, mel_window: np.ndarray) -> np.ndarray:
    """Run one inference. `mel_window` is the raw i8 mel window (40, 98)
    = (n_mels, n_frames), exactly what the runtime's frame ring assembles.
    Returns the quantized logits (int8, one per class)."""
    x = quantize_input(mel_window, model.input_scale)
    if x.ndim == 2:
        x = x[:, :, None]                                  # HWC with c=1
    for l in model.layers:
        x = _APPLY[l.kind](x, l)
    return x


def margin(logits: np.ndarray) -> int:
    """Quantized logit margin fed to the state machine (twin of lib.rs)."""
    m = int(logits[1]) - int(logits[0])
    return max(-128, min(127, m))
