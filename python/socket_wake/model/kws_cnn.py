# SPDX-License-Identifier: Apache-2.0
"""KWSConvNet: small depthwise-separable CNN for 1 s mel windows.

Standard-KWS shape (Hello Edge DS-CNN lineage, sized down): the input is
one (1, 40, 98) INT8 mel window = 1.0 s of audio; output is 2-class
logits. ~2.3K parameters, ~0.36M MACs per inference -- at a 30 ms
inference cadence that's ~12M MACs/s, a few percent of one ESP32-P4 core.

BatchNorm is used during training and folded into the preceding conv for
inference/quantization (`fold_batchnorm`), so the deployed graph is pure
conv/ReLU/dense -- the shape the runtime kernel executes.

`int8_sim_forward` simulates post-training INT8 quantization (symmetric
per-tensor weights + activations, scales calibrated on training data) so
we can measure the quantization penalty *before* committing to the
runtime export work.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class KWSConvNet(nn.Module):
    def __init__(self, n_classes: int = 2) -> None:
        super().__init__()
        # (1, 40, 98) -> (16, 20, 49)
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # depthwise separable, stride 2: -> (32, 10, 25)
        self.dw2 = nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16, bias=False)
        self.bn2a = nn.BatchNorm2d(16)
        self.pw2 = nn.Conv2d(16, 32, 1, bias=False)
        self.bn2b = nn.BatchNorm2d(32)
        # depthwise separable, stride 2: -> (32, 5, 13)
        self.dw3 = nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32, bias=False)
        self.bn3a = nn.BatchNorm2d(32)
        self.pw3 = nn.Conv2d(32, 32, 1, bias=False)
        self.bn3b = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32, n_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2a(self.dw2(x)))
        x = self.relu(self.bn2b(self.pw2(x)))
        x = self.relu(self.bn3a(self.dw3(x)))
        x = self.relu(self.bn3b(self.pw3(x)))
        x = x.mean(dim=(2, 3))            # global average pool
        return self.fc(x)


def fold_batchnorm(model: KWSConvNet) -> KWSConvNet:
    """Return a copy with every BN folded into its preceding conv.

    After folding, each conv gains a bias and the BN layers become
    identity -- the graph matches what an integer runtime executes.
    """
    m = copy.deepcopy(model).eval()
    pairs = [(m.conv1, m.bn1), (m.dw2, m.bn2a), (m.pw2, m.bn2b),
             (m.dw3, m.bn3a), (m.pw3, m.bn3b)]
    for conv, bn in pairs:
        w = conv.weight.data
        gamma = bn.weight.data
        beta = bn.bias.data
        mean = bn.running_mean.data
        var = bn.running_var.data
        eps = bn.eps
        scale = gamma / torch.sqrt(var + eps)          # (out_c,)
        conv.weight.data = w * scale.reshape(-1, 1, 1, 1)
        bias = beta - mean * scale
        conv.bias = nn.Parameter(bias.clone())
        # neutralize the BN
        bn.weight.data.fill_(1.0)
        bn.bias.data.zero_()
        bn.running_mean.data.zero_()
        bn.running_var.data.fill_(1.0 - eps)
    return m


def _qdq(t: torch.Tensor, scale: float) -> torch.Tensor:
    """Symmetric INT8 quantize-dequantize at the given scale."""
    return (t / scale).round().clamp(-127, 127) * scale


class Int8Sim:
    """Post-training INT8 simulation of a BN-folded KWSConvNet.

    Weights: symmetric per-tensor scale = max|w| / 127.
    Activations: symmetric per-tensor scales calibrated as the 99.9th
    percentile of |activation| over a calibration batch (clipping the
    long tail is standard and loses less than scaling to the absolute max).
    """

    STAGES = ("input", "conv1", "dw2", "pw2", "dw3", "pw3")

    def __init__(self, folded: KWSConvNet, calib_x: torch.Tensor) -> None:
        self.m = folded.eval()
        self.act_scales: dict[str, float] = {}
        with torch.no_grad():
            acts = self._collect(calib_x)
        for name, a in acts.items():
            q = torch.quantile(a.abs().flatten().float(), 0.999).item()
            self.act_scales[name] = max(q, 1e-6) / 127.0

    def _collect(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        m, r = self.m, self.m.relu
        out: dict[str, torch.Tensor] = {}
        if x.dim() == 3:
            x = x.unsqueeze(1)
        out["input"] = x
        x = r(m.conv1(x)); out["conv1"] = x
        x = r(m.dw2(x));   out["dw2"] = x
        x = r(m.pw2(x));   out["pw2"] = x
        x = r(m.dw3(x));   out["dw3"] = x
        x = r(m.pw3(x));   out["pw3"] = x
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with fake-quantized weights and activations."""
        m, r, s = self.m, self.m.relu, self.act_scales

        def qconv(conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
            wscale = conv.weight.abs().max().item() / 127.0
            w = _qdq(conv.weight, max(wscale, 1e-9))
            return nn.functional.conv2d(
                x, w, conv.bias, conv.stride, conv.padding,
                conv.dilation, conv.groups)

        with torch.no_grad():
            if x.dim() == 3:
                x = x.unsqueeze(1)
            x = _qdq(x, s["input"])
            x = _qdq(r(qconv(m.conv1, x)), s["conv1"])
            x = _qdq(r(qconv(m.dw2, x)), s["dw2"])
            x = _qdq(r(qconv(m.pw2, x)), s["pw2"])
            x = _qdq(r(qconv(m.dw3, x)), s["dw3"])
            x = _qdq(r(qconv(m.pw3, x)), s["pw3"])
            x = x.mean(dim=(2, 3))
            wscale = m.fc.weight.abs().max().item() / 127.0
            w = _qdq(m.fc.weight, max(wscale, 1e-9))
            return nn.functional.linear(x, w, m.fc.bias)


def count_macs(n_classes: int = 2) -> dict[str, int]:
    """Analytic multiply-accumulate count per single inference."""
    macs = {
        "conv1": 16 * 1 * 9 * 20 * 49,
        "dw2": 16 * 9 * 10 * 25,
        "pw2": 16 * 32 * 10 * 25,
        "dw3": 32 * 9 * 5 * 13,
        "pw3": 32 * 32 * 5 * 13,
        "fc": 32 * n_classes,
    }
    macs["total"] = sum(macs.values())
    return macs


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
