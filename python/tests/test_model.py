# SPDX-License-Identifier: Apache-2.0
import torch

from socket_wake.model.ds_cnn import DSCNN


def test_dscnn_forward_shape_2d_input():
    m = DSCNN(n_classes=12)
    x = torch.zeros(2, 40)               # one mel frame as 2D
    out = m(x)
    assert out.shape == (2, 12)


def test_dscnn_forward_shape_4d_input():
    m = DSCNN(n_classes=12)
    x = torch.zeros(2, 1, 40, 10)         # batched stacked frames
    out = m(x)
    assert out.shape == (2, 12)


def test_dscnn_param_count_under_30k():
    m = DSCNN(n_classes=12)
    n = sum(p.numel() for p in m.parameters())
    assert n < 30_000, f"DS-CNN-L should be ~24K params, got {n}"