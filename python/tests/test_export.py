# SPDX-License-Identifier: Apache-2.0
import struct

import numpy as np
import torch

from socket_wake import int8_ref
from socket_wake.export import quantize, serialize
from socket_wake.model.kws_cnn import KWSConvNet, fold_batchnorm


def _fresh_qmodel():
    torch.manual_seed(0)
    net = KWSConvNet(n_classes=2)
    # Push some data through so BatchNorm has non-trivial running stats.
    net.train()
    with torch.no_grad():
        net(torch.randn(8, 1, 40, 98))
    net.eval()
    folded = fold_batchnorm(net)
    calib = torch.randint(-127, 128, (32, 1, 40, 98), dtype=torch.int8)
    return quantize(folded, calib.float() / 127.0)


def test_serialized_blob_has_v2_header_and_all_layers():
    model = _fresh_qmodel()
    blob = serialize(model)
    assert blob[:4] == b"SWWT"
    version, n_layers = struct.unpack("<HH", blob[4:8])
    assert version == 2
    assert n_layers == 6
    (input_scale,) = struct.unpack("<f", blob[8:12])
    assert input_scale > 0
    thr, hold, refractory = struct.unpack("<bBH", blob[12:16])
    assert 1 <= thr <= 127
    assert hold == 2
    assert refractory == 32
    # ~2.4K params INT8 + f32 biases + headers: well under the 50 KB target.
    assert len(blob) < 50 * 1024


def test_integer_ref_runs_and_is_deterministic():
    model = _fresh_qmodel()
    rng = np.random.default_rng(1)
    win = rng.integers(-127, 128, size=(40, 98)).astype(np.int8)
    a = int8_ref.forward(model, win)
    b = int8_ref.forward(model, win)
    assert a.shape == (2,)
    assert a.dtype == np.int8
    assert np.array_equal(a, b)


def test_layer_shapes_thread_through():
    model = _fresh_qmodel()
    dims = [(l.in_h, l.in_w, l.in_c, l.out_c) for l in model.layers]
    assert dims == [
        (40, 98, 1, 16),
        (20, 49, 16, 16),
        (10, 25, 16, 32),
        (10, 25, 32, 32),
        (5, 13, 32, 32),
        (5, 13, 32, 2),
    ]
