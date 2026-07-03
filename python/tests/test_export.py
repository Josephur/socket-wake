# SPDX-License-Identifier: Apache-2.0
import struct
from pathlib import Path

import torch

from socket_wake.export import export
from socket_wake.model.ds_cnn import KWSClassifier


def test_export_produces_valid_blob(tmp_path: Path):
    m = KWSClassifier(n_classes=2)
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": m.state_dict(), "n_classes": 2}, ckpt)
    out_dir = tmp_path / "out"
    blob = export(ckpt, out_dir)
    assert blob.exists()
    head = blob.read_bytes()[:8]
    assert head[:4] == b"SWWT"
    version = struct.unpack("<H", head[4:6])[0]
    assert version == 1
    n_layers = struct.unpack("<H", head[6:8])[0]
    assert n_layers == 1
    # Whole-file size must respect the global < 50 KB target on a tiny
    # synthetic model -- on a real trained model this should be ~800 B.
    assert blob.stat().st_size < 50 * 1024


def test_export_writes_header_h(tmp_path: Path):
    m = KWSClassifier(n_classes=2)
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": m.state_dict(), "n_classes": 2}, ckpt)
    out_dir = tmp_path / "out"
    export(ckpt, out_dir)
    header = (out_dir / "header.h").read_text()
    assert "weights_len" in header
    assert "SOCKET_WAKE_WEIGHTS_H" in header