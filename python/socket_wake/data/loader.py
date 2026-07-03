# SPDX-License-Identifier: Apache-2.0
"""Speech Commands dataset loader (torchaudio).

Optional dependency: requires ``torchaudio`` for the underlying dataset.
v1 callers can skip the download by passing user-supplied WAVs to
``train.py`` instead.
"""

from pathlib import Path


def load_speech_commands(root: Path):
    """Load the Speech Commands v2 dataset via torchaudio.

    Returns a ``torchaudio.datasets.SPEECHCOMMANDS`` instance compatible
    with the standard ``DataLoader`` protocol.
    """
    try:
        import torchaudio
    except ImportError as e:
        raise RuntimeError(
            "torchaudio is required for the Speech Commands loader; "
            "install with `pip install -e ./python[train]`"
        ) from e
    root.mkdir(parents=True, exist_ok=True)
    return torchaudio.datasets.SPEECHCOMMANDS(str(root), download=True, subset="training")