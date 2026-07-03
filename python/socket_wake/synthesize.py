# SPDX-License-Identifier: Apache-2.0
"""TTS data augmentation via Piper.

Piper is an Apache/MIT-licensed offline TTS engine. Voice variety comes
from ~40 built-in voices (30+ languages) plus per-call pitch/speed jitter
that Piper applies automatically when invoked from the CLI.

v1 callers can substitute recorded utterances for synthesis; v2 adds a
``synthesize_dataset(phrase, out_dir)`` helper that emits a labeled
manifest compatible with the training pipeline.
"""

import shutil
import subprocess
from pathlib import Path


def piper_available() -> bool:
    """Whether the ``piper`` CLI is on PATH."""
    return shutil.which("piper") is not None


def synthesize_phrase(phrase: str, voice: str, out_dir: Path, n: int = 100) -> list[Path]:
    """Generate ``n`` synthetic utterances of ``phrase`` with the given voice.

    Each call invokes Piper once per utterance; voice variety comes from
    Piper's built-in pitch/speed jitter (no extra config needed).
    """
    if not piper_available():
        raise RuntimeError("piper-tts not installed; see docs/training.md")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(n):
        out = out_dir / f"{phrase.replace(' ', '_')}_{voice}_{i:04d}.wav"
        subprocess.run(
            ["piper", "--model", voice, "--output_file", str(out)],
            input=phrase, text=True, check=True,
        )
        paths.append(out)
    return paths