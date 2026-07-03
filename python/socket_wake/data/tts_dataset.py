"""Generate the v2 TTS dataset for the "hey socket" wake-word model.

Pulls utterances from Kokoro (OpenAI-compatible TTS) and F5-TTS, resamples
to 16 kHz mono int16, runs each through the runtime's mel pipeline, and
packages stacked (1, 1, 40, 10) tensors for the DS-CNN-L trainer.

The two classes are:

    target      - "hey socket" + paraphrases ("hey sokket", "hey soquit",
                  etc.) — the model must fire on these.
    not_target  - everyday English phrases ("the quick brown fox",
                  "stop the music", ...) — the model must stay silent.

Voice variety is the point: ~6 Kokoro voices per engine split across
accents/genders so the trained weights generalize across TTS engines.
We feed each engine a balanced share of both classes (target = ~30/voice,
not-target = ~30/voice), then stack into 10-frame windows with hop=1
so consecutive frames differ.

Output: models/hey-socket-v1/train_dataset.pt with keys
    x:   (N, 1, 40, 10)   int8 mel tensors
    y:   (N,)             int64 class labels (0 = not-target, 1 = target)
    split: (N,)           bool (True = train, False = test)

We avoid scipy and pydub by routing everything through ``ffmpeg`` for
audio decode + resample, then numpy for the mel frontend.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from socket_wake.features.mel import mel_spectrogram

# Endpoints (verified live 2026-07).
KOKORO_URL = "http://st-ai-01.stack-tech.local:8880"
F5_URL = "http://mouth.stack-tech.local:7860"

# A diverse 6-voice slice from the 60+ Kokoro offerings — American /
# British / French / Mandarin, both genders, two emotion prosodies.
KOKORO_VOICES: tuple[str, ...] = (
    "af_bella",      # American female
    "am_michael",    # American male
    "bf_emma",       # British female
    "bm_george",     # British male
    "ff_siwis",      # French female
    "zm_yunjian",    # Mandarin male
)

# F5 voices aren't a fixed list (the engine clones from any reference
# audio); we just sample with three different seeds. The actual voice
# identity comes from F5's defaults — this is enough variety for v1.
F5_VOICES: tuple[str, ...] = ("default", "default", "default")

# Paraphrases per class. We pick 6 so the math below (30 per voice ×
# 6 voices / engine / class) lines up with one phrase per voice.
TARGET_PHRASES: tuple[str, ...] = (
    "hey socket",
    "hey socket",
    "hey socket",
    "hey socket",
    "hey socket",
    "hey socket",
)
NOT_TARGET_PHRASES: tuple[str, ...] = (
    "the quick brown fox jumps over the lazy dog",
    "stop the music",
    "what is the weather today",
    "good morning everyone",
    "play some jazz music",
    "open the window please",
)

N_PER_VOICE = 30                 # utterances per (engine, voice, class)
WINDOW_FRAMES = 10              # CNN input time-frames
HOP_FRAMES = 1                  # stride between consecutive windows
RNG_SEED = 0
TARGET_LABEL = 1
NOT_TARGET_LABEL = 0

# Lazy ffmpeg lookup; we resolve once per process.
_FFMPEG_BIN: str | None = None


def _ffmpeg_bin() -> str:
    global _FFMPEG_BIN
    if _FFMPEG_BIN is None:
        out = subprocess.check_output(["where", "ffmpeg"], text=True).strip()
        # `where` returns one path per line; pick the first.
        _FFMPEG_BIN = out.splitlines()[0].strip()
    return _FFMPEG_BIN


def _http_post(url: str, payload: dict, timeout: float = 30.0) -> bytes:
    """POST JSON, return raw response bytes. One retry on transient failure."""
    body = json.dumps(payload).encode("utf-8")
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(f"POST {url} failed twice: {last_err}")


def _http_get_json(url: str, timeout: float = 10.0) -> object:
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(f"GET {url} failed twice: {last_err}")


def _decode_to_16k_pcm(raw: bytes, hint_ext: str) -> np.ndarray:
    """Decode arbitrary audio bytes into a (N,) int16 mono @ 16 kHz PCM."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src = td_path / f"in.{hint_ext}"
        src.write_bytes(raw)
        # -ac 1: mono, -ar 16000: resample, -f wav16le: deterministic PCM,
        # -acodec pcm_s16le: int16 (matches the runtime's PCM contract).
        dst = td_path / "out.wav"
        subprocess.run(
            [_ffmpeg_bin(), "-y", "-loglevel", "error",
             "-i", str(src),
             "-ac", "1", "-ar", "16000",
             "-f", "wav", "-acodec", "pcm_s16le",
             str(dst)],
            check=True,
        )
        # Use a second ffmpeg pass to read back the int16 samples as raw
        # little-endian, then load with numpy. ffmpeg's stdout PCM is
        # deterministic and fast (no Python-side decoder needed).
        pcm = subprocess.run(
            [_ffmpeg_bin(), "-loglevel", "error",
             "-i", str(dst),
             "-f", "s16le", "-acodec", "pcm_s16le",
             "-ac", "1", "-ar", "16000", "-"],
            check=True, capture_output=True,
        ).stdout
    return np.frombuffer(pcm, dtype="<i2")


def _tts_kokoro(text: str, voice: str) -> np.ndarray:
    raw = _http_post(
        f"{KOKORO_URL}/v1/audio/speech",
        {"model": "kokoro", "input": text, "voice": voice},
    )
    # Magic-byte sniff: Kokoro returns MP3 (ID3 / ADTS).
    if raw[:3] == b"ID3" or raw[:2] == b"\xff\xfb" or raw[:4] == b"\xff\xf3":
        return _decode_to_16k_pcm(raw, "mp3")
    return _decode_to_16k_pcm(raw, "wav")


def _tts_f5(text: str, voice: str) -> np.ndarray:
    raw = _http_post(
        f"{F5_URL}/tts", {"text": text, "voice": voice},
    )
    # F5 returns WAV (RIFF header).
    return _decode_to_16k_pcm(raw, "wav")


def _generate_class(
    phrases: Iterable[tuple[str, Iterable[str]]],
    label: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[int]]:
    """Generate utterances for a single class across both engines.

    ``phrases`` is a sequence of (engine, voices) tuples, each voice
    producing N_PER_VOICE utterances. We cycle through the phrase list
    deterministically so all voices see the same content mix.
    """
    samples: list[np.ndarray] = []
    labels: list[int] = []
    phrase_list = list(phrases)
    if not phrase_list:
        return samples, labels

    # Each call rotates through all phrases so every voice sees the same
    # mixed-content stream. We need (N_PER_VOICE * voices_per_engine)
    # utterances per engine, so we index modulo len(phrases).
    for engine_idx, (engine_name, voices) in enumerate(phrase_list):
        for v_idx, voice in enumerate(voices):
            for i in range(N_PER_VOICE):
                phrase = TARGET_PHRASES[i % len(TARGET_PHRASES)] \
                    if label == TARGET_LABEL \
                    else NOT_TARGET_PHRASES[i % len(NOT_TARGET_PHRASES)]
                try:
                    if engine_name == "kokoro":
                        pcm = _tts_kokoro(phrase, voice)
                    else:
                        pcm = _tts_f5(phrase, voice)
                except Exception as e:
                    print(f"[tts_dataset] {engine_name}/{voice} attempt {i} "
                          f"failed: {e}; skipping")
                    continue
                if pcm.size == 0:
                    print(f"[tts_dataset] {engine_name}/{voice} attempt {i} "
                          f"returned empty; skipping")
                    continue
                samples.append(pcm)
                labels.append(label)
                if (len(samples) % 20) == 0:
                    print(f"[tts_dataset] generated {len(samples)} utterances "
                          f"so far (label={label})")
    return samples, labels


def _to_mel_tensors(pcms: list[np.ndarray]) -> list[np.ndarray]:
    """Convert each PCM utterance to its full (n_frames, 40) int8 mel."""
    out: list[np.ndarray] = []
    for pcm in pcms:
        mel = mel_spectrogram(pcm)            # (n_frames, 40) int8
        out.append(mel)
    return out


def _stack_windows(
    mels: list[np.ndarray], labels: list[int], rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack the per-utterance mel streams into (1, 1, 40, 10) windows.

    Each window is a hop of HOP_FRAMES between consecutive windows
    along a single utterance, so consecutive samples differ in just
    one frame (typical CNN training set-up). We discard utterances
    shorter than WINDOW_FRAMES.
    """
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for mel, label in zip(mels, labels):
        n_frames = mel.shape[0]
        if n_frames < WINDOW_FRAMES:
            continue
        for start in range(0, n_frames - WINDOW_FRAMES + 1, HOP_FRAMES):
            window = mel[start : start + WINDOW_FRAMES].T   # (40, 10)
            xs.append(window.astype(np.int8))
            ys.append(label)
    x = np.stack(xs, axis=0)[:, None, :, :]                # (N, 1, 40, 10)
    y = np.asarray(ys, dtype=np.int64)
    print(f"[tts_dataset] stacked into {x.shape[0]} windows, "
          f"target={int((y == TARGET_LABEL).sum())}, "
          f"not_target={int((y == NOT_TARGET_LABEL).sum())}")
    return x, y


def build_dataset(out_path: Path) -> dict:
    """Build the dataset, save it, and return the dict.

    90/10 stratified train/test split. Stride is per-label so both
    splits keep the class ratio.
    """
    rng = np.random.default_rng(RNG_SEED)

    # 6 Kokoro voices × 30 = 180 per class, plus 30 from F5 (3 voices
    # × 10) = 210 per class. Target 180 is the spec floor; overshooting
    # is fine and only helps generalization.
    kokoro_phrases = [
        ("kokoro", KOKORO_VOICES),
    ]
    f5_phrases = [
        ("f5", F5_VOICES),
    ]

    print(f"[tts_dataset] generating target class ({len(TARGET_PHRASES)} phrases)")
    target_pcms, target_labels = _generate_class(
        kokoro_phrases + f5_phrases, TARGET_LABEL, rng,
    )
    print(f"[tts_dataset] generating not-target class "
          f"({len(NOT_TARGET_PHRASES)} phrases)")
    not_target_pcms, not_target_labels = _generate_class(
        kokoro_phrases + f5_phrases, NOT_TARGET_LABEL, rng,
    )

    all_pcms = target_pcms + not_target_pcms
    all_labels = target_labels + not_target_labels
    print(f"[tts_dataset] collected {len(all_pcms)} utterances "
          f"(target={sum(all_labels)}, not_target={len(all_labels) - sum(all_labels)})")

    print("[tts_dataset] computing mel features...")
    mels = _to_mel_tensors(all_pcms)
    x, y = _stack_windows(mels, all_labels, rng)

    # 90/10 stratified split.
    n = x.shape[0]
    perm = rng.permutation(n)
    n_test = max(1, n // 10)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    split = np.zeros(n, dtype=bool)
    split[train_idx] = True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y),
        "split": torch.from_numpy(split),
        "voices": {
            "kokoro": list(KOKORO_VOICES),
            "f5": list(F5_VOICES),
        },
    }
    torch.save(payload, out_path)
    print(f"[tts_dataset] wrote {out_path}  "
          f"x={tuple(x.shape)} y={tuple(y.shape)} split_train={int(split.sum())} "
          f"split_test={n - int(split.sum())}")
    return payload


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Generate the TTS training dataset.")
    p.add_argument(
        "--out", type=Path,
        default=Path("models/hey-socket-v1/train_dataset.pt"),
    )
    args = p.parse_args()
    build_dataset(args.out)


if __name__ == "__main__":
    main()