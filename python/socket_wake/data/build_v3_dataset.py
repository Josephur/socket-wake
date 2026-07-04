# SPDX-License-Identifier: Apache-2.0
"""Build the v3 "hey socket" dataset: standard-KWS windowing and labeling.

v3 exists because v2, while it fixed the *data sources* (real noise, hard
negatives, SNR mixing), still deviated from standard keyword-spotting
methodology in four ways that made its models untrustworthy:

1. **100 ms windows.** v2's WINDOW_FRAMES=10 (100 ms) cannot contain the
   ~600-900 ms phrase "hey socket"; the model could only learn "sounds like
   a fragment of the phrase," which caps precision. Standard KWS (Google
   Speech Commands, microWakeWord) uses ~1 s windows. v3 uses
   WINDOW_FRAMES=98 = exactly 1.0 s at the runtime's 30 ms window / 10 ms
   hop mel frontend.
2. **Every sliding window of a positive clip was labeled positive** —
   including edge windows that are mostly silence. v3 trims silence and
   places the *whole* utterance inside the 1 s window at jittered offsets
   (the standard "time-shift augmentation"), so label 1 always means "this
   window contains the complete wake phrase."
3. **Window-level train/test split.** v2 split after windowing, so
   near-duplicate overlapping windows of the same clip landed in both
   train and test, inflating held-out metrics. v3 splits at the *clip*
   level: all windows derived from one source clip share a split.
4. **No PCM cache.** v2 re-synthesized TTS on every build (and identical
   requests may or may not return identical audio). v3 caches decoded
   16 kHz PCM to disk (`tts_cache/`), dedupes clips by content hash, and
   records clip IDs in the dataset so the streaming evaluator can rebuild
   audio streams from the exact held-out clips.

Buckets (same recipe as v2, re-windowed):
    positives:  clean + SNR-mixed [5,30] dB, jittered placement
    negatives:  hard-negative TTS speech (sliding 1 s windows),
                MUSAN noise (non-overlapping windows, per-clip cap),
                silence / near-silence windows

Output: models/hey-socket-v1/train_dataset_v3.pt with keys
    x: (N, 1, 40, 98) int8    y: (N,) int64    split: (N,) bool (True=train)
    clip_id: (N,) int32       clips: {clip_id: {"file": ..., "kind": ...,
                                                "train": bool}}
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import time
import wave
from pathlib import Path

import numpy as np
import torch

from socket_wake.data.build_v2_dataset import (
    HARD_NEG_PHRASES,
    _to_int16,
)
from socket_wake.data.snr_mix import mix_at_snr
from socket_wake.data.tts_dataset import (
    F5_URL,
    KOKORO_URL,
    KOKORO_VOICES,
    _decode_to_16k_pcm,
    _http_post,
)
from socket_wake.features.mel import HOP, N_MELS, WINDOW, mel_spectrogram

TARGET_LABEL = 1
NOT_TARGET_LABEL = 0
RNG_SEED = 3026

# 98 frames = 480 + 97*160 = 16000 samples = exactly 1.0 s @ 16 kHz.
WINDOW_FRAMES = 98
WINDOW_SAMPLES = WINDOW + (WINDOW_FRAMES - 1) * HOP
assert WINDOW_SAMPLES == 16_000

POS_PHRASE_VARIANTS = ("hey socket", "hey socket.", "Hey socket!")
KOKORO_SPEEDS = (0.85, 1.0, 1.15)
N_F5_POS = 12                    # F5 requests per phrase variant (voice clone)
POS_JITTERS = 3                  # placements of each positive per window
SNR_COPIES_PER_PLACEMENT = 2
SNR_RANGE_DB = (5.0, 30.0)

NEG_SPEECH_HOP_FRAMES = 49       # 0.5 s hop over hard-negative speech
NOISE_HOP_FRAMES = WINDOW_FRAMES # non-overlapping windows over noise
NOISE_WINDOWS_PER_CLIP_CAP = 8
N_SILENCE_WINDOWS = 300

MODEL_DIR = Path("models/hey-socket-v1")
NOISE_DIR = MODEL_DIR / "noise_clips"
TTS_CACHE = MODEL_DIR / "tts_cache"
OUT_PATH = MODEL_DIR / "train_dataset_v3.pt"

MAX_WORKERS = 8                  # F5 timed out under 12 in the v2 build

TEST_FRACTION = 0.1


# ---------------------------------------------------------------------------
# TTS with on-disk PCM cache
# ---------------------------------------------------------------------------

def _cache_key(engine: str, voice: str, speed: float, phrase: str, salt: str = "") -> str:
    raw = f"{engine}|{voice}|{speed}|{phrase}|{salt}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _write_wav(path: Path, pcm: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16_000)
        wf.writeframes(pcm.astype("<i2").tobytes())


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == 16_000 and wf.getnchannels() == 1
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")


def _tts_cached(engine: str, voice: str, speed: float, phrase: str,
                salt: str = "") -> np.ndarray | None:
    """Synthesize (or load cached) a clip; returns 16 kHz int16 PCM."""
    key = _cache_key(engine, voice, speed, phrase, salt)
    path = TTS_CACHE / f"{engine}_{key}.wav"
    if path.exists():
        return read_wav(path)
    try:
        if engine == "kokoro":
            raw = _http_post(
                f"{KOKORO_URL}/v1/audio/speech",
                {"model": "kokoro", "input": phrase, "voice": voice,
                 "speed": speed},
                timeout=60.0,
            )
            ext = "mp3" if (raw[:3] == b"ID3" or raw[:2] in (b"\xff\xfb", b"\xff\xf3")) else "wav"
            pcm = _decode_to_16k_pcm(raw, ext)
        else:
            raw = _http_post(f"{F5_URL}/tts", {"text": phrase, "voice": voice},
                             timeout=90.0)
            pcm = _decode_to_16k_pcm(raw, "wav")
    except Exception as e:
        print(f"[build_v3] {engine}/{voice} '{phrase[:30]}' failed: {e}")
        return None
    if pcm.size == 0:
        return None
    _write_wav(path, pcm)
    return pcm


def _fetch_all(jobs: list[tuple[str, str, float, str, str]], desc: str) -> list[tuple[str, np.ndarray]]:
    """Run TTS jobs in parallel; dedupe by decoded-PCM hash.

    Returns (cache_file_name, pcm) pairs for unique clips only.
    """
    results: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_tts_cached, *j): j for j in jobs}
        for i, fut in enumerate(cf.as_completed(futs)):
            engine, voice, speed, phrase, salt = futs[fut]
            pcm = fut.result()
            if pcm is None:
                continue
            digest = hashlib.sha1(pcm.tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            key = _cache_key(engine, voice, speed, phrase, salt)
            results.append((f"{engine}_{key}.wav", pcm))
            if (i + 1) % 25 == 0:
                print(f"[build_v3] {desc}: {i + 1}/{len(jobs)} done, "
                      f"{len(results)} unique")
    print(f"[build_v3] {desc}: {len(results)} unique clips "
          f"({len(jobs)} requested)")
    return results


# ---------------------------------------------------------------------------
# Silence trimming + window placement
# ---------------------------------------------------------------------------

def trim_active(pcm: np.ndarray, thresh_frac: float = 0.08,
                margin: int = 800) -> np.ndarray:
    """Trim leading/trailing silence via a moving-RMS envelope."""
    x = pcm.astype(np.float32)
    win = 400
    if x.size < win:
        return pcm
    e = np.sqrt(np.convolve(x * x, np.ones(win) / win, mode="same"))
    active = np.nonzero(e > e.max() * thresh_frac)[0]
    if active.size == 0:
        return pcm
    lo = max(0, int(active[0]) - margin)
    hi = min(pcm.size, int(active[-1]) + margin)
    return pcm[lo:hi]


def place_in_window(pcm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Place a (trimmed) utterance whole inside a 1 s window.

    Longer-than-window utterances are center-cropped (rare for the wake
    phrase; logged by the caller via length checks).
    """
    if pcm.size >= WINDOW_SAMPLES:
        start = (pcm.size - WINDOW_SAMPLES) // 2
        return pcm[start : start + WINDOW_SAMPLES].copy()
    out = np.zeros(WINDOW_SAMPLES, dtype=pcm.dtype)
    offset = int(rng.integers(0, WINDOW_SAMPLES - pcm.size + 1))
    out[offset : offset + pcm.size] = pcm
    return out


def window_mel(pcm_1s: np.ndarray) -> np.ndarray:
    """Mel of exactly one window: (40, 98) int8."""
    mel = mel_spectrogram(pcm_1s.astype(np.int16))
    assert mel.shape[0] >= WINDOW_FRAMES, mel.shape
    return mel[:WINDOW_FRAMES].T.astype(np.int8)


def sliding_windows(pcm: np.ndarray, hop_frames: int,
                    cap: int | None = None,
                    rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """All (40, 98) windows of a longer clip at the given frame hop."""
    mel = mel_spectrogram(pcm.astype(np.int16))
    n = mel.shape[0]
    if n < WINDOW_FRAMES:
        return []
    starts = list(range(0, n - WINDOW_FRAMES + 1, hop_frames))
    if cap is not None and len(starts) > cap:
        rng = rng or np.random.default_rng()
        starts = list(rng.choice(starts, size=cap, replace=False))
    return [mel[s : s + WINDOW_FRAMES].T.astype(np.int8) for s in starts]


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def _load_noise_clips() -> list[tuple[str, np.ndarray]]:
    clips = []
    for p in sorted(NOISE_DIR.glob("*.wav")):
        clips.append((p.name, read_wav(p)))
    print(f"[build_v3] loaded {len(clips)} noise clips")
    return clips


def build_dataset(out_path: Path = OUT_PATH) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)
    TTS_CACHE.mkdir(parents=True, exist_ok=True)

    noise_clips = _load_noise_clips()
    if not noise_clips:
        raise RuntimeError(f"no noise clips in {NOISE_DIR}; run fetch_musan first")

    # --- synthesize / load positives -----------------------------------
    pos_jobs: list[tuple[str, str, float, str, str]] = []
    for voice in KOKORO_VOICES:
        for speed in KOKORO_SPEEDS:
            for phrase in POS_PHRASE_VARIANTS:
                pos_jobs.append(("kokoro", voice, speed, phrase, ""))
    for phrase in POS_PHRASE_VARIANTS:
        for i in range(N_F5_POS):
            # F5 clones a voice per request; salt separates cache entries.
            pos_jobs.append(("f5", "default", 1.0, phrase, f"s{i}"))
    pos_clips = _fetch_all(pos_jobs, "positives")

    # --- synthesize / load hard negatives -------------------------------
    neg_jobs: list[tuple[str, str, float, str, str]] = []
    for i, phrase in enumerate(HARD_NEG_PHRASES):
        voice = KOKORO_VOICES[i % len(KOKORO_VOICES)]
        speed = KOKORO_SPEEDS[i % len(KOKORO_SPEEDS)]
        neg_jobs.append(("kokoro", voice, speed, phrase, ""))
        if i % 5 == 0:  # a fifth of hard negatives also via F5
            neg_jobs.append(("f5", "default", 1.0, phrase, "s0"))
    neg_clips = _fetch_all(neg_jobs, "hard negatives")

    # --- clip registry + clip-level split -------------------------------
    # clip kinds: pos_tts, neg_tts, noise, silence (silence has no file)
    clips: dict[int, dict] = {}
    next_id = 0

    def register(file: str | None, kind: str) -> int:
        nonlocal next_id
        cid = next_id
        next_id += 1
        clips[cid] = {"file": file, "kind": kind, "train": True}
        return cid

    pos_ids = [register(f, "pos_tts") for f, _ in pos_clips]
    neg_ids = [register(f, "neg_tts") for f, _ in neg_clips]
    noise_ids = [register(f, "noise") for f, _ in noise_clips]

    # Clip-level 90/10 split, stratified per kind. Noise clips used to mix
    # into TRAIN positives must themselves be train clips (no leakage of
    # held-out noise into training inputs) -- handled below by drawing mix
    # noise from the train-noise pool only for train positives, and from
    # the test pool for test positives.
    for ids in (pos_ids, neg_ids, noise_ids):
        perm = rng.permutation(len(ids))
        n_test = max(1, int(len(ids) * TEST_FRACTION))
        for j in perm[:n_test]:
            clips[ids[j]]["train"] = False

    noise_train = [(cid, pcm) for cid, (_, pcm) in zip(noise_ids, noise_clips)
                   if clips[cid]["train"]]
    noise_test = [(cid, pcm) for cid, (_, pcm) in zip(noise_ids, noise_clips)
                  if not clips[cid]["train"]]

    all_x: list[np.ndarray] = []
    all_y: list[int] = []
    all_cid: list[int] = []
    counts = {"pos_clean": 0, "pos_mixed": 0, "neg_speech": 0,
              "neg_noise": 0, "neg_silence": 0, "pos_cropped": 0}

    # --- positives: jittered placement + SNR mixing ----------------------
    for cid, (_, pcm) in zip(pos_ids, pos_clips):
        active = trim_active(pcm)
        if active.size >= WINDOW_SAMPLES:
            counts["pos_cropped"] += 1
        pool = noise_train if clips[cid]["train"] else noise_test
        for _ in range(POS_JITTERS):
            placed = place_in_window(active, rng)
            all_x.append(window_mel(placed))
            all_y.append(TARGET_LABEL)
            all_cid.append(cid)
            counts["pos_clean"] += 1
            for _ in range(SNR_COPIES_PER_PLACEMENT):
                _, noise = pool[int(rng.integers(0, len(pool)))]
                snr = float(rng.uniform(*SNR_RANGE_DB))
                mixed = _to_int16(mix_at_snr(placed, noise, snr, rng))
                all_x.append(window_mel(mixed))
                all_y.append(TARGET_LABEL)
                all_cid.append(cid)
                counts["pos_mixed"] += 1

    # --- negatives: hard-negative speech, sliding ------------------------
    for cid, (_, pcm) in zip(neg_ids, neg_clips):
        # Pad short clips so even sub-1s phrases yield one window.
        if pcm.size < WINDOW_SAMPLES:
            pcm = np.pad(pcm, (0, WINDOW_SAMPLES - pcm.size))
        for w in sliding_windows(pcm, NEG_SPEECH_HOP_FRAMES):
            all_x.append(w)
            all_y.append(NOT_TARGET_LABEL)
            all_cid.append(cid)
            counts["neg_speech"] += 1

    # --- negatives: noise, non-overlapping + cap -------------------------
    for cid, (_, pcm) in zip(noise_ids, noise_clips):
        for w in sliding_windows(pcm, NOISE_HOP_FRAMES,
                                 cap=NOISE_WINDOWS_PER_CLIP_CAP, rng=rng):
            all_x.append(w)
            all_y.append(NOT_TARGET_LABEL)
            all_cid.append(cid)
            counts["neg_noise"] += 1

    # --- negatives: silence / near-silence -------------------------------
    silence_cid = register(None, "silence")
    for i in range(N_SILENCE_WINDOWS):
        amp = float(rng.uniform(0, 60))  # up to ~ -55 dBFS hiss
        pcm = (rng.standard_normal(WINDOW_SAMPLES) * amp).astype(np.int16)
        all_x.append(window_mel(pcm))
        all_y.append(NOT_TARGET_LABEL)
        all_cid.append(silence_cid)
        counts["neg_silence"] += 1

    x = np.stack(all_x)[:, None, :, :]           # (N, 1, 40, 98)
    y = np.asarray(all_y, dtype=np.int64)
    cid = np.asarray(all_cid, dtype=np.int32)
    split = np.asarray([clips[c]["train"] for c in all_cid], dtype=bool)

    n_pos = int((y == TARGET_LABEL).sum())
    n_neg = int((y == NOT_TARGET_LABEL).sum())
    print(f"[build_v3] windows: {x.shape[0]} (pos={n_pos}, neg={n_neg}, "
          f"ratio {n_neg / max(1, n_pos):.1f}:1)")
    print(f"[build_v3] sources: {counts}")
    print(f"[build_v3] split: train={int(split.sum())} test={int((~split).sum())} "
          f"(clip-level, {len(clips)} clips)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y),
        "split": torch.from_numpy(split),
        "clip_id": torch.from_numpy(cid),
        "clips": clips,
        "window_frames": WINDOW_FRAMES,
        "counts": counts,
    }
    torch.save(payload, out_path)
    print(f"[build_v3] wrote {out_path} in {time.time() - t0:.1f}s")
    return payload


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Build the v3 (1 s window) dataset.")
    p.add_argument("--out", type=Path, default=OUT_PATH)
    args = p.parse_args()
    build_dataset(args.out)


if __name__ == "__main__":
    main()
