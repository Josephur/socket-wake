# SPDX-License-Identifier: Apache-2.0
"""Build the v2 "hey socket" dataset: adds real-noise negatives and
hard-negative speech on top of the v1 TTS-only dataset.

See docs/training.md ("Reproducing the v2 (noise-augmented) dataset") for
the full recipe this implements. Summary of the four buckets combined here:

    1. Positives (clean)      - resynthesized "hey socket" TTS clips.
    2. Positives (SNR-mixed)  - each positive clip mixed with a random
                                 MUSAN noise clip at a random SNR in
                                 [5, 30] dB, 2 copies per positive.
    3. Negatives (TTS)        - the v1 dataset's existing negative windows
                                 (already-mel'd, reused as-is) PLUS ~150 new
                                 hard-negative TTS phrases (phonetic near
                                 misses + unrelated sentences).
    4. Negatives (noise)      - raw MUSAN noise clips, no speech at all.

v1's train_dataset.pt only stores windowed INT8 mel tensors, not raw PCM,
so positives must be resynthesized via TTS to get waveform-level audio for
SNR-mixing (this doubles as a refresh of the positive class with a
different RNG draw). Negatives are reused directly from v1 where possible
to save TTS calls and network time.
"""

from __future__ import annotations

import concurrent.futures as cf
import time
from pathlib import Path

import numpy as np
import torch

from socket_wake.data.tts_dataset import (
    F5_VOICES,
    HOP_FRAMES,
    KOKORO_VOICES,
    WINDOW_FRAMES,
    _tts_f5,
    _tts_kokoro,
)
from socket_wake.data.snr_mix import mix_at_snr
from socket_wake.features.mel import mel_spectrogram

TARGET_LABEL = 1
NOT_TARGET_LABEL = 0
RNG_SEED = 2026  # distinct from v1's seed=0, documented in training.md

POS_PHRASE = "hey socket"
N_POS_PER_VOICE = 10   # 9 voices (6 kokoro + 3 f5) * 10 = 90 base positives
SNR_COPIES_PER_POS = 2
SNR_RANGE_DB = (5.0, 30.0)

# MUSAN clips run tens of seconds; dense hop=1 windowing over the full clip
# produces thousands of near-duplicate windows per clip (observed: ~2400
# windows/clip on average, 2.2M total across 930 clips) which swamps the
# other buckets and collapses class balance to ~86:1 negative:positive.
# Use a coarse, non-overlapping-ish hop and cap windows per clip so noise
# contributes a comparable order of magnitude to the other negative buckets.
NOISE_HOP_FRAMES = WINDOW_FRAMES  # non-overlapping windows
NOISE_WINDOWS_PER_CLIP_CAP = 30

V1_DATASET = Path("models/hey-socket-v1/train_dataset.pt")
NOISE_DIR = Path("models/hey-socket-v1/noise_clips")
OUT_PATH = Path("models/hey-socket-v1/train_dataset_v2.pt")

MAX_WORKERS = 12

# ---------------------------------------------------------------------------
# Hard-negative phrases: phonetic near-misses to "hey socket" (~75) and
# unrelated everyday sentences (~75). See docs/training.md for rationale.
# ---------------------------------------------------------------------------

HARD_NEG_PHONETIC: tuple[str, ...] = (
    "hey rocket", "hey pocket", "hey wallet", "hey doctor", "okay socket",
    "hey soccer", "hey socks it", "hay socket", "hey sockets",
    "hey socket please", "hey sock it", "hey saw kit", "hey soc-ket",
    "hey sockit", "hey sarcket", "hey socked", "hey soccer ket",
    "hey soccket", "hey suck it", "hey soft it", "hey soggy",
    "hey solvent", "hey socket on", "hey rockets", "hey pockets",
    "hey pocket it", "a socket", "hey locket", "hey jacket", "hey blanket",
    "hey basket", "hey packet", "hey ticket", "hey socket now",
    "hey sprocket", "hey socket off", "hi socket", "hey socket hey",
    "hey soquette", "hey soquet", "hey soquit", "hey saucer",
    "hey soggier", "hey soccket please", "hey soc it", "okay rocket",
    "okay pocket", "okay wallet", "hey walla", "hey wall it", "hey talk it",
    "hey stock it", "hey stopwatch", "hey stock pit", "hey stopgap",
    "hey shocked", "hey shock it", "hey shocking", "hey soft kit",
    "hey sock hit", "hey saw kit please", "hey soft cap", "hey saw cat",
    "hey soft cat", "hey sock cap", "hey soft cot", "hey soc cot",
    "hey soc pot", "hey sock pot", "hey soggy pot", "hey rock hit",
    "hey rock it", "hey pock it", "hey lock it", "hey dock it",
)

HARD_NEG_UNRELATED: tuple[str, ...] = (
    "what's the weather like today", "turn off the lights",
    "play some music", "how do I get to the store", "what time is it",
    "set a timer for ten minutes", "remind me to call mom",
    "is it going to rain tomorrow", "what's on my calendar today",
    "add milk to the shopping list", "tell me a joke",
    "how far is the airport", "what's the capital of france",
    "read me the news", "turn up the volume", "skip this song",
    "pause the video", "what's the score of the game",
    "send a text to john", "call the office", "lock the front door",
    "open the garage", "what's the traffic like right now",
    "give me directions home", "how many calories are in an apple",
    "convert five miles to kilometers", "what's the best pizza place nearby",
    "schedule a meeting for tomorrow", "what's my battery percentage",
    "increase the thermostat", "play my favorite playlist",
    "what's the definition of gravity", "how tall is mount everest",
    "translate hello into spanish", "set an alarm for seven am",
    "what day is it today", "who won the election",
    "what's the exchange rate for euros", "how do I make pancakes",
    "what's a good recipe for dinner", "turn on the fan",
    "dim the bedroom lights", "what's the stock price of apple",
    "play the news briefing", "how long is the flight to london",
    "what's my next appointment", "remind me to take out the trash",
    "what's the square root of 144", "how many ounces in a gallon",
    "what's the population of tokyo", "play a podcast",
    "show me the latest headlines", "what's my step count today",
    "how do I reset my password", "is the store open right now",
    "what movies are playing tonight", "how long does it take to boil an egg",
    "what's the closest gas station", "play some jazz",
    "skip to the next track", "lower the volume",
    "what's the humidity outside", "cancel my alarm",
    "what's the wifi password", "connect to bluetooth",
    "how much battery is left", "what's a good book to read",
    "tell me about the weather this weekend", "how do I spell necessary",
    "what's the tallest building in the world", "play some classical music",
    "remind me about the dentist appointment", "what's the distance to the moon",
    "how do plants make food", "what's the meaning of life",
)

HARD_NEG_PHRASES: tuple[str, ...] = HARD_NEG_PHONETIC + HARD_NEG_UNRELATED

_VOICE_COMBOS: tuple[tuple[str, str], ...] = tuple(
    [("kokoro", v) for v in KOKORO_VOICES] + [("f5", v) for v in F5_VOICES]
)


def _tts_one(engine: str, voice: str, phrase: str) -> np.ndarray | None:
    try:
        if engine == "kokoro":
            return _tts_kokoro(phrase, voice)
        return _tts_f5(phrase, voice)
    except Exception as e:
        print(f"[build_v2] {engine}/{voice} '{phrase[:30]}' failed: {e}")
        return None


def _generate_positive_pcms() -> list[np.ndarray]:
    jobs: list[tuple[str, str, str]] = []
    for engine, voice in _VOICE_COMBOS:
        for _ in range(N_POS_PER_VOICE):
            jobs.append((engine, voice, POS_PHRASE))

    pcms: list[np.ndarray] = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(_tts_one, e, v, p) for e, v, p in jobs]
        for i, fut in enumerate(futs):
            res = fut.result()
            if res is not None and res.size > 0:
                pcms.append(res)
            if (i + 1) % 20 == 0:
                print(f"[build_v2] positives: {i + 1}/{len(jobs)} requested, "
                      f"{len(pcms)} succeeded")
    print(f"[build_v2] generated {len(pcms)} positive PCM clips")
    return pcms


def _generate_hard_negative_pcms() -> list[np.ndarray]:
    jobs: list[tuple[str, str, str]] = []
    for i, phrase in enumerate(HARD_NEG_PHRASES):
        engine, voice = _VOICE_COMBOS[i % len(_VOICE_COMBOS)]
        jobs.append((engine, voice, phrase))

    pcms: list[np.ndarray] = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(_tts_one, e, v, p) for e, v, p in jobs]
        for i, fut in enumerate(futs):
            res = fut.result()
            if res is not None and res.size > 0:
                pcms.append(res)
            if (i + 1) % 20 == 0:
                print(f"[build_v2] hard negatives: {i + 1}/{len(jobs)} requested, "
                      f"{len(pcms)} succeeded")
    print(f"[build_v2] generated {len(pcms)} hard-negative PCM clips")
    return pcms


def _load_noise_clips() -> list[np.ndarray]:
    """Load 16 kHz mono int16 noise WAVs already saved to NOISE_DIR."""
    import wave

    clips: list[np.ndarray] = []
    for wav_path in sorted(NOISE_DIR.glob("*.wav")):
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
            raw = wf.readframes(wf.getnframes())
        clips.append(np.frombuffer(raw, dtype="<i2"))
    print(f"[build_v2] loaded {len(clips)} noise clips from {NOISE_DIR}")
    return clips


def _to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), -32768, 32767).astype(np.int16)


def _windows_from_pcm(
    pcm: np.ndarray, label: int, hop: int = HOP_FRAMES, cap: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[list[np.ndarray], list[int]]:
    """Stack an utterance's mel frames into WINDOW_FRAMES-wide windows.

    ``hop`` controls window stride (HOP_FRAMES=1 for short TTS utterances,
    matching v1's dense overlap). ``cap`` bounds how many windows a single
    clip may contribute -- needed for MUSAN noise clips, which run tens of
    seconds and would otherwise produce thousands of near-duplicate windows
    each, swamping the other buckets and wrecking class balance. When the
    raw window count exceeds ``cap``, we randomly subsample.
    """
    mel = mel_spectrogram(pcm)
    n_frames = mel.shape[0]
    xs: list[np.ndarray] = []
    ys: list[int] = []
    if n_frames < WINDOW_FRAMES:
        return xs, ys
    starts = list(range(0, n_frames - WINDOW_FRAMES + 1, hop))
    if cap is not None and len(starts) > cap:
        rng = rng or np.random.default_rng()
        starts = list(rng.choice(starts, size=cap, replace=False))
    for start in starts:
        window = mel[start : start + WINDOW_FRAMES].T  # (40, 10)
        xs.append(window.astype(np.int8))
        ys.append(label)
    return xs, ys


def build_dataset(out_path: Path = OUT_PATH) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)

    # --- Bucket 4: noise clips (loaded from disk; see fetch_musan.py) -----
    noise_clips = _load_noise_clips()
    if not noise_clips:
        raise RuntimeError(
            f"No noise clips found in {NOISE_DIR}; run fetch_musan.py first"
        )

    # --- Bucket 1+2: positives, clean + SNR-mixed --------------------------
    print("[build_v2] generating positive TTS clips...")
    pos_pcms = _generate_positive_pcms()

    all_xs: list[np.ndarray] = []
    all_ys: list[int] = []

    n_pos_clean_windows = 0
    n_pos_mixed_windows = 0
    for pcm in pos_pcms:
        xs, ys = _windows_from_pcm(pcm, TARGET_LABEL)
        all_xs.extend(xs)
        all_ys.extend(ys)
        n_pos_clean_windows += len(xs)

        for _ in range(SNR_COPIES_PER_POS):
            noise = noise_clips[rng.integers(0, len(noise_clips))]
            snr_db = float(rng.uniform(*SNR_RANGE_DB))
            mixed = _to_int16(mix_at_snr(pcm, noise, snr_db, rng))
            xs_m, ys_m = _windows_from_pcm(mixed, TARGET_LABEL)
            all_xs.extend(xs_m)
            all_ys.extend(ys_m)
            n_pos_mixed_windows += len(xs_m)
    print(f"[build_v2] positive windows: clean={n_pos_clean_windows} "
          f"snr_mixed={n_pos_mixed_windows}")

    # --- Bucket 3a: reuse v1's existing negative windows -------------------
    v1 = torch.load(V1_DATASET)
    v1_x, v1_y = v1["x"].numpy(), v1["y"].numpy()
    v1_neg_mask = v1_y == NOT_TARGET_LABEL
    n_v1_neg = int(v1_neg_mask.sum())
    for row in v1_x[v1_neg_mask]:
        all_xs.append(row[0])  # drop the leading channel dim, re-added later
        all_ys.append(NOT_TARGET_LABEL)
    print(f"[build_v2] reused {n_v1_neg} negative windows from v1 dataset")

    # --- Bucket 3b: new hard-negative TTS phrases ---------------------------
    print("[build_v2] generating hard-negative TTS clips...")
    hard_neg_pcms = _generate_hard_negative_pcms()
    n_hard_neg_windows = 0
    for pcm in hard_neg_pcms:
        xs, ys = _windows_from_pcm(pcm, NOT_TARGET_LABEL)
        all_xs.extend(xs)
        all_ys.extend(ys)
        n_hard_neg_windows += len(xs)
    print(f"[build_v2] hard-negative windows: {n_hard_neg_windows}")

    # --- Bucket 4b: raw noise clips as pure negatives -----------------------
    n_noise_windows = 0
    for clip in noise_clips:
        xs, ys = _windows_from_pcm(
            clip, NOT_TARGET_LABEL,
            hop=NOISE_HOP_FRAMES, cap=NOISE_WINDOWS_PER_CLIP_CAP, rng=rng,
        )
        all_xs.extend(xs)
        all_ys.extend(ys)
        n_noise_windows += len(xs)
    print(f"[build_v2] pure-noise windows: {n_noise_windows}")

    x = np.stack(all_xs, axis=0)[:, None, :, :]  # (N, 1, 40, 10)
    y = np.asarray(all_ys, dtype=np.int64)
    print(f"[build_v2] total windows: {x.shape[0]} "
          f"(target={int((y == TARGET_LABEL).sum())}, "
          f"not_target={int((y == NOT_TARGET_LABEL).sum())})")

    # 90/10 stratified split.
    n = x.shape[0]
    split = np.zeros(n, dtype=bool)
    for label in (TARGET_LABEL, NOT_TARGET_LABEL):
        idx = np.nonzero(y == label)[0]
        perm = rng.permutation(idx)
        n_test = max(1, len(perm) // 10)
        split[perm[n_test:]] = True  # train
        # perm[:n_test] stays False -> test

    sources = {
        "pos_clean_windows": n_pos_clean_windows,
        "pos_snr_mixed_windows": n_pos_mixed_windows,
        "neg_v1_reused_windows": n_v1_neg,
        "neg_hard_negative_windows": n_hard_neg_windows,
        "neg_pure_noise_windows": n_noise_windows,
        "n_positive_base_clips": len(pos_pcms),
        "n_hard_negative_clips": len(hard_neg_pcms),
        "n_noise_clips": len(noise_clips),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y),
        "split": torch.from_numpy(split),
        "voices": {
            "kokoro": list(KOKORO_VOICES),
            "f5": list(F5_VOICES),
        },
        "sources": sources,
    }
    torch.save(payload, out_path)
    elapsed = time.time() - t0
    print(f"[build_v2] wrote {out_path} x={tuple(x.shape)} y={tuple(y.shape)} "
          f"split_train={int(split.sum())} split_test={n - int(split.sum())} "
          f"elapsed={elapsed:.1f}s")
    print(f"[build_v2] sources: {sources}")
    return payload


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Generate the v2 noise-augmented dataset.")
    p.add_argument("--out", type=Path, default=OUT_PATH)
    args = p.parse_args()
    build_dataset(args.out)


if __name__ == "__main__":
    main()
