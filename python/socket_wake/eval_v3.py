# SPDX-License-Identifier: Apache-2.0
"""Streaming FAR/FRR benchmark for the v3 model -- the real acceptance gate.

Per-window classifier accuracy is NOT the deployment metric. This script
simulates exactly what the device does: slide a 1 s window over a
continuous mel stream at a fixed inference cadence, threshold the
target-class probability, and require K consecutive above-threshold
inferences before firing (then hold a refractory period). It reports:

  - false accepts per hour on held-out MUSAN noise (streamed)
  - false accepts per hour on held-out hard-negative speech (streamed)
  - per-utterance recall on held-out positives embedded in held-out
    noise at 20 / 10 / 5 dB SNR

for both the float (BN-folded) model and a post-training INT8 simulation,
across a threshold sweep. Design bar (DESIGN.md): <= 1 false accept/hour
at usefully high recall.

All held-out audio comes from clips whose *entire* window set was held out
of training (clip-level split; see build_v3_dataset.py).

Usage:
    python -m socket_wake.eval_v3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from socket_wake.data.build_v3_dataset import (
    MODEL_DIR,
    NOISE_DIR,
    TTS_CACHE,
    WINDOW_FRAMES,
    WINDOW_SAMPLES,
    place_in_window,
    read_wav,
    trim_active,
)
from socket_wake.data.snr_mix import mix_at_snr
from socket_wake.data.build_v2_dataset import _to_int16
from socket_wake.features.mel import mel_spectrogram
from socket_wake.model.kws_cnn import (
    Int8Sim,
    KWSConvNet,
    count_macs,
    count_params,
    fold_batchnorm,
)

CADENCE_FRAMES = 3        # one inference per 30 ms (microWakeWord cadence)
CONSECUTIVE = 2           # inferences >= threshold required to fire
REFRACTORY_FRAMES = 98    # 1 s lockout after a fire
THRESHOLDS = (0.5, 0.9, 0.95, 0.99, 0.995, 0.999)
SNRS_DB = (20.0, 10.0, 5.0)
RNG_SEED = 4026


def _clip_path(info: dict) -> Path | None:
    if info["file"] is None:
        return None
    for base in (TTS_CACHE, NOISE_DIR):
        p = base / info["file"]
        if p.exists():
            return p
    return None


def load_test_clips(payload: dict) -> dict[str, list[np.ndarray]]:
    """Held-out clips by kind, loaded from the PCM caches."""
    out: dict[str, list[np.ndarray]] = {"pos_tts": [], "neg_tts": [], "noise": []}
    missing = 0
    for cid, info in payload["clips"].items():
        if info["train"] or info["kind"] not in out:
            continue
        p = _clip_path(info)
        if p is None:
            missing += 1
            continue
        out[info["kind"]].append(read_wav(p))
    if missing:
        print(f"[eval_v3] WARNING: {missing} held-out clips missing from cache")
    for k, v in out.items():
        dur = sum(c.size for c in v) / 16_000.0
        print(f"[eval_v3] test {k}: {len(v)} clips, {dur:.1f}s audio")
    return out


def stream_scores(pcm: np.ndarray, score_fn) -> np.ndarray:
    """Score a continuous PCM stream at the inference cadence.

    Returns target-class probabilities, one per inference step.
    """
    mel = mel_spectrogram(pcm.astype(np.int16))       # (n_frames, 40)
    n = mel.shape[0]
    if n < WINDOW_FRAMES:
        return np.zeros(0, dtype=np.float32)
    starts = list(range(0, n - WINDOW_FRAMES + 1, CADENCE_FRAMES))
    wins = np.stack([mel[s : s + WINDOW_FRAMES].T for s in starts])
    x = torch.from_numpy(wins).float().unsqueeze(1) / 127.0
    probs: list[np.ndarray] = []
    for i in range(0, x.shape[0], 256):
        logits = score_fn(x[i : i + 256])
        probs.append(torch.softmax(logits, dim=1)[:, 1].numpy())
    return np.concatenate(probs)


def count_fires(probs: np.ndarray, thr: float) -> int:
    """Streaming detector: CONSECUTIVE hits >= thr fire, then refractory."""
    fires = 0
    consec = 0
    lockout = 0
    refractory_steps = REFRACTORY_FRAMES // CADENCE_FRAMES
    for p in probs:
        if lockout > 0:
            lockout -= 1
            continue
        if p >= thr:
            consec += 1
            if consec >= CONSECUTIVE:
                fires += 1
                consec = 0
                lockout = refractory_steps
        else:
            consec = 0
    return fires


def eval_model(name: str, score_fn, streams: dict, pos_trials: list[np.ndarray]) -> None:
    # Pre-compute scores once per stream; thresholding is cheap afterwards.
    noise_probs = stream_scores(streams["noise"], score_fn)
    speech_probs = stream_scores(streams["speech"], score_fn)
    noise_hours = streams["noise"].size / 16_000.0 / 3600.0
    speech_hours = streams["speech"].size / 16_000.0 / 3600.0
    trial_probs = [stream_scores(t, score_fn) for t in pos_trials]
    n_per_snr = len(pos_trials) // len(SNRS_DB)

    print(f"\n=== {name} ===")
    print(f"noise stream: {noise_hours * 60:.1f} min, "
          f"speech stream: {speech_hours * 60:.1f} min, "
          f"positive trials: {len(pos_trials)} ({n_per_snr}/SNR)")
    hdr = f"{'thr':>6} {'FA/h noise':>11} {'FA/h speech':>12}"
    for snr in SNRS_DB:
        hdr += f" {'rec@' + str(int(snr)) + 'dB':>9}"
    print(hdr)
    for thr in THRESHOLDS:
        fa_noise = count_fires(noise_probs, thr) / max(noise_hours, 1e-9)
        fa_speech = count_fires(speech_probs, thr) / max(speech_hours, 1e-9)
        row = f"{thr:>6} {fa_noise:>11.2f} {fa_speech:>12.2f}"
        for si in range(len(SNRS_DB)):
            hits = sum(
                1 for tp in trial_probs[si * n_per_snr : (si + 1) * n_per_snr]
                if count_fires(tp, thr) > 0
            )
            row += f" {hits / max(1, n_per_snr):>9.3f}"
        print(row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=MODEL_DIR / "train_dataset_v3.pt")
    p.add_argument("--ckpt", type=Path, default=MODEL_DIR / "checkpoint_v3.pt")
    args = p.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    payload = torch.load(args.data, weights_only=False)
    test = load_test_clips(payload)
    if not test["pos_tts"] or not test["noise"]:
        raise RuntimeError("held-out positives/noise missing; rebuild dataset")

    # Continuous negative streams.
    noise_stream = np.concatenate(test["noise"])
    gap = np.zeros(8_000, dtype=np.int16)  # 0.5 s between utterances
    speech_parts: list[np.ndarray] = []
    for c in test["neg_tts"]:
        speech_parts.extend((c, gap))
    speech_stream = np.concatenate(speech_parts) if speech_parts else gap

    # Positive trials: each held-out positive embedded mid-way in 3 s of
    # held-out noise at a controlled SNR. Ordered [all@20dB, all@10, all@5].
    pos_trials: list[np.ndarray] = []
    for snr in SNRS_DB:
        for clip in test["pos_tts"]:
            active = trim_active(clip)
            placed = place_in_window(active, rng)         # 1 s, phrase whole
            noise = test["noise"][int(rng.integers(0, len(test["noise"])))]
            bg = np.tile(noise, int(np.ceil(48_000 / noise.size)))[:48_000]
            trial = bg.astype(np.float32).copy()
            mixed = mix_at_snr(placed, bg[16_000:32_000], snr, rng)
            trial[16_000:32_000] = mixed
            pos_trials.append(_to_int16(trial))

    state = torch.load(args.ckpt, weights_only=False)
    model = KWSConvNet(n_classes=state.get("n_classes", 2))
    model.load_state_dict(state["model"])
    model.eval()
    folded = fold_batchnorm(model)

    # Sanity: folding must not change outputs (beyond fp error).
    with torch.no_grad():
        probe = torch.from_numpy(
            np.stack([mel_spectrogram(t[:16_000].astype(np.int16))[:WINDOW_FRAMES].T
                      for t in pos_trials[:4]])
        ).float().unsqueeze(1) / 127.0
        delta = (model(probe) - folded(probe)).abs().max().item()
    print(f"BN-fold max logit delta: {delta:.2e}")
    assert delta < 1e-3, "BN folding changed the model"

    streams = {"noise": noise_stream, "speech": speech_stream}
    with torch.no_grad():
        eval_model("float (BN-folded)", lambda x: folded(x), streams, pos_trials)

    # INT8 simulation, calibrated on a slice of training windows.
    x_all, split = payload["x"], payload["split"].bool()
    calib = x_all[split][:512].float() / 127.0
    sim = Int8Sim(folded, calib)
    eval_model("INT8 (post-training, simulated)", sim.forward, streams, pos_trials)

    # Compute / footprint report for the low-power question.
    macs = count_macs()
    params = count_params(folded)
    inf_per_s = 100 / CADENCE_FRAMES  # mel frames arrive at 100 Hz
    print("\n=== compute / footprint ===")
    print(f"params: {params} (~{params / 1024:.1f} KB int8 + biases)")
    print(f"MACs/inference: {macs['total']:,} ({ {k: v for k, v in macs.items() if k != 'total'} })")
    print(f"inference cadence: every {CADENCE_FRAMES * 10} ms -> "
          f"{inf_per_s:.1f} inf/s -> {macs['total'] * inf_per_s / 1e6:.1f}M MACs/s")
    print("ESP32-P4 @ 360 MHz, ~2 cycles/MAC scalar int8: "
          f"~{macs['total'] * inf_per_s * 2 / 360e6 * 100:.1f}% of one core "
          "(plus mel frontend ~3M mul/s)")


if __name__ == "__main__":
    main()
