#!/usr/bin/env python
"""Live PC microphone test for socket-wake v3.

Requires ffmpeg on PATH. On Windows this uses DirectShow by default:
    python live_mic_detect.py

List microphone names:
    ffmpeg -list_devices true -f dshow -i dummy

Use a specific mic:
    python live_mic_detect.py --device "Microphone Name Here"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))

from socket_wake.features.mel import mel_spectrogram  # noqa: E402
from socket_wake.model.kws_cnn import KWSConvNet, fold_batchnorm  # noqa: E402

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 16_000
CADENCE_SAMPLES = 480  # 30 ms
CONSECUTIVE = 2
REFRACTORY_STEPS = 98 // 3


def load_model(ckpt: Path):
    state = torch.load(ckpt, weights_only=False, map_location="cpu")
    model = KWSConvNet(n_classes=state.get("n_classes", 2))
    model.load_state_dict(state["model"])
    model.eval()
    return fold_batchnorm(model).eval()


def ffmpeg_cmd(device: str | None) -> list[str]:
    audio = f"audio={device}" if device else "audio=default"
    return [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "dshow", "-i", audio,
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-",
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=ROOT / "models" / "hey-socket-v1" / "checkpoint_v3.pt")
    p.add_argument("--device", help="DirectShow microphone name; omit for audio=default")
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--print-every", type=float, default=0.5, help="seconds between status lines")
    args = p.parse_args()

    model = load_model(args.ckpt)
    cmd = ffmpeg_cmd(args.device)
    print("starting:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None

    ring: deque[int] = deque(maxlen=WINDOW_SAMPLES)
    consec = 0
    lockout = 0
    fires = 0
    last_print = 0.0
    last_prob = 0.0

    print("listening. Say 'hey socket'. Ctrl+C to stop.", flush=True)
    try:
        while True:
            raw = proc.stdout.read(CADENCE_SAMPLES * 2)
            if len(raw) < CADENCE_SAMPLES * 2:
                print("ffmpeg ended or no audio device opened", file=sys.stderr)
                return 1
            chunk = np.frombuffer(raw, dtype="<i2").copy()
            ring.extend(int(x) for x in chunk)
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            peak = int(np.max(np.abs(chunk))) if chunk.size else 0

            if len(ring) < WINDOW_SAMPLES:
                continue
            pcm = np.asarray(ring, dtype=np.int16)
            mel = mel_spectrogram(pcm)[:98].T.astype(np.float32)
            x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0) / 127.0
            with torch.no_grad():
                logits = model(x)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
            last_prob = prob

            fired = False
            if lockout > 0:
                lockout -= 1
            elif prob >= args.threshold:
                consec += 1
                if consec >= CONSECUTIVE:
                    fires += 1
                    fired = True
                    consec = 0
                    lockout = REFRACTORY_STEPS
            else:
                consec = 0

            now = time.time()
            if fired or now - last_print >= args.print_every:
                marker = " DETECT" if fired else ""
                print(f"p={last_prob:.3f} rms={rms:6.0f} peak={peak:5d} consec={consec} fires={fires}{marker}", flush=True)
                last_print = now
    except KeyboardInterrupt:
        print(f"\nstopped. fires={fires}")
        return 0
    finally:
        proc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
