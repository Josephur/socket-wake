# SPDX-License-Identifier: Apache-2.0
"""Download the MUSAN noise-only subset and save it as 16 kHz mono int16
WAV files under ``models/hey-socket-v1/noise_clips/``.

Source: ``bilguun/musan-noise`` on HuggingFace (~930 clips, noise only, no
speech/music -- exactly the "noise" bucket of the three-bucket KWS recipe).
Its WAVs are already 16 kHz mono int16 (standard MUSAN format), so no
resampling is needed -- verified by inspecting one downloaded clip's wave
header before writing this.

We deliberately do NOT use ``datasets.load_dataset()`` here: it builds a
full audiofolder index (metadata + hashing) over the HF resolver before
yielding a single sample, which took >20 minutes and never completed in
practice (killed). Individual file downloads via the HF "list repo files"
API (a single fast JSON call) + plain HTTP GET per file are near-instant
(~0.7s per ~550 KB clip via direct curl testing) and this dataset has no
split/metadata structure we'd lose by skipping the datasets lib.

Fallback: ``noisy-alpaca-test/MUSAN-noise-audio-only`` if the primary
repo or its file listing is unreachable. See docs/training.md "MUSAN"
section for the exact results of the run used to build v2.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import time
import urllib.request
from pathlib import Path

PRIMARY = "bilguun/musan-noise"
FALLBACK = "noisy-alpaca-test/MUSAN-noise-audio-only"

OUT_DIR = Path("models/hey-socket-v1/noise_clips")
MAX_WORKERS = 16


def _list_wav_files(repo: str) -> list[str]:
    url = f"https://huggingface.co/api/datasets/{repo}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        meta = json.loads(resp.read().decode("utf-8"))
    return [
        s["rfilename"] for s in meta["siblings"]
        if s["rfilename"].lower().endswith(".wav")
    ]


def _download_one(repo: str, rfilename: str, out_dir: Path) -> bool:
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{rfilename}"
    # Flatten any subdirectory in rfilename into a safe flat filename.
    flat_name = rfilename.replace("/", "_")
    dest = out_dir / flat_name
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                dest.write_bytes(resp.read())
            return True
        except Exception as e:
            if attempt == 1:
                print(f"[fetch_musan] failed to fetch {rfilename}: {e}")
            time.sleep(0.5)
    return False


def fetch(out_dir: Path = OUT_DIR, limit: int | None = None) -> tuple[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = PRIMARY
    t0 = time.time()
    try:
        wav_files = _list_wav_files(PRIMARY)
        if not wav_files:
            raise RuntimeError("no .wav files listed")
    except Exception as e:
        print(f"[fetch_musan] {PRIMARY} listing failed ({e}); "
              f"falling back to {FALLBACK}")
        dataset_name = FALLBACK
        wav_files = _list_wav_files(FALLBACK)

    if limit is not None:
        wav_files = wav_files[:limit]
    print(f"[fetch_musan] {dataset_name}: {len(wav_files)} wav files listed "
          f"in {time.time() - t0:.1f}s")

    count = 0
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {
            ex.submit(_download_one, dataset_name, f, out_dir): f
            for f in wav_files
        }
        for i, fut in enumerate(cf.as_completed(futs)):
            if fut.result():
                count += 1
            if (i + 1) % 100 == 0:
                print(f"[fetch_musan] {i + 1}/{len(wav_files)} processed, "
                      f"{count} saved...")

    print(f"[fetch_musan] done: {count} clips saved to {out_dir} "
          f"(source={dataset_name}, elapsed={time.time() - t0:.1f}s)")
    return dataset_name, count


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Download MUSAN noise clips.")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    fetch(args.out, args.limit)


if __name__ == "__main__":
    main()
