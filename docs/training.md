# Training a wake word

This is the reusable, step-by-step recipe for training a `socket-wake`
model — both the current "hey socket" canned model and any future custom
word. Each section documents what to run and why; agents extending this
pipeline should update this file in the same commit as their code changes,
not after the fact.

## Why the v1 canned model isn't good enough

The first "hey-socket-v1" model (see git history around
`fix(export): emit trained KWSClassifier head...`) was trained **only on
TTS-synthesized speech** — the wake phrase plus other TTS phrases as
negatives. It has never seen:

- Non-speech noise (fans, traffic, music, room hum)
- Impulsive sounds (coughs, claps, door slams)
- The wake phrase mixed with any of the above

A linear classifier (which is what the SWWT v1 export currently produces —
see "Known limitation" below) trained this way will very likely
false-trigger on loud, broadband noise, because nothing in its training
data taught it to distinguish "the wake word" from "any sufficiently loud
sound with roughly the right spectral shape."

## The three-bucket data recipe (standard KWS practice)

Real wake-word training uses three data buckets, not two:

1. **Positive** — the wake phrase, many voices/accents/speeds.
2. **Hard-negative speech** — phonetically similar near-misses (e.g. "hey
   rocket" for "hey socket") plus unrelated normal speech.
3. **Noise negatives** — non-speech audio: music, traffic, hum, impulsive
   sounds. **MUSAN** (Music, Speech, And Noise corpus, OpenSLR resource 17,
   free) is the standard corpus for this. For impulsive sounds specifically
   (coughs, claps), ESC-50 or FSD50K are common companions.

### Augmentation: SNR-mixing (the part that matters most)

Don't just add noise clips as their own negative examples. For every
**positive** utterance, generate 2-3 copies mixed with a random noise clip
at a randomized SNR (5-30 dB is typical). This teaches the model "wake
word + fan noise" is still a positive — without it, the model only knows
clean-room speech and falls apart the moment there's any background noise.

Pure noise clips (no wake word) and hard-negative speech clips are used
as-is for the negative class.

### Hard-negative mining (iterative refinement)

After a first training pass, run the trained detector against audio it's
never seen (ideally real recordings, or at minimum a held-out noise/speech
split) and collect false triggers. Add those to the negative set and
retrain. Repeat 2-3 rounds. This is how production wake-word models get
their false-accept rate down over time.

## Pipeline stages

1. **TTS generation** (positives + hard negatives) — see
   `python/socket_wake/data/tts_dataset.py`. Uses whatever TTS endpoints
   are available on the network (Kokoro OpenAI-compatible API, F5-TTS
   REST). Ask the user for endpoint URLs if you don't have them; don't
   guess.
2. **Noise acquisition** — MUSAN or equivalent. See "MUSAN" section below
   for the exact download/subset approach used.
3. **Augmentation** — SNR-mixing module (TODO: document path once built).
4. **Training** — `models/hey-socket-v1/train.py`, using
   `socket_wake.model.ds_cnn.KWSClassifier` (400-dim input, matches the
   runtime's stacked-mel-frame buffer exactly — see "Known limitation"
   for why this must stay true even if the architecture changes).
5. **Export** — `socket_wake.export.export()`, produces
   `models/hey-socket-v1/weights.bin` in the SWWT v1 format
   (`runtime/src/weights.rs`).
6. **Benchmark (FAR/FRR)** — TODO: document path once the benchmark
   harness exists. This is the real acceptance gate: false-accept rate
   should be ≤ 1/hour on ambient noise (per the original design spec's
   quality bar), and false-reject rate should be reasonably low on
   held-out positives. "Compiles and doesn't crash" is NOT sufficient
   evidence the model is usable — always run the FAR/FRR benchmark before
   calling a model "done."

## MUSAN

The full corpus (`musan.tar.gz`, OpenSLR resource 17,
https://www.openslr.org/17/) is ~10.3 GB and splits into `speech/`,
`music/`, and `noise/` top-level directories — but downloading and
extracting just the subset we need from a 10 GB tarball is slower than
using a pre-split mirror.

**We use the HuggingFace pre-split noise subset instead:**
[`bilguun/musan-noise`](https://huggingface.co/datasets/bilguun/musan-noise)
(~930 noise-only samples, no speech/music). This is exactly the "noise"
bucket from the three-bucket recipe above — small, fast to pull, and we
don't need MUSAN's speech/music subsets since our hard-negative *speech*
already comes from TTS.

**Do NOT use `datasets.load_dataset()` for this** — it auto-decodes the
Audio feature via `torchcodec`, which needs FFmpeg's "full-shared" DLL
build. That's not what's normally installed on a dev box and isn't worth
fighting; `load_dataset()` will throw a wrapped `DatasetGenerationError`
whose real cause (buried several frames down) is
`ModuleNotFoundError: No module named 'torchcodec'` or, after installing
it, `OSError: Could not load this library: ...libtorchcodec_core8.dll`.

Instead, pull the raw `.wav` files directly via `huggingface_hub`
(`list_repo_files` + threaded HTTP GET). This is what
`python/socket_wake/data/fetch_musan.py` does:

```powershell
cd D:\Arduino\socket-wake
python -m socket_wake.data.fetch_musan
# -> models/hey-socket-v1/noise_clips/*.wav (930 files, ~16 kHz mono
#    int16 already -- verified by inspecting a downloaded clip's wave
#    header, no resampling needed)
```

Result of the actual run: 930/930 clips downloaded successfully via 16
parallel workers, essentially instant once the repo file listing (a
single JSON API call) completes.

If `bilguun/musan-noise` becomes unavailable, the fallback is
[`noisy-alpaca-test/MUSAN-noise-audio-only`](https://huggingface.co/datasets/noisy-alpaca-test/MUSAN-noise-audio-only)
(6.71 GB, still much smaller than the full tarball), or the full OpenSLR
tarball as a last resort.

## Known limitation: the SWWT v1 export is a single linear layer

`socket_wake.export.export()` currently collapses the *entire* trained
model (including any hidden layers) into a single 400→n_classes dense
layer via `lstsq` fit on a calibration set. This throws away any
nonlinearity the model learned during training — what ships to the device
is functionally logistic regression on raw mel-time energy, regardless of
how deep the trained PyTorch model was.

This means: even with a properly augmented dataset (real noise, hard
negatives, SNR-mixing), the *exported* model's representational capacity
is capped at "linear boundary in 400-dim mel-energy space." That's a real
ceiling. If the FAR/FRR benchmark (once built) shows the linear model
isn't good enough even with better data, the next step is exporting the
real depthwise/pointwise conv stack layer-by-layer (the runtime's
`Cnn::run` already supports iterating multiple layers — see
`runtime/src/weights.rs`'s `Layers` iterator and `runtime/src/cnn.rs`'s
`apply_layer` — nobody has used this multi-layer path yet; today's
"1 layer" exports are the only thing that's been exercised).

## Reproducing the v1 (TTS-only) canned model

```powershell
cd D:\Arduino\socket-wake
python -m socket_wake.train --word "hey socket" --out models/hey-socket-v1
python -m socket_wake.export models/hey-socket-v1/checkpoint.pt models/hey-socket-v1
cargo test -p socket-wake-runtime --test canned_model_test -- --nocapture
```

This is the TTS-only dataset (`train_dataset.pt`) with no noise negatives
— superseded by the v2 recipe below, kept for reference/comparison.

## Reproducing the v2 (noise-augmented) dataset

```powershell
cd D:\Arduino\socket-wake
python -m socket_wake.data.fetch_musan          # -> noise_clips/*.wav (930 files)
python -m socket_wake.data.build_v2_dataset     # -> train_dataset_v2.pt
```

`build_v2_dataset.py` assembles four buckets into one dataset:

1. **Positives (clean)** — fresh "hey socket" TTS synthesis across 6
   Kokoro voices + F5-TTS (`_VOICE_COMBOS`, 9 combos × `N_POS_PER_VOICE`
   requests each).
2. **Positives (SNR-mixed)** — each clean positive gets
   `SNR_COPIES_PER_POS` (2) copies mixed with a random noise clip at a
   random SNR in `SNR_RANGE_DB` (5-30 dB), via `snr_mix.mix_at_snr()`.
3. **Negatives (speech)** — v1's existing negative windows reused as-is
   (saves TTS calls) PLUS ~150 new hard-negative phrases: phonetic
   near-misses to "hey socket" (`HARD_NEG_PHONETIC`) and unrelated
   everyday sentences (`HARD_NEG_UNRELATED`).
4. **Negatives (noise)** — the 930 MUSAN clips, windowed and capped.

### Critical gotcha: cap windows-per-noise-clip, or class balance breaks silently

**First attempt at this pipeline produced a dataset with an 86:1
negative:positive ratio** (2,305,878 negative windows vs 26,688
positive) that would have silently trained a useless model — one that
always predicts "not target" and scores ~99% "accuracy" while having 0%
recall on the actual wake word. The cause: MUSAN noise clips run many
seconds long, and windowing with a 1-frame hop (matching the short TTS
utterances' stacking) produces thousands of near-duplicate overlapping
windows per clip — averaged out to ~2,400 windows per clip across all
930 clips.

**Fix:** noise clips use a *different*, non-overlapping hop
(`NOISE_HOP_FRAMES = WINDOW_FRAMES`) plus a hard per-clip cap
(`NOISE_WINDOWS_PER_CLIP_CAP = 30`, randomly subsampled if a clip would
exceed it). This keeps noise diversity (windows drawn from all 930
clips) without letting clip *length* dominate the negative pool.

**Lesson for future pipeline changes:** whenever a new negative source
is added, sanity-check the resulting class balance (`Counter(y)`) before
training — don't assume "more negative data" is automatically fine. A
K:1 negative:positive ratio beyond roughly 5-10:1 should be treated as a
red flag worth investigating, not a given.

### Result of the actual v2 build (2026-07-03)

```
total windows: 111,958  (target=19,311, not_target=92,647, ratio ~4.8:1)
sources:
  pos_clean_windows:        6,437
  pos_snr_mixed_windows:   12,874
  neg_v1_reused_windows:   52,130
  neg_hard_negative_windows: 15,057
  neg_pure_noise_windows:   25,460
  n_positive_base_clips:       60   (of 90 requested -- F5-TTS endpoint
                                      timed out on some requests under
                                      load; see below)
  n_hard_negative_clips:      102   (of 150 requested, same reason)
  n_noise_clips:               930  (all succeeded)
train/test split: 100,763 / 11,195  (90/10, stratified by class)
```

**F5-TTS reliability note:** the `mouth.stack-tech.local:7860` F5-TTS
endpoint threw repeated `POST ... failed twice: timed out` errors under
the ~12-worker concurrent load this script uses (`MAX_WORKERS = 12`,
shared across both Kokoro and F5 requests). Kokoro's OpenAI-compatible
endpoint held up fine at the same concurrency. If F5 clip yield drops
noticeably below what you requested, consider lowering `MAX_WORKERS` or
routing more of the load to Kokoro.

**Concurrency gotcha:** don't run `build_v2_dataset.py` from two
processes at once against the same output path — a race between two
runs (one with the pre-cap code, one with the post-cap code) produced a
corrupted intermediate file that silently carried over the *old*,
imbalanced window counts even after the cap fix landed, because the
slower/older process's write won the race and overwrote the correct
one. If results look suspicious after a code change, verify with
`tasklist` (Windows) / `ps` (Unix) that no stale process is still
running the old code before trusting a rebuild's output.

### Then train + export + verify, same as v1

```powershell
# train.py needs to point at train_dataset_v2.pt instead of
# train_dataset.pt -- confirm/update the script's dataset path before
# running if it hasn't been updated yet.
python models/hey-socket-v1/train.py
python -m socket_wake.export models/hey-socket-v1/checkpoint.pt models/hey-socket-v1
cargo test -p socket-wake-runtime --test canned_model_test -- --nocapture
```
