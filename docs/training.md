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

```powershell
# via huggingface_hub / datasets, or a direct file pull -- document the
# exact command here once the data-build stage confirms it works.
```

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

## Reproducing the current canned model

```powershell
cd D:\Arduino\socket-wake
python -m socket_wake.train --word "hey socket" --out models/hey-socket-v1
python -m socket_wake.export models/hey-socket-v1/checkpoint.pt models/hey-socket-v1
cargo test -p socket-wake-runtime --test canned_model_test -- --nocapture
```

(TODO: the v2 pipeline with real noise negatives will replace or extend
this — update this section once that lands.)
