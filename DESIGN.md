# socket-wake — Design Spec

Date: 2026-07-03
Status: approved (user: "Sounds good, lets roll.")
Owner: Josephur

## Problem

Wake-word detection on resource-constrained MCUs is underserved by open tooling.
Espressif's WakeNet (the obvious choice for ESP32-P4) ships with a closed training
pipeline — custom words require a paid engagement with ≥20,000 user-recorded
samples and a 2-3 week turnaround (espressif/esp-sr docs:
`wake_word_engine/ESP_Wake_Words_Customization.html`). Picovoice Porcupine
solves the problem with a paid product but is closed-source.

There's no open-source, embeddable, easy-to-retrain wake-word library. We
build one.

## Goals

- **Small**: ≤ 50 KB INT8 weights, ≤ 24 KB peak RAM, ≤ 200 KB flash total
  (runtime + weights). Embeddable on Cortex-M4F with 64 KB RAM, no PSRAM.
- **Easy to retrain**: anyone with a GPU box and ~10-20 real recordings of a
  target phrase can train a new model in 30 minutes via a one-line CLI.
  Synthetic-data augmentation (TTS) is the headline feature — published work
  (Google Speech Commands, Picovoice, Snips) shows it closes the gap to
  ~95-98% of human-recorded accuracy.
- **Embeddable everywhere**: portable C ABI so consumers don't need to use Rust.
- **Honest**: Apache-2.0 throughout, *including the trained model weights*.
  No GPL contamination.

## Non-goals (v1)

- Multi-word detection. (v2: small dictionary.)
- Multi-mic / beamforming / AEC. (v2: optionally consume an upstream AFE.)
- Speaker verification / identification.
- On-device personalization. (v2: classifier-head fine-tune from on-device samples.)
- Cloud-hosted training portal. (v1 is offline-capable; cloud is a service someone else could build.)

## Architecture

```
[mic I2S] -> [mel features] -> [tiny INT8 CNN] -> [state machine] -> "DETECTED"
                                  ^
                                  | weights.bin (Apache-2.0, committed)
[Python training: real + synthetic audio] -> [quantize + export] -+
```

### Repository layout

```
Josephur/socket-wake/
├── LICENSE                              Apache-2.0
├── README.md                            what + why + quickstart
├── Cargo.toml                           workspace root
├── runtime/                             Rust, no_std, no_std_compat
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs              public C ABI + re-exports
│   │   ├── mel.rs              mel-filterbank feature extractor
│   │   ├── cnn.rs              INT8 quantized CNN kernel (HWC layout)
│   │   ├── state.rs            detection state machine
│   │   ├── weights.rs          weight loader
│   │   └── arena.rs            fixed-size arena allocator (no heap)
│   ├── c/                      C ABI surface for non-Rust consumers
│   │   ├── socket_wake.h
│   │   └── examples/arduino/   prebuilt .a + minimal sketch
│   ├── esp_idf/                ESP-IDF component (P4 demo path)
│   └── tests/                  host-side cargo tests (run on Linux/macOS)
├── python/                              training pipeline
│   ├── pyproject.toml                   uv/pip install -e .
│   ├── socket_wake/
│   │   ├── data/                        audio dataset loaders (Speech Commands + user WAVs)
│   │   ├── features/                    mel extraction (matches runtime byte-for-byte)
│   │   ├── model/                       DS-CNN-L PyTorch definition
│   │   ├── train.py                     main training entry
│   │   ├── synthesize.py                Piper-TTS data augmentation
│   │   └── export.py                    trained model -> INT8 weights.bin
│   └── tests/
├── models/                              pre-trained model files (Apache-2.0 weights)
│   └── hey-socket-v1/                   canned word for bring-up
└── docs/
    ├── architecture.md
    ├── training.md                      how to train your own word
    └── porting.md                       how to embed runtime on a new MCU
```

### Public C ABI (the only thing consumers see)

```c
// runtime/c/socket_wake.h
typedef struct socket_wake_detector socket_wake_detector_t;

socket_wake_detector_t *socket_wake_create(const void *weights,
                                            size_t weights_bytes,
                                            size_t sample_rate_hz);
void socket_wake_feed(sw_detector_t *d, const int16_t *pcm, size_t samples);
bool socket_wake_detected(sw_detector_t *d);   // true ONCE per detection
void socket_wake_destroy(sw_detector_t *d);

// Memory introspection (for the memory-profile CI test)
size_t socket_wake_peak_ram_bytes(const sw_detector_t *d);
size_t socket_wake_weights_bytes(const sw_detector_t *d);
```

Three call sites to embed: create, feed (in mic ISR or main loop), check
detected. No callbacks, no owned RTOS tasks.

### Quality bars (v1 acceptance)

- **Accuracy**: ≥ 92% on Speech Commands 12-class test set; ≥ 95% on the
  canned "hey socket" word's held-out real-recording set.
- **False-accept rate**: ≤ 1 per hour on MUSAN ambient living-room noise floor.
- **False-reject rate**: ≤ 5% at 1 m on a 16 kHz mic.
- **Latency**: end-to-end detect ≤ 300 ms from end of word.
- **Footprint**: ≤ 50 KB weights, ≤ 24 KB peak RAM, ≤ 200 KB flash on Cortex-M4F.
- **Inference time**: ≤ 10 ms per 30 ms window on ESP32-S3 at 240 MHz.

### Training pipeline (the "easy retrain" half)

`python/socket_wake/train.py` is the entry point. Steps:

1. **Data**: download Speech Commands (open-licensed), plus user-supplied WAVs
   of the target word (≥ 10 recordings recommended).
2. **Augmentation**: room impulse response convolution, MUSAN noise mixing,
   SpecAugment, time-stretch. Standard KWS data-aug pipeline.
3. **Synthetic data**: `synthesize.py` calls **Piper TTS** (MIT-licensed, ONNX,
   offline, ~40 voices, 30+ languages) to generate ~5k utterances of the
   target phrase across voice/accent variations. Mixed 50/50 with real
   recordings.
4. **Model**: **DS-CNN-L** (Depthwise Separable CNN, ~24K params, ~25 KB INT8).
   Published baseline from Google's KWS paper, well-understood, fast on small MCUs.
5. **Train**: 50 epochs, single GPU, ~30 minutes.
6. **Export**: `export.py` quantizes with calibration on a held-out set,
   emits `models/<word>/weights.bin` (~25 KB) and `models/<word>/header.h`.

### Why Rust for the runtime (recap)

- `no_std` + `core::arch::arm` intrinsics = tight, predictable code on any MCU.
- Type system catches buffer overruns / wrong casts in the CNN kernel —
  these have caused silent accuracy regressions in C/C++ KWS projects.
- `cargo test` for host-side unit tests (feed known PCM, assert detected).
- `bindgen` + `#[no_mangle] extern "C"` gives the consumer a C ABI for free.

The Arduino example links a prebuilt `libsocket_wake.a` against the user's
sketch — no Cargo in their build.

## Components / files (v1 deliverables)

| File | One-liner | Depends on |
|---|---|---|
| `runtime/src/lib.rs` | C ABI surface + re-exports of mel/cnn/state/arena/weights | all other runtime modules |
| `runtime/src/mel.rs` | Raw int16 PCM -> mel spectrogram frames; bit-matches Python | `core::math` |
| `runtime/src/cnn.rs` | INT8 DS-CNN inference kernel, HWC layout, cache-aware | `weights.rs`, `arena.rs` |
| `runtime/src/state.rs` | Detection state machine; hysteresis across consecutive frames | `cnn.rs` |
| `runtime/src/weights.rs` | Parse + validate weights.bin at create time | nothing |
| `runtime/src/arena.rs` | Bump allocator; peak usage tracked for the CI memory test | nothing |
| `runtime/c/socket_wake.h` | The only public header | — |
| `runtime/esp_idf/main/` | ESP-IDF demo binary for P4 (or S3, C3, ...) | runtime crate |
| `runtime/tests/` | Host-side tests (Linux/macOS), including memory profile CI test | runtime crate |
| `python/socket_wake/features/` | Mel extraction that byte-matches Rust `mel.rs` | numpy |
| `python/socket_wake/model/` | DS-CNN-L PyTorch definition | torch |
| `python/socket_wake/data/` | Audio dataset loaders (Speech Commands + user WAVs) | torchaudio |
| `python/socket_wake/synthesize.py` | Piper-TTS data augmentation | piper-tts |
| `python/socket_wake/train.py` | Main training entry | torch + everything above |
| `python/socket_wake/export.py` | Trained model -> INT8 weights.bin + header.h | numpy |
| `models/hey-socket-v1/` | Pre-trained weights + header.h (Apache-2.0) | — |

## Testing strategy

### Runtime tests (host-side, `cargo test`)

- `mel`: feed known PCM, assert frame output bit-matches Python reference.
- `cnn`: feed known activations, assert output bit-matches reference.
- `state`: feed scripted frame sequences, assert detected/not-detected.
- `detector`: end-to-end with recorded test audio; assert detected within tolerance.
- **Memory profile CI**: `cargo test --test memory_profile` creates a detector,
  feeds a long random PCM stream, asserts `peak_ram_bytes() <= 24 * 1024`.

### Python tests (`pytest`)

- `features`: assert bit-equivalence with Rust mel on shared test vectors.
- `model`: train for 1 epoch on tiny subset, assert loss decreases.
- `synthesize`: smoke test that Piper invocation produces expected WAV count.
- `export`: train a tiny model, export, re-load in Python, assert forward
  pass matches pre-export float model within INT8 quantization tolerance.

### End-to-end

- Train `hey-socket-v1` on Speech Commands + a few user recordings + Piper
  synth.
- Verify on the device (or in a Linux simulator that mimics I2S).
- Latency and RAM measured under realistic mic conditions.

## Verification

For the project's own bring-up: the memory profile test catches
regressions in real time, and `cargo test` runs in < 5 seconds on the
host, so every PR is validated.

For the canned model: validation against the Speech Commands test set + a
held-out real-recording set, with the FAR measured against MUSAN's living
room noise floor.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Rust toolchain friction for contributors | Provide prebuilt `libsocket_wake.a` + C header for the common ESP32 path; Rust source is for those who want to modify or rebuild. |
| Piper TTS voice variety is limited (~40 voices) | Augment with formant / pitch perturbation in `synthesize.py`; this is standard in published KWS work and largely removes the variety limit. |
| DS-CNN-L accuracy not competitive for noisy environments | Add optional AFE/VAD pre-filter (v2); the v1 spec explicitly targets the 92-95% range, not 99%+. |
| INT8 quantization drops accuracy unexpectedly | Calibration set used in `export.py`; the CI test asserts pre/post export accuracy within 1% absolute. |
| Embedded allocator fragmentation | `arena.rs` is a single bump allocator, no free — fragmentation impossible by construction. |

## Out of scope for v1 (but tracked)

- On-device personalization (record 5 samples, fine-tune, deploy).
- Multi-word dictionary.
- Speaker verification.
- Cloud training portal.
- Multi-mic / beamforming.

## License

Apache-2.0 throughout. Specifically, the trained `models/hey-socket-v1/weights.bin`
is committed under Apache-2.0, so consumers don't need to retrain to use the
project. Users who train their own words own those weights and can license
them however they like; the *tooling* that produced them stays Apache-2.0.