# socket-wake

Open-source keyword spotting for resource-constrained MCUs.

- **Tiny**: ≤ 50 KB INT8 weights, ≤ 24 KB peak RAM, runs on Cortex-M4F with no PSRAM.
- **Embeddable**: stable C ABI — link a prebuilt `.a` against your sketch, or vendor the Rust source.
- **Easy to retrain**: TTS-augmented training pipeline. Bring ~10-20 real recordings and a GPU box, walk away with a custom wake word in 30 minutes.
- **Apache-2.0 throughout**, including the trained weights we ship.

See [DESIGN.md](DESIGN.md) for the architecture, quality bars, and v1 scope.

## Status

Pre-implementation. The design is approved; the repo is bootstrapped; implementation plan is the next step.

## License

Apache-2.0. See `LICENSE` (to be added at first implementation commit).