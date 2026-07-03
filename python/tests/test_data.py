# SPDX-License-Identifier: Apache-2.0
import numpy as np

from socket_wake.data.augment import augment


def test_augment_preserves_length():
    rng = np.random.default_rng(0)
    wav = rng.normal(0, 0.1, size=16_000).astype(np.float32)
    out = augment(wav, 16_000, rng=rng)
    assert out.shape == wav.shape


def test_augment_does_not_mutate_input():
    rng = np.random.default_rng(0)
    wav = np.ones(1000, dtype=np.float32)
    original = wav.copy()
    _ = augment(wav, 16_000, rng=rng)
    np.testing.assert_array_equal(wav, original)


def test_augment_produces_different_output_across_calls():
    rng = np.random.default_rng(0)
    wav = np.zeros(1000, dtype=np.float32)
    a = augment(wav, 16_000, rng=rng)
    b = augment(wav, 16_000, rng=rng)
    # With the same input and a deterministic RNG, calls produce different
    # output only if the RNG is advanced across calls. We pass the same
    # RNG so the result is deterministic per call -- but noise samples
    # differ, so the output should also differ.
    assert not np.array_equal(a, b)


def test_synthesize_piper_available_returns_bool():
    from socket_wake.synthesize import piper_available
    # Just exercise the existence check; piper may or may not be installed.
    assert isinstance(piper_available(), bool)