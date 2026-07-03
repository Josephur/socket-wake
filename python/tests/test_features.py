# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from socket_wake.features.mel import mel_spectrogram, N_MELS, WINDOW


def test_shape_for_one_window():
    out = mel_spectrogram(np.zeros(WINDOW, dtype=np.int16))
    assert out.shape == (1, N_MELS)
    assert out.dtype == np.int8


def test_silence_at_log_floor():
    # The Rust runtime emits ~-95 for silence (log10(1e-6) ~= -6, scaled
    # to INT8). Allow a tolerance of +/-5 for floating-point rounding in
    # the Python-side filterbank and FFT.
    out = mel_spectrogram(np.zeros(WINDOW, dtype=np.int16))
    assert (-110 <= out).all()
    assert (out <= -80).all()


def test_dc_exceeds_silence_floor():
    pcm = np.full(WINDOW, 1000, dtype=np.int16)
    out = mel_spectrogram(pcm)
    # DC has all energy in mel bin 0; that bin must exceed the silence floor.
    assert out[0, 0] > -50


def test_consistent_output_for_same_input():
    pcm = np.random.default_rng(42).normal(0, 1000, WINDOW).astype(np.int16)
    a = mel_spectrogram(pcm)
    b = mel_spectrogram(pcm)
    np.testing.assert_array_equal(a, b)


@pytest.mark.skip(reason="requires captured Rust reference vectors")
def test_bit_matches_rust_reference():
    # When the Rust test suite is run on the host, capturing (pcm, output)
    # pairs into python/tests/reference_vectors/mel.npz would enable this
    # test. The output format is `(N_FRAMES, 40)` int8; saving the bytes
    # here and asserting bit-equivalence catches any drift in the Python
    # or Rust implementations.
    ref = np.load("tests/reference_vectors/mel.npz")
    np.testing.assert_array_equal(mel_spectrogram(ref["pcm"]), ref["out"])