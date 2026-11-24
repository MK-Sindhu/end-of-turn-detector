"""Tests for the deployable detector and its numpy feature code."""

import numpy as np
import pytest

from indic_eot.detector import ASSETS, EndOfTurnDetector
from indic_eot.features import log_mel, prepare

needs_export = pytest.mark.skipif(
    not (ASSETS / "export.json").exists(), reason="run the export stage first"
)


def test_prepare_shapes_and_mask():
    filters = np.ones((201, 80)) / 201
    mel, mask = prepare(np.zeros(16000, dtype=np.float32), filters, max_seconds=8.0)
    assert mel.shape == (1, 80, 800)
    assert mask.shape == (1, 400)
    # One second of audio = 50 real frames, at the right-hand end.
    assert mask.sum() == 50
    assert mask[0, -50:].all()


def test_prepare_keeps_the_end_of_long_audio():
    filters = np.ones((201, 80)) / 201
    _, mask = prepare(np.zeros(16000 * 20, dtype=np.float32), filters, max_seconds=8.0)
    assert mask.all()


@needs_export
def test_numpy_features_match_whisper_extractor():
    from transformers import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny", chunk_length=8)
    audio = np.random.default_rng(0).normal(0, 0.1, 8 * 16000).astype(np.float32)
    expected = extractor(audio, sampling_rate=16000, return_tensors="np").input_features[0]
    ours = log_mel(audio, np.load(ASSETS / "mel_filters.npy"))
    assert np.abs(ours - expected).max() < 1e-3


@needs_export
def test_detector_returns_a_probability():
    detector = EndOfTurnDetector.load()
    audio = np.random.default_rng(0).normal(0, 0.05, 3 * 16000).astype(np.float32)
    assert 0.0 <= detector.predict(audio) <= 1.0
    assert 0.0 <= detector.predict(audio[:800], text="ignored") <= 1.0
