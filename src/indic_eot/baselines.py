"""Wrappers for outside models we compare against."""

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import WhisperFeatureExtractor


class SmartTurn:
    """Smart Turn (Pipecat/Daily, BSD-2): an open audio end-of-turn model.

    The preprocessing follows the project's own inference.py
    (https://github.com/pipecat-ai/smart-turn): keep the last 8 s, pad with
    zeros at the start, Whisper log-mel features. The model outputs a
    probability directly.
    """

    SECONDS = 8
    SAMPLE_RATE = 16000

    def __init__(self, repo: str, file: str):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(hf_hub_download(repo, file), sess_options=options)
        self.extractor = WhisperFeatureExtractor(chunk_length=self.SECONDS)

    def predict(self, audio: np.ndarray) -> float:
        """P(turn complete) for mono 16 kHz float32 audio."""
        n = self.SECONDS * self.SAMPLE_RATE
        audio = audio[-n:] if len(audio) >= n else np.pad(audio, (n - len(audio), 0))
        features = self.extractor(
            audio,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=n,
            truncation=True,
            do_normalize=True,
        ).input_features.astype(np.float32)
        return float(self.session.run(None, {"input_features": features})[0].reshape(-1)[0])
