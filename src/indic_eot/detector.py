"""The public interface: load the exported model and score audio.

    from indic_eot import EndOfTurnDetector

    det = EndOfTurnDetector.load()
    p = det.predict(audio)   # audio: float32 mono 16 kHz; returns 0..1

Needs only numpy and onnxruntime. No PyTorch, no network.
"""

import json
from pathlib import Path

import numpy as np

from indic_eot.features import SAMPLE_RATE, prepare

ASSETS = Path(__file__).resolve().parent / "assets"


class EndOfTurnDetector:
    def __init__(self, model_dir: Path | str = ASSETS, quantized: bool = True):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        with open(model_dir / "export.json", encoding="utf-8") as f:
            self.info = json.load(f)
        self.max_seconds = self.info["max_seconds"]
        self.mel_filters = np.load(model_dir / "mel_filters.npy")
        name = self.info["quantized_file"] if quantized else self.info["file"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # predictable speed on a shared server
        self.session = ort.InferenceSession(
            str(model_dir / name), options, providers=["CPUExecutionProvider"]
        )

    @classmethod
    def load(cls, model_dir: Path | str = ASSETS, quantized: bool = True):
        return cls(model_dir, quantized)

    def predict(self, audio: np.ndarray, text: str | None = None) -> float:
        """Probability (0..1) that the speaker has finished their turn.

        audio: mono float32 at 16 kHz, everything said so far up to the pause.
        Only the last 8 seconds are used. `text` is accepted so callers do not
        have to change later, but the exported model is audio-only and ignores it.
        """
        mel, mask = prepare(audio, self.mel_filters, self.max_seconds)
        (prob,) = self.session.run(None, {"mel": mel, "mask": mask})
        return float(prob[0])


__all__ = ["SAMPLE_RATE", "EndOfTurnDetector"]
