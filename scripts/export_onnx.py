"""Export the trained audio model to ONNX, and check the export is faithful.

Why: the trained model needs PyTorch (about 2 GB installed) to run. ONNX is a
portable model format that runs with the small onnxruntime library on a server,
and with onnxruntime-web inside a browser. This is what gets deployed.

Writes to src/indic_eot/assets/ (so the package can ship it):
  audio_eot.onnx        full precision (32-bit numbers)
  audio_eot.int8.onnx   weights stored as 8-bit numbers: about 4x smaller
  mel_filters.npy       the 80 mel bands used to make the model's input
  export.json           file names and settings

Then it scores the whole test split with both files and compares against the
PyTorch predictions saved during training, and measures CPU speed.
"""

import csv
import json
import time

import numpy as np
import soundfile as sf
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

from indic_eot.audio_model import AudioEOT
from indic_eot.config import REPO_ROOT, load_params
from indic_eot.detector import ASSETS, EndOfTurnDetector
from indic_eot.training import load_rows

MODEL_FILE = REPO_ROOT / "models" / "audio" / "model.pt"
PREDICTIONS = REPO_ROOT / "results" / "audio_predictions.csv"
METRICS_FILE = REPO_ROOT / "results" / "export_metrics.json"
FULL, SMALL = "audio_eot.onnx", "audio_eot.int8.onnx"


class Exportable(torch.nn.Module):
    """The trained model, returning a probability instead of a logit."""

    def __init__(self, model: AudioEOT):
        super().__init__()
        self.model = model

    def forward(self, mel: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(mel, mask))


def export(params: dict) -> None:
    encoder = params["train_audio"]["encoder"]
    max_seconds = params["audio"]["max_seconds"]
    ASSETS.mkdir(parents=True, exist_ok=True)

    model = AudioEOT(encoder, max_seconds, 0.0)
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
    frames = int(max_seconds * 50)
    torch.onnx.export(
        Exportable(model).eval(),
        (torch.zeros(1, 80, frames * 2), torch.ones(1, frames, dtype=torch.bool)),
        str(ASSETS / FULL),
        input_names=["mel", "mask"],
        output_names=["probability"],
        dynamic_axes={"mel": {0: "batch"}, "mask": {0: "batch"}, "probability": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    # Only the big matrix multiplications are stored as 8-bit. The two small
    # convolution layers at the input stay full precision: they are sensitive
    # and hold little of the size.
    quantize_dynamic(
        str(ASSETS / FULL),
        str(ASSETS / SMALL),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    extractor = WhisperFeatureExtractor.from_pretrained(encoder, chunk_length=int(max_seconds))
    np.save(ASSETS / "mel_filters.npy", np.asarray(extractor.mel_filters, dtype=np.float64))
    with open(ASSETS / "export.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "file": FULL,
                "quantized_file": SMALL,
                "max_seconds": max_seconds,
                "sample_rate": params["audio"]["sample_rate"],
                "encoder": encoder,
                "inputs": {"mel": "[batch, 80, 800] float32", "mask": "[batch, 400] bool"},
                "output": "probability the turn is complete, [batch]",
            },
            f,
            indent=2,
        )


def check(detector: EndOfTurnDetector, rows: list[dict], torch_probs: dict) -> dict:
    """Score the test split and compare with the PyTorch predictions."""
    probs, seconds = [], []
    for row in tqdm(rows, desc="scoring"):
        audio, _ = sf.read(REPO_ROOT / row["path"], dtype="float32")
        start = time.perf_counter()
        probs.append(detector.predict(audio))
        seconds.append(time.perf_counter() - start)
    probs = np.array(probs)
    labels = np.array([int(r["label"]) for r in rows])
    reference = np.array([torch_probs[r["id"]] for r in rows])
    return {
        "accuracy": round(float(((probs >= 0.5) == labels).mean()), 4),
        "pytorch_accuracy": round(float(((reference >= 0.5) == labels).mean()), 4),
        "max_probability_difference": round(float(np.abs(probs - reference).max()), 4),
        "mean_probability_difference": round(float(np.abs(probs - reference).mean()), 5),
        "decisions_changed": int(((probs >= 0.5) != (reference >= 0.5)).sum()),
        "cpu_ms_median": round(float(np.median(seconds) * 1000), 1),
        "cpu_ms_p95": round(float(np.percentile(seconds, 95) * 1000), 1),
    }


def main() -> None:
    params = load_params()
    export(params)

    rows = load_rows("test")
    with open(PREDICTIONS, encoding="utf-8") as f:
        torch_probs = {r["id"]: float(r["prob"]) for r in csv.DictReader(f)}

    report = {"test_clips": len(rows), "cpu_threads": 1}
    for name, quantized in (("full_precision", False), ("int8", True)):
        detector = EndOfTurnDetector.load(quantized=quantized)
        file = ASSETS / (SMALL if quantized else FULL)
        report[name] = {
            "size_mb": round(file.stat().st_size / 1e6, 1),
            **check(detector, rows, torch_probs),
        }

    METRICS_FILE.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
