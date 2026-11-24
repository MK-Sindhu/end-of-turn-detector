"""Stage: score an existing open detector on the same decision points.

Smart Turn (by Pipecat/Daily, BSD-2 licence) is an open audio end-of-turn model
with the same shape as ours: a Whisper-tiny encoder looking at the last 8 s.
It was trained on far more data, in 23 languages including Hindi.

Running it on exactly the audio our models saw answers the question a reader
will ask first: are our numbers good, or just numbers?

The preprocessing below follows the project's own inference.py
(https://github.com/pipecat-ai/smart-turn): keep the last 8 s, pad with zeros
at the start, Whisper log-mel features, and the model outputs a probability.

Not included: LiveKit's turn detector. Its licence only allows use inside the
LiveKit Agents framework, not standalone, so it is not run here.

Output: results/baseline_points.csv, one row per decision point.
"""

import csv

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

from indic_eot.config import REPO_ROOT, load_params
from indic_eot.streaming import iter_decision_clips

OUT_FILE = REPO_ROOT / "results" / "baseline_points.csv"
SECONDS = 8
SAMPLE_RATE = 16000


def last_n_seconds(audio: np.ndarray) -> np.ndarray:
    """Keep the last 8 s, or pad with zeros at the beginning (as upstream does)."""
    n = SECONDS * SAMPLE_RATE
    if len(audio) >= n:
        return audio[-n:]
    return np.pad(audio, (n - len(audio), 0))


def main() -> None:
    params = load_params()
    cfg = params["baselines"]["smart_turn"]
    assert params["audio"]["sample_rate"] == SAMPLE_RATE

    model_path = hf_hub_download(cfg["repo"], cfg["file"])
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options=options)
    extractor = WhisperFeatureExtractor(chunk_length=SECONDS)

    rows = []
    for record, split, points, clips, _ in tqdm(iter_decision_clips(params), desc="smart-turn"):
        for point, clip in zip(points, clips):
            features = extractor(
                last_n_seconds(clip),
                sampling_rate=SAMPLE_RATE,
                return_tensors="np",
                padding="max_length",
                max_length=SECONDS * SAMPLE_RATE,
                truncation=True,
                do_normalize=True,
            ).input_features.astype(np.float32)
            prob = session.run(None, {"input_features": features})[0].reshape(-1)[0]
            rows.append(
                {
                    "utterance": record["path"],
                    "split": split,
                    "kind": point["kind"],
                    "word_index": point["word_index"],
                    "p_smart_turn": f"{float(prob):.5f}",
                }
            )

    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} decision points scored with {cfg['file']} -> {OUT_FILE.name}")


if __name__ == "__main__":
    main()
