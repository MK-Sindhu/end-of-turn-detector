"""Stage: the away game. Score our audio model on Smart Turn's test set.

Both models see identical clips. Reported per language:
  hin   Hindi: the language our model was trained on, but different speakers,
        microphones and speaking styles from anything it has seen.
  ben, mar   Bengali and Marathi: languages our model has NEVER been trained
        on. This measures how well it transfers to other Indian languages.

Smart Turn was trained on the training half of this same collection, so it
has the home advantage here (as we had in the streaming simulation).

Only the audio model is scored: the dataset has no transcripts, so the text
and fusion models cannot be run.

Output: results/smart_turn_test_metrics.json, results/smart_turn_test_predictions.csv
"""

import csv
import json

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from indic_eot.audio_model import AudioEOT, FeaturePrep
from indic_eot.baselines import SmartTurn
from indic_eot.config import REPO_ROOT, load_params
from indic_eot.training import classification_metrics

MANIFEST = REPO_ROOT / "data" / "external" / "smart_turn_test" / "manifest.csv"
AUDIO_MODEL = REPO_ROOT / "models" / "audio" / "model.pt"
METRICS_FILE = REPO_ROOT / "results" / "smart_turn_test_metrics.json"
PREDICTIONS_FILE = REPO_ROOT / "results" / "smart_turn_test_predictions.csv"


def metrics_with_ci(labels: np.ndarray, probs: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """Metrics plus 95% intervals from resampling clips. (This dataset gives no
    speaker or recording ids, so clips are treated as independent.)"""
    point = classification_metrics(labels, probs)
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in point}
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        if len(set(labels[idx])) < 2:
            continue
        for k, v in classification_metrics(labels[idx], probs[idx]).items():
            samples[k].append(v)
    return {
        "clips": len(labels),
        **{k: round(v, 4) for k, v in point.items()},
        # How each class fares on its own, at threshold 0.5.
        "complete_called_complete": round(float((probs[labels == 1] >= 0.5).mean()), 4),
        "incomplete_called_incomplete": round(float((probs[labels == 0] < 0.5).mean()), 4),
        "ci95": {
            k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
            for k, v in samples.items()
        },
    }


def main() -> None:
    params = load_params()
    sr = params["audio"]["sample_rate"]
    max_seconds = params["audio"]["max_seconds"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))

    encoder = params["train_audio"]["encoder"]
    prep = FeaturePrep(encoder, max_seconds, sr)
    ours = AudioEOT(encoder, max_seconds, 0.0)
    ours.load_state_dict(torch.load(AUDIO_MODEL, map_location="cpu"))
    ours = ours.to(device).eval()
    cfg = params["baselines"]["smart_turn"]
    smart_turn = SmartTurn(cfg["repo"], cfg["file"])

    probs = {"ours_audio": [], "smart_turn": []}
    for start in tqdm(range(0, len(rows), 32), desc="scoring"):
        clips = [
            sf.read(REPO_ROOT / r["path"], dtype="float32")[0] for r in rows[start : start + 32]
        ]
        with torch.inference_mode():
            mel, mask = prep(clips)
            probs["ours_audio"] += (
                torch.sigmoid(ours(mel.to(device), mask.to(device))).cpu().tolist()
            )
        probs["smart_turn"] += [smart_turn.predict(c) for c in clips]

    labels = np.array([int(r["label"]) for r in rows])
    probs = {k: np.array(v) for k, v in probs.items()}
    langs = np.array([r["lang"] for r in rows])
    synthetic = np.array([r["synthetic"] == "1" for r in rows])

    report = {
        "note": (
            "Smart Turn was trained on the training half of this collection (home advantage). "
            "Our audio model was trained only on FLEURS Hindi and has never seen this data."
        ),
        "by_language": {},
    }
    for lang in sorted(set(langs)):
        report["by_language"][lang] = {}
        for subset_name, subset in {
            "all": langs == lang,
            "human_recordings": (langs == lang) & ~synthetic,
            "synthetic_speech": (langs == lang) & synthetic,
        }.items():
            if subset.sum() < 50 or len(set(labels[subset])) < 2:
                continue
            report["by_language"][lang][subset_name] = {
                name: metrics_with_ci(labels[subset], p[subset]) for name, p in probs.items()
            }

    METRICS_FILE.write_text(json.dumps(report, indent=2) + "\n")
    with open(PREDICTIONS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "lang", "label", "p_ours_audio", "p_smart_turn"]
        )
        writer.writeheader()
        for r, a, b in zip(rows, probs["ours_audio"], probs["smart_turn"]):
            writer.writerow(
                {
                    "id": r["id"],
                    "lang": r["lang"],
                    "label": r["label"],
                    "p_ours_audio": f"{a:.5f}",
                    "p_smart_turn": f"{b:.5f}",
                }
            )

    print(
        f"\n{'language':9} {'subset':17} {'clips':>6}  {'model':11} {'accuracy':>9} {'auroc':>7} {'ends ok':>8} {'pauses ok':>10}"
    )
    for lang, subsets in report["by_language"].items():
        for subset_name, models in subsets.items():
            for name, m in models.items():
                print(
                    f"{lang:9} {subset_name:17} {m['clips']:6d}  {name:11} {m['accuracy']:9.4f} {m['auroc']:7.4f} "
                    f"{m['complete_called_complete']:8.4f} {m['incomplete_called_incomplete']:10.4f}"
                )


if __name__ == "__main__":
    main()
