"""Stage 5: can the dataset be solved WITHOUT understanding the speech?

We train deliberately dumb classifiers (logistic regression) on features that
contain no words at all, and measure how well each one separates complete from
incomplete clips. AUROC 0.5 = pure guessing, 1.0 = perfect.

  length       clip duration and tail length. MUST be ~0.5: examples are made
               in pairs with the same word count and tail, so anything higher
               means a construction bug. The script fails above the limit.
  tail         loudness of the real silence after the last word.
  last_30ms    loudness of the final 30 ms of speech.
  last_500ms   loudness contour over the final 500 ms of speech.

The last three are reported, not enforced. Nothing in these clips is edited,
so any signal there is how people really sound (trailing off at the end of a
sentence, breathing in during a pause). They are the bar the real model has
to beat clearly: if it did no better, it would just be a loudness meter.

Writes results/shortcut_check.json.
"""

import csv
import json
import sys

import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from indic_eot.config import REPO_ROOT, load_params

METADATA = REPO_ROOT / "data" / "processed" / "eot" / "metadata.csv"
OUT_FILE = REPO_ROOT / "results" / "shortcut_check.json"


def rms_db(x: np.ndarray) -> float:
    """Loudness of a piece of audio in decibels."""
    return float(20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-7)) if len(x) else -140.0


def features(row: dict) -> dict[str, list[float]]:
    audio, sr = sf.read(REPO_ROOT / row["path"], dtype="float32")
    tail_len = int(int(row["tail_ms"]) / 1000 * sr)
    speech, tail = audio[:-tail_len], audio[-tail_len:]

    def last(ms_from: int, ms_to: int) -> np.ndarray:
        """Speech between ms_from and ms_to milliseconds before the cut."""
        a, b = int(ms_from / 1000 * sr), int(ms_to / 1000 * sr)
        return speech[len(speech) - a : len(speech) - b]

    overall = rms_db(speech)  # used to make loudness relative to the speaker's level
    return {
        "length": [len(audio) / sr, tail_len / sr],
        "tail": [rms_db(tail), rms_db(tail) - overall],
        "last_30ms": [rms_db(last(30, 0)) - overall],
        "last_500ms": [rms_db(last(a, a - 100)) - overall for a in (500, 400, 300, 200, 100)],
    }


def main() -> None:
    cfg = load_params()["shortcut_check"]
    with open(METADATA, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    feats = [features(r) for r in tqdm(rows, desc="extracting features")]
    labels = np.array([int(r["label"]) for r in rows])
    is_train = np.array([r["split"] == "train" for r in rows])
    is_test = np.array([r["split"] == "test" for r in rows])

    results = {}
    for group in feats[0]:
        x = np.array([f[group] for f in feats])
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(x[is_train], labels[is_train])
        scores = model.predict_proba(x[is_test])[:, 1]
        results[group] = round(float(roc_auc_score(labels[is_test], scores)), 4)

    limit = cfg["max_auroc"]
    failed = [g for g in cfg["must_be_chance"] if abs(results[g] - 0.5) > limit - 0.5]
    report = {
        "test_clips": int(is_test.sum()),
        "auroc": results,
        "limit_for_must_be_chance": limit,
        "passed": not failed,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failed:
        sys.exit(f"Shortcut check failed for: {failed}")


if __name__ == "__main__":
    main()
