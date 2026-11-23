"""Stage 5: can the dataset be solved WITHOUT understanding the speech?

We train deliberately dumb classifiers (logistic regression) on features that
contain no words at all, and measure how well each one separates complete from
incomplete clips. AUROC 0.5 = pure guessing, 1.0 = perfect.

Feature groups, from "must be useless" to "allowed to help a bit":

  length      clip duration and tail length. Must be ~0.5: we matched these on
              purpose, so anything higher means a construction bug.
  tail        loudness of the silence after the speech. Must be ~0.5: both
              classes get the same room tone.
  edit_point  loudness of the last 30 ms of speech, right where we cut. A high
              score here means the model could detect our edit, not the speech.
  ending      loudness contour over the last 500 ms of speech. Speakers really
              do trail off at the end of a sentence, so some signal here is
              genuine. This is the bar the real model has to beat clearly.

The first two are hard limits (the script fails above the threshold). The last
two are reported so we know how much of the model's score could be cheap.

Is the edit_point signal our editing, or real speech? To find out we score it
again on two subsets of the incomplete clips:
  real_pause        the speaker genuinely paused there (long gap in the source
                    recording), so our cut removed nothing audible.
  connected_speech  the next word followed almost immediately, so our cut
                    landed inside flowing speech.
If the edit were the cause, connected_speech would score much higher than
real_pause. If both score about the same, the difference is how people
actually sound at the end of a sentence.

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
        "edit_point": [rms_db(last(30, 0)) - overall],
        "ending": [rms_db(last(a, a - 100)) - overall for a in (500, 400, 300, 200, 100)],
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

    # Edit-point loudness on its own, split by how the speaker behaved at the cut.
    edit = np.array([f["edit_point"][0] for f in feats])
    gap = np.array([int(r["gap_ms"]) for r in rows])
    complete = is_test & (labels == 1)
    by_cut_type = {}
    for name, mask in {
        "real_pause": gap >= cfg["real_pause_ms"],
        "connected_speech": gap <= cfg["connected_speech_ms"],
    }.items():
        subset = complete | (is_test & (labels == 0) & mask)
        auroc = roc_auc_score(labels[subset], -edit[subset])  # quieter ending = complete
        by_cut_type[name] = {
            "auroc": round(float(auroc), 4),
            "incomplete_clips": int((subset & (labels == 0)).sum()),
        }

    limit = cfg["max_auroc"]
    failed = [g for g in cfg["must_be_chance"] if abs(results[g] - 0.5) > limit - 0.5]
    report = {
        "test_clips": int(is_test.sum()),
        "auroc": results,
        "edit_point_by_cut_type": by_cut_type,
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
