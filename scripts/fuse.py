"""Stage 8: combine the audio and text models (fusion).

Each model gives a probability. Fusion learns how much to trust each one:

    fused_logit = w_audio * logit(p_audio) + w_text * logit(p_text) + bias

That is a logistic regression with two inputs. It is fitted on the VALIDATION
predictions and scored on the TEST predictions, so the test split is never
used to tune anything.

Besides the usual metrics, this reports whether fusion is really better than
audio alone, using a paired bootstrap: resample the test recordings many
times and look at the spread of (fusion accuracy - audio accuracy). If the
95% interval excludes zero, the improvement is unlikely to be luck.

Outputs:
  models/fusion.json               the three learned numbers
  results/fusion_metrics.json      audio vs text vs fusion on the test split
  results/fusion_predictions.csv   fused P(complete) for val and test clips
"""

import csv
import json

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression

from indic_eot.config import REPO_ROOT, load_params
from indic_eot.evaluation import split_report
from indic_eot.training import load_rows, start_mlflow_run

RESULTS = REPO_ROOT / "results"
WEIGHTS_FILE = REPO_ROOT / "models" / "fusion.json"
METRICS_FILE = RESULTS / "fusion_metrics.json"
PREDICTIONS_FILE = RESULTS / "fusion_predictions.csv"


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def load_predictions(name: str) -> dict[str, float]:
    with open(RESULTS / f"{name}_predictions.csv") as f:
        return {r["id"]: float(r["prob"]) for r in csv.DictReader(f)}


def paired_difference(labels, probs_a, probs_b, groups, mask=None, n=2000, seed=0) -> dict:
    """Accuracy of B minus accuracy of A, with a 95% bootstrap interval.

    With a mask, accuracy is measured only on those clips (used for the
    clause-boundary pauses, where every clip is incomplete, so accuracy = recall).
    """
    correct_a = (probs_a >= 0.5) == labels
    correct_b = (probs_b >= 0.5) == labels
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        if mask[i]:
            by_group.setdefault(g, []).append(i)
    members = list(by_group.values())
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        diffs.append(correct_b[idx].mean() - correct_a[idx].mean())
    return {
        "difference": round(float(correct_b[mask].mean() - correct_a[mask].mean()), 4),
        "ci95": [
            round(float(np.percentile(diffs, 2.5)), 4),
            round(float(np.percentile(diffs, 97.5)), 4),
        ],
        "clips": int(mask.sum()),
    }


def main() -> None:
    params = load_params()
    audio, text = load_predictions("audio"), load_predictions("text")

    data = {}
    for split in ("val", "test"):
        rows = load_rows(split)
        data[split] = {
            "rows": rows,
            "labels": np.array([int(r["label"]) for r in rows]),
            "audio": np.array([audio[r["id"]] for r in rows]),
            "text": np.array([text[r["id"]] for r in rows]),
        }

    def features(d: dict) -> np.ndarray:
        return np.column_stack([logit(d["audio"]), logit(d["text"])])

    fusion = LogisticRegression().fit(features(data["val"]), data["val"]["labels"])
    weights = {
        "w_audio": round(float(fusion.coef_[0][0]), 4),
        "w_text": round(float(fusion.coef_[0][1]), 4),
        "bias": round(float(fusion.intercept_[0]), 4),
    }
    for d in data.values():
        d["fusion"] = fusion.predict_proba(features(d))[:, 1]

    test = data["test"]
    groups = [r["source_path"] for r in test["rows"]]
    at_boundary = np.array([r["after_punct"] == "1" for r in test["rows"]])
    boundary_pauses = (test["labels"] == 0) & at_boundary

    report = {
        "weights": weights,
        "test": {
            name: split_report(test["labels"], test[name], test["rows"])
            for name in ("audio", "text", "fusion")
        },
        "fusion_minus_audio": {
            "accuracy": paired_difference(test["labels"], test["audio"], test["fusion"], groups),
            "clause_boundary_recall": paired_difference(
                test["labels"], test["audio"], test["fusion"], groups, boundary_pauses
            ),
        },
        # How often the two models disagree, and who is right when they do.
        "disagreement": {},
    }
    a_right = (test["audio"] >= 0.5) == test["labels"]
    t_right = (test["text"] >= 0.5) == test["labels"]
    report["disagreement"] = {
        "both_right": int((a_right & t_right).sum()),
        "only_audio_right": int((a_right & ~t_right).sum()),
        "only_text_right": int((~a_right & t_right).sum()),
        "both_wrong": int((~a_right & ~t_right).sum()),
    }

    WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_FILE.write_text(json.dumps(weights, indent=2) + "\n")
    METRICS_FILE.write_text(json.dumps(report, indent=2) + "\n")
    with open(PREDICTIONS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "split", "label", "prob"])
        writer.writeheader()
        for split, d in data.items():
            for r, p in zip(d["rows"], d["fusion"]):
                writer.writerow(
                    {"id": r["id"], "split": split, "label": r["label"], "prob": f"{p:.5f}"}
                )

    with start_mlflow_run(params["mlflow"]["experiment"], run_name="fusion"):
        mlflow.log_params(weights)
        fused = report["test"]["fusion"]
        mlflow.log_metrics(
            {
                "test_accuracy": fused["accuracy"],
                "test_f1": fused["f1"],
                "test_auroc": fused["auroc"],
                "test_clause_boundary_recall": fused["incomplete_recall_clause_boundary"]["recall"],
            }
        )
        mlflow.log_artifact(str(METRICS_FILE))

    # A compact table for the terminal.
    print(f"\nweights: {weights}\n")
    print(
        f"{'model':8} {'accuracy':>9} {'auroc':>7} {'clause-boundary recall':>23} {'mid-clause recall':>18}"
    )
    for name, m in report["test"].items():
        print(
            f"{name:8} {m['accuracy']:9.4f} {m['auroc']:7.4f} "
            f"{m['incomplete_recall_clause_boundary']['recall']:23.4f} "
            f"{m['incomplete_recall_mid_clause']['recall']:18.4f}"
        )
    print(f"\nfusion - audio: {json.dumps(report['fusion_minus_audio'])}")
    print(f"disagreement:   {json.dumps(report['disagreement'])}")


if __name__ == "__main__":
    main()
