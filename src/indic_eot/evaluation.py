"""Scoring a model's predictions on one split, the same way for every model."""

import numpy as np

from indic_eot.training import bootstrap_ci, classification_metrics


def split_report(labels: np.ndarray, probs: np.ndarray, rows: list[dict]) -> dict:
    """Metrics, confidence intervals, and a breakdown of the incomplete clips.

    The breakdown asks: of the pauses where the speaker was NOT finished, how
    many does the model correctly call incomplete?
      clause_boundary  the pause fell at a comma or similar. The words so far
                       often form a complete thought, so this is the hard case.
      mid_clause       the pause fell in the middle of a clause.
    """
    report = {
        "clips": len(labels),
        **{k: round(v, 4) for k, v in classification_metrics(labels, probs).items()},
        "ci95": bootstrap_ci(labels, probs, [r["source_path"] for r in rows]),
    }
    at_boundary = np.array([r["after_punct"] == "1" for r in rows])
    for name, mask in {"clause_boundary": at_boundary, "mid_clause": ~at_boundary}.items():
        subset = (labels == 0) & mask
        report[f"incomplete_recall_{name}"] = {
            "clips": int(subset.sum()),
            "recall": round(float((probs[subset] < 0.5).mean()), 4),
        }
    return report
