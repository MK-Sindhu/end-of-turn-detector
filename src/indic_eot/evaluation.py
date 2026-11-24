"""Scoring a model's predictions on one split, the same way for every model."""

import numpy as np

from indic_eot.training import bootstrap_ci, classification_metrics


def split_report(labels: np.ndarray, probs: np.ndarray, rows: list[dict], params: dict) -> dict:
    """Metrics, confidence intervals, and the real-pause breakdown.

    The breakdown asks: of the incomplete clips, how many does the model catch
    when the speaker genuinely paused there, versus when the next word followed
    straight away? Real pauses are what a live agent actually faces.
    """
    report = {
        "clips": len(labels),
        **{k: round(v, 4) for k, v in classification_metrics(labels, probs).items()},
        "ci95": bootstrap_ci(labels, probs, [r["source_path"] for r in rows]),
    }
    gap = np.array([int(r["gap_ms"]) for r in rows])
    check = params["shortcut_check"]
    for name, mask in {
        "real_pause": gap >= check["real_pause_ms"],
        "connected_speech": gap <= check["connected_speech_ms"],
    }.items():
        subset = (labels == 0) & mask
        report[f"incomplete_recall_{name}"] = {
            "clips": int(subset.sum()),
            "recall": round(float((probs[subset] < 0.5).mean()), 4),
        }
    return report
