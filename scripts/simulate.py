"""Stage 10: the streaming simulation. Turn model scores into agent behaviour.

Two ways an agent can decide the user has finished:

  silence timeout T   "respond once the user has been quiet for T ms."
      - cuts the user off at any mid-sentence pause that lasts T or longer
      - always responds T ms after the user really finishes

  model-gated         "after min_silence ms of quiet, ask the model. If
                       P(complete) >= threshold, respond now. Otherwise keep
                       waiting, and respond anyway once the quiet reaches the
                       fallback timeout."
      - cuts the user off at a pause if the model wrongly says complete, or
        if the pause outlasts the fallback
      - responds after min_silence when the model recognises the real end,
        after the fallback when it does not

For every recording we know each pause (how long, and what the model said) and
the true end, so both policies can be replayed exactly. Two numbers come out:

  false cut-off rate   share of recordings where the agent spoke too early
  response delay       quiet time between the user finishing and the agent
                       deciding to respond (recordings that were not cut off)

Lowering one normally raises the other. The plot shows that trade-off.

Two versions of the pauses are simulated:
  as_recorded        pause lengths exactly as in the recordings (read speech)
  with_hesitations   each natural pause is stretched to a length drawn from a
                     log-normal distribution, to mimic someone thinking
                     mid-sentence. This is an ASSUMPTION, set in params.yaml.

Outputs: results/streaming_metrics.json, results/streaming_tradeoff.png
"""

import csv
import hashlib
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from indic_eot.config import REPO_ROOT, load_params

POINTS_FILE = REPO_ROOT / "results" / "stream_points.csv"
METRICS_FILE = REPO_ROOT / "results" / "streaming_metrics.json"
PLOT_FILE = REPO_ROOT / "results" / "streaming_tradeoff.png"
MODELS = ("audio", "text", "fusion")


def load_utterances(split: str) -> list[dict]:
    """Group decision points by recording: {"pauses": [...], "end": {...}}."""
    grouped = defaultdict(lambda: {"pauses": [], "end": None})
    with open(POINTS_FILE) as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            point = {
                "word_index": int(row["word_index"]),
                "gap_ms": int(row["gap_ms"]),
                "after_punct": int(row["after_punct"]),
                **{m: float(row[f"p_{m}"]) for m in MODELS},
            }
            utt = grouped[row["utterance"]]
            utt["name"] = row["utterance"]
            if row["kind"] == "pause":
                utt["pauses"].append(point)
            else:
                utt["end"] = point
    return list(grouped.values())


def hesitation_ms(name: str, word_index: int, cfg: dict, seed: int) -> float:
    """A pause length drawn from a log-normal distribution. The draw depends
    only on the recording and position, so it is the same on every run."""
    digest = hashlib.sha256(f"{seed}:{name}:{word_index}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return float(np.exp(rng.normal(np.log(cfg["median_ms"]), cfg["sigma"])))


def pause_lengths(utt: dict, variant: str, cfg: dict, seed: int) -> list[float]:
    natural = [p["gap_ms"] for p in utt["pauses"]]
    if variant == "as_recorded":
        return natural
    return [
        max(gap, hesitation_ms(utt["name"], p["word_index"], cfg["hesitation"], seed))
        for gap, p in zip(natural, utt["pauses"])
    ]


def summarise(cut: np.ndarray, delay: np.ndarray) -> dict:
    """cut[i]: was recording i cut off. delay[i]: its response delay if not."""
    kept = delay[~cut]
    return {
        "false_cutoff_rate": round(float(cut.mean()), 4),
        "false_cutoffs": int(cut.sum()),
        "recordings": len(cut),
        "mean_delay_ms": round(float(kept.mean()), 1) if len(kept) else None,
        "median_delay_ms": round(float(np.median(kept)), 1) if len(kept) else None,
    }


def silence_policy(utts, lengths, timeout_ms) -> tuple[np.ndarray, np.ndarray]:
    cut = np.array([any(d >= timeout_ms for d in ds) for ds in lengths])
    return cut, np.full(len(utts), float(timeout_ms))


def gated_policy(utts, lengths, model, threshold, cfg) -> tuple[np.ndarray, np.ndarray]:
    min_silence, fallback = cfg["min_silence_ms"], cfg["fallback_ms"]
    cut = np.array(
        [
            any(
                d >= min_silence and (p[model] >= threshold or d >= fallback)
                for p, d in zip(u["pauses"], ds)
            )
            for u, ds in zip(utts, lengths)
        ]
    )
    delay = np.array(
        [float(min_silence) if u["end"][model] >= threshold else float(fallback) for u in utts]
    )
    return cut, delay


def bootstrap_rate_ci(cut: np.ndarray, n: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    rates = [cut[rng.integers(0, len(cut), len(cut))].mean() for _ in range(n)]
    return [round(float(np.percentile(rates, 2.5)), 4), round(float(np.percentile(rates, 97.5)), 4)]


def main() -> None:
    params = load_params()
    cfg = params["streaming"]
    seed = params["seed"]
    thresholds = np.round(np.arange(0.02, 0.99, 0.02), 2)
    utts = {split: load_utterances(split) for split in ("val", "test")}

    report = {
        "settings": cfg,
        "test_recordings": len(utts["test"]),
        "test_pauses": sum(len(u["pauses"]) for u in utts["test"]),
    }
    # Before any policy: how often is each model right at a single decision?
    report["per_decision"] = {}
    for model in MODELS:
        pauses = [p for u in utts["test"] for p in u["pauses"]]
        ends = [u["end"] for u in utts["test"]]

        def called_complete(points, model=model):
            return {
                "points": len(points),
                "called_complete": round(float(np.mean([p[model] >= 0.5 for p in points])), 4),
            }

        report["per_decision"][model] = {
            "ends": called_complete(ends),  # should be high
            "pauses": called_complete(pauses),  # should be low
            "pauses_after_punctuation": called_complete([p for p in pauses if p["after_punct"]]),
            "pauses_mid_clause": called_complete([p for p in pauses if not p["after_punct"]]),
        }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, variant in zip(axes, ("as_recorded", "with_hesitations")):
        lengths = {s: [pause_lengths(u, variant, cfg, seed) for u in utts[s]] for s in utts}
        result = {"silence_timeout": [], "model_gated": {}}

        # --- plain silence timeouts
        for timeout in cfg["silence_timeouts_ms"]:
            cut, delay = silence_policy(utts["test"], lengths["test"], timeout)
            result["silence_timeout"].append({"timeout_ms": timeout, **summarise(cut, delay)})

        # --- model-gated, one curve per model (sweeping the threshold)
        for model in MODELS:
            result["model_gated"][model] = [
                {
                    "threshold": float(t),
                    **summarise(*gated_policy(utts["test"], lengths["test"], model, t, cfg)),
                }
                for t in thresholds
            ]

        # --- model-gated at a few fixed thresholds, for a readable table
        result["operating_points"] = {
            model: [
                {
                    "threshold": t,
                    **summarise(
                        *(out := gated_policy(utts["test"], lengths["test"], model, t, cfg))
                    ),
                    "ci95": bootstrap_rate_ci(out[0]),
                }
                for t in cfg["report_thresholds"]
            ]
            for model in MODELS
        }
        for row in result["silence_timeout"]:
            cut, _ = silence_policy(utts["test"], lengths["test"], row["timeout_ms"])
            row["ci95"] = bootstrap_rate_ci(cut)
        report[variant] = result

        # --- plot
        xs = [r["mean_delay_ms"] for r in result["silence_timeout"]]
        ys = [100 * r["false_cutoff_rate"] for r in result["silence_timeout"]]
        ax.plot(xs, ys, "o-", color="black", label="silence timeout")
        for r, x, y in zip(result["silence_timeout"], xs, ys):
            ax.annotate(
                f"{r['timeout_ms']} ms",
                (x, y),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )
        for model, colour in zip(MODELS, ("tab:blue", "tab:orange", "tab:green")):
            curve = [r for r in result["model_gated"][model] if r["mean_delay_ms"] is not None]
            ax.plot(
                [r["mean_delay_ms"] for r in curve],
                [100 * r["false_cutoff_rate"] for r in curve],
                "-",
                color=colour,
                label=f"{model} model + {cfg['min_silence_ms']} ms",
            )
        ax.set_title(variant.replace("_", " "))
        ax.set_xlabel("mean response delay after the user finishes (ms)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("recordings where the agent cut the user off (%)")
    axes[0].legend()
    fig.suptitle(
        f"Hindi (FLEURS test, {report['test_recordings']} recordings, "
        f"{report['test_pauses']} natural pauses): lower-left is better"
    )
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150)

    METRICS_FILE.write_text(json.dumps(report, indent=2) + "\n")

    # A compact table for the terminal.
    print("\nPer decision point (test), share the model calls COMPLETE at threshold 0.5:")
    print(
        f"{'model':8} {'true ends':>10} {'all pauses':>11} {'clause-boundary pauses':>23} {'mid-clause pauses':>18}"
    )
    for model, d in report["per_decision"].items():
        print(
            f"{model:8} {100 * d['ends']['called_complete']:9.1f}% {100 * d['pauses']['called_complete']:10.1f}% "
            f"{100 * d['pauses_after_punctuation']['called_complete']:22.1f}% "
            f"{100 * d['pauses_mid_clause']['called_complete']:17.1f}%"
        )
    for variant in ("as_recorded", "with_hesitations"):
        print(f"\n=== {variant} (test, per recording) ===")
        print(f"{'policy':28} {'cut-off rate':>13} {'mean delay':>11}")
        for row in report[variant]["silence_timeout"]:
            print(
                f"{'silence ' + str(row['timeout_ms']) + ' ms':28} "
                f"{100 * row['false_cutoff_rate']:12.1f}% {row['mean_delay_ms']:8.0f} ms"
            )
        for model in MODELS:
            for row in report[variant]["operating_points"][model]:
                print(
                    f"{model + ' model, threshold ' + str(row['threshold']):28} "
                    f"{100 * row['false_cutoff_rate']:12.1f}% {row['mean_delay_ms']:8.0f} ms"
                )


if __name__ == "__main__":
    main()
