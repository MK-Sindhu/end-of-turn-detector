"""Stage: score an existing open detector on the same decision points.

Smart Turn (by Pipecat/Daily, BSD-2 licence) is an open audio end-of-turn model
with the same shape as ours: a Whisper-tiny encoder looking at the last 8 s.
It was trained on far more data, in 23 languages including Hindi.

Running it on exactly the audio our models saw answers the question a reader
will ask first: are our numbers good, or just numbers?

Not included: LiveKit's turn detector. Its licence only allows use inside the
LiveKit Agents framework, not standalone, so it is not run here.

Output: results/baseline_points.csv, one row per decision point.
"""

import csv

from tqdm import tqdm

from indic_eot.baselines import SmartTurn
from indic_eot.config import REPO_ROOT, load_params
from indic_eot.streaming import iter_decision_clips

OUT_FILE = REPO_ROOT / "results" / "baseline_points.csv"


def main() -> None:
    params = load_params()
    cfg = params["baselines"]["smart_turn"]
    assert params["audio"]["sample_rate"] == SmartTurn.SAMPLE_RATE
    model = SmartTurn(cfg["repo"], cfg["file"])

    rows = []
    for record, split, points, clips, _ in tqdm(iter_decision_clips(params), desc="smart-turn"):
        for point, clip in zip(points, clips):
            rows.append(
                {
                    "utterance": record["path"],
                    "split": split,
                    "kind": point["kind"],
                    "word_index": point["word_index"],
                    "p_smart_turn": f"{model.predict(clip):.5f}",
                }
            )

    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} decision points scored with {cfg['file']} -> {OUT_FILE.name}")


if __name__ == "__main__":
    main()
