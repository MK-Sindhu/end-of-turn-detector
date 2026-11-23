"""Stage 4: validate the dataset before any model is trained on it.

Cheap sanity checks that catch pipeline bugs early. If any check fails the
script exits with an error, which stops `dvc repro` before training starts.

Writes results/validation.json with the outcome of every check.
"""

import csv
import json
import sys
from collections import Counter, defaultdict

import soundfile as sf

from indic_eot.config import REPO_ROOT, load_params

METADATA = REPO_ROOT / "data" / "processed" / "eot" / "metadata.csv"
OUT_FILE = REPO_ROOT / "results" / "validation.json"


def main() -> None:
    params = load_params()
    sr = params["audio"]["sample_rate"]
    max_seconds = params["audio"]["max_seconds"]
    cfg = params["validation"]

    with open(METADATA, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    checks = {}

    def check(name: str, passed: bool, detail) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    # 1. Every split we expect exists and has data.
    splits = Counter(r["split"] for r in rows)
    check("all_splits_present", all(splits[s] > 0 for s in ("train", "val", "test")), dict(splits))

    # 2. No clip id appears twice.
    duplicates = [i for i, n in Counter(r["id"] for r in rows).items() if n > 1]
    check("unique_ids", not duplicates, f"{len(duplicates)} duplicated ids")

    # 3. Classes are balanced within every split and language.
    worst = 0.0
    groups = defaultdict(Counter)
    for r in rows:
        groups[(r["split"], r["lang"])][r["label"]] += 1
    for counts in groups.values():
        share = counts["1"] / (counts["0"] + counts["1"])
        worst = max(worst, abs(share - 0.5))
    check("class_balance", worst <= cfg["max_class_imbalance"], f"largest deviation from 50/50: {worst:.3f}")

    # 4. No sentence is shared between splits (otherwise the test set leaks).
    seen = defaultdict(set)
    for r in rows:
        seen[(r["lang"], r["sentence_id"])].add(r["split"])
    leaked = sum(1 for s in seen.values() if len(s) > 1)
    check("no_sentence_leak", leaked == 0, f"{leaked} sentences appear in more than one split")

    # 5. Every audio file exists, is mono 16 kHz, and has a sensible length.
    bad_files = []
    for r in rows:
        try:
            info = sf.info(REPO_ROOT / r["path"])
        except (RuntimeError, OSError):
            bad_files.append(r["id"])
            continue
        seconds = info.frames / info.samplerate
        ok = (
            info.samplerate == sr
            and info.channels == 1
            and cfg["min_seconds"] <= seconds <= max_seconds + 0.01
        )
        if not ok:
            bad_files.append(r["id"])
    check("audio_files_valid", not bad_files, f"{len(bad_files)} bad files; first few: {bad_files[:5]}")

    # 6. Every clip has a transcript (the text model needs it).
    empty = sum(1 for r in rows if not r["text"].strip())
    check("transcripts_present", empty == 0, f"{empty} clips with empty text")

    passed = all(c["passed"] for c in checks.values())
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({"passed": passed, "checks": checks}, indent=2) + "\n")

    for name, c in checks.items():
        print(f"{'PASS' if c['passed'] else 'FAIL'}  {name}: {c['detail']}")
    if not passed:
        sys.exit("Dataset validation failed.")


if __name__ == "__main__":
    main()
