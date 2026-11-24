"""Stage 3: build the end-of-turn dataset from the aligned clips.

Reads   data/interim/alignments.jsonl
Writes  data/processed/eot/<split>/<id>.wav   16 kHz mono clips
        data/processed/eot/metadata.csv       one row per clip (label, text, ...)
        results/dataset_stats.json            counts, for a quick sanity check

Every example is one moment where a live agent would have to decide: a natural
pause inside a sentence (incomplete) or the true end (complete), unedited.
See src/indic_eot/dataset.py for the details.
"""

import csv
import hashlib
import json
from collections import Counter
from itertools import combinations

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F
from tqdm import tqdm

from indic_eot.config import REPO_ROOT, load_params
from indic_eot.dataset import (
    ends_at_clause_boundary,
    example_text,
    gap_after,
    is_usable,
    plan_examples,
    render,
)

ALIGNMENTS = REPO_ROOT / "data" / "interim" / "alignments.jsonl"
OUT_DIR = REPO_ROOT / "data" / "processed" / "eot"
STATS_FILE = REPO_ROOT / "results" / "dataset_stats.json"


def clip_rng(seed: int, path: str) -> np.random.Generator:
    """A random generator that depends only on the seed and the clip, so the
    dataset comes out identical no matter what order clips are processed in."""
    digest = hashlib.sha256(f"{seed}:{path}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def load_audio(path, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = F.resample(torch.from_numpy(audio), sr, sample_rate).numpy()
    return audio


def main() -> None:
    params = load_params()
    sr = params["audio"]["sample_rate"]
    max_seconds = params["audio"]["max_seconds"]
    cfg = params["dataset"]

    with open(ALIGNMENTS, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    # FLEURS split name -> our split name (set in params.yaml).
    split_of = {src: ours for ours, sources in cfg["splits"].items() for src in sources}

    # The same sentence must never appear in two splits, or the test would leak.
    sentences = {}
    for r in records:
        sentences.setdefault(split_of[r["split"]], set()).add((r["lang"], r["sentence_id"]))
    for (a, in_a), (b, in_b) in combinations(sentences.items(), 2):
        if in_a & in_b:
            raise SystemExit(f"{len(in_a & in_b)} sentences appear in both {a} and {b}")

    rows, dropped = [], 0
    for record in tqdm(records, desc="building examples"):
        if not is_usable(record, cfg["min_words"], cfg["min_align_score"]):
            dropped += 1
            continue
        rng = clip_rng(params["seed"], record["path"])
        examples = plan_examples(
            record,
            rng,
            max_pauses=cfg["max_pauses_per_clip"],
            min_words=cfg["min_words_in_example"],
            min_gap_s=cfg["min_gap_ms"] / 1000,
            tail_ms_range=tuple(cfg["tail_ms_range"]),
            tail_guard_ms=cfg["tail_guard_ms"],
        )
        if not examples:
            dropped += 1
            continue

        audio = load_audio(REPO_ROOT / record["path"], sr)
        split = split_of[record["split"]]
        source_id = record["path"].split("/")[-1].removesuffix(".wav")
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

        for i, ex in enumerate(examples):
            clip = render(audio, record, ex, sr, max_seconds)
            # How long the speaker was really silent after this point.
            gap_ms = round(gap_after(record, ex.last_word) * 1000)
            clip_id = f"{record['lang']}_{source_id}_{ex.label}_{i}"
            path = OUT_DIR / split / f"{clip_id}.wav"
            sf.write(path, clip, sr, subtype="PCM_16")
            rows.append(
                {
                    "id": clip_id,
                    "split": split,
                    "lang": record["lang"],
                    "label": ex.label,
                    "path": str(path.relative_to(REPO_ROOT)),
                    "text": example_text(record, ex),
                    "seconds": round(len(clip) / sr, 3),
                    "tail_ms": ex.tail_ms,
                    "gap_ms": gap_ms,
                    "after_punct": int(ends_at_clause_boundary(record, ex)),
                    "sentence_id": record["sentence_id"],
                    "source_path": record["path"],
                }
            )

    with open(OUT_DIR / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((r["split"], r["lang"], r["label"]) for r in rows)
    stats = {
        "source_clips": len(records),
        "source_clips_dropped": dropped,
        "examples": len(rows),
        "hours": round(sum(r["seconds"] for r in rows) / 3600, 2),
        "by_split_lang_label": {
            f"{split}/{lang}/{'complete' if label else 'incomplete'}": n
            for (split, lang, label), n in sorted(counts.items())
        },
    }
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
