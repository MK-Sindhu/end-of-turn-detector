"""Decision points: the moments in a recording where a live agent must decide.

Shared by every script that scores a model for the streaming simulation, so
that our models and outside baselines are given exactly the same audio.
"""

import json
from collections.abc import Iterator

import numpy as np
import soundfile as sf

from indic_eot.config import REPO_ROOT
from indic_eot.dataset import BOUNDARY_PUNCTUATION, is_usable

ALIGNMENTS = REPO_ROOT / "data" / "interim" / "alignments.jsonl"


def decision_points(record: dict, min_silence_s: float) -> list[dict]:
    """Every natural pause of at least min_silence, plus the true end."""
    words = record["words"]
    points = []
    for k in range(len(words) - 1):
        gap = words[k + 1]["start"] - words[k]["end"]
        if gap >= min_silence_s:
            points.append(
                {
                    "kind": "pause",
                    "word_index": k,
                    "gap_ms": round(gap * 1000),
                    # Did the pause fall at a comma or similar (a clause boundary)?
                    "after_punct": int(words[k]["word"].endswith(BOUNDARY_PUNCTUATION)),
                }
            )
    trailing = record["duration"] - words[-1]["end"]
    points.append(
        {
            "kind": "end",
            "word_index": len(words) - 1,
            "gap_ms": round(trailing * 1000),
            "after_punct": 1,
        }
    )
    return points


def iter_decision_clips(params: dict) -> Iterator[tuple[dict, str, list[dict], list, list[str]]]:
    """For each val/test recording yield (record, split, points, clips, texts).

    clips[i] is the UNEDITED audio from the start of the recording up to
    points[i] plus min_silence of the silence that followed. texts[i] is the
    reference transcript up to that word.
    """
    sr = params["audio"]["sample_rate"]
    min_silence_s = params["streaming"]["min_silence_ms"] / 1000
    cfg = params["dataset"]
    split_of = {src: ours for ours, sources in cfg["splits"].items() for src in sources}

    with open(ALIGNMENTS, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    for record in records:
        split = split_of[record["split"]]
        if split not in ("val", "test"):
            continue
        if not is_usable(record, cfg["min_words"], cfg["min_align_score"]):
            continue

        audio, file_sr = sf.read(REPO_ROOT / record["path"], dtype="float32", always_2d=True)
        assert file_sr == sr, f"{record['path']} is {file_sr} Hz"
        audio = audio.mean(axis=1)
        words = record["words"]
        points = decision_points(record, min_silence_s)

        clips, texts = [], []
        for p in points:
            end = int((words[p["word_index"]]["end"] + min_silence_s) * sr)
            clip = audio[:end]
            if end > len(audio):  # recording stops early: the rest is silence
                clip = np.concatenate([clip, np.zeros(end - len(audio), dtype=np.float32)])
            clips.append(clip)
            texts.append(" ".join(w["word"] for w in words[: p["word_index"] + 1]))
        yield record, split, points, clips, texts
