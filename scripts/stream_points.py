"""Stage 9: score the models at every moment a live agent would have to decide.

Take a whole test recording, exactly as recorded. A live agent listening to it
would have to make a decision every time the speaker goes quiet:

  - at each natural pause inside the sentence (should keep waiting), and
  - at the true end of the sentence        (should respond).

For each such decision point we feed the models the UNEDITED audio from the
start of the recording up to that moment plus `min_silence_ms` of the silence
that followed, and record P(complete).

Unlike the training clips, nothing here is cut, faded or padded with room
tone. This is the closest thing to live use we can get from this corpus.

The text model gets the reference transcript up to that word. A live agent
would have a speech recogniser's transcript instead, with errors, so the text
and fusion results here are an upper bound.

Output: results/stream_points.csv, one row per decision point.
"""

import csv
import json

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from indic_eot.audio_model import AudioEOT, FeaturePrep
from indic_eot.config import REPO_ROOT, load_params
from indic_eot.dataset import BOUNDARY_PUNCTUATION, is_usable
from indic_eot.text_model import encode, load_model, load_tokenizer

ALIGNMENTS = REPO_ROOT / "data" / "interim" / "alignments.jsonl"
AUDIO_MODEL = REPO_ROOT / "models" / "audio" / "model.pt"
TEXT_MODEL = REPO_ROOT / "models" / "text"
FUSION = REPO_ROOT / "models" / "fusion.json"
OUT_FILE = REPO_ROOT / "results" / "stream_points.csv"


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


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


def main() -> None:
    params = load_params()
    sr = params["audio"]["sample_rate"]
    max_seconds = params["audio"]["max_seconds"]
    cfg = params["streaming"]
    min_silence_s = cfg["min_silence_ms"] / 1000
    dataset_cfg = params["dataset"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # FLEURS split name -> our split name; we only need val and test here.
    split_of = {src: ours for ours, sources in dataset_cfg["splits"].items() for src in sources}
    with open(ALIGNMENTS, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    records = [
        r
        for r in records
        if split_of[r["split"]] in ("val", "test")
        and is_usable(r, dataset_cfg["min_words"], dataset_cfg["min_align_score"])
    ]

    encoder = params["train_audio"]["encoder"]
    prep = FeaturePrep(encoder, max_seconds, sr)
    audio_model = AudioEOT(encoder, max_seconds, 0.0)
    audio_model.load_state_dict(torch.load(AUDIO_MODEL, map_location="cpu"))
    audio_model = audio_model.to(device).eval()
    tokenizer = load_tokenizer(TEXT_MODEL)
    text_model = load_model(TEXT_MODEL).to(device).eval()
    fusion = json.loads(FUSION.read_text())

    rows = []
    for record in tqdm(records, desc="scoring decision points"):
        audio, file_sr = sf.read(REPO_ROOT / record["path"], dtype="float32", always_2d=True)
        assert file_sr == sr, f"{record['path']} is {file_sr} Hz"
        audio = audio.mean(axis=1)
        words = record["words"]
        points = decision_points(record, min_silence_s)

        clips, texts = [], []
        for p in points:
            # Everything heard so far, up to min_silence after the word ended.
            end = int((words[p["word_index"]]["end"] + min_silence_s) * sr)
            clip = audio[:end]
            if end > len(audio):  # recording stops early: the rest is silence
                clip = np.concatenate([clip, np.zeros(end - len(audio), dtype=np.float32)])
            clips.append(clip)
            texts.append(" ".join(w["word"] for w in words[: p["word_index"] + 1]))

        with torch.inference_mode():
            mel, mask = prep(clips)
            p_audio = torch.sigmoid(audio_model(mel.to(device), mask.to(device))).cpu().numpy()
            inputs = encode(tokenizer, texts).to(device)
            p_text = torch.sigmoid(text_model(**inputs).logits.squeeze(-1)).cpu().numpy()
        fused = (
            fusion["w_audio"] * logit(p_audio) + fusion["w_text"] * logit(p_text) + fusion["bias"]
        )
        p_fusion = 1 / (1 + np.exp(-fused))

        for i, p in enumerate(points):
            rows.append(
                {
                    "utterance": record["path"],
                    "split": split_of[record["split"]],
                    **p,
                    "p_audio": f"{p_audio[i]:.5f}",
                    "p_text": f"{p_text[i]:.5f}",
                    "p_fusion": f"{p_fusion[i]:.5f}",
                }
            )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    pauses = sum(1 for r in rows if r["kind"] == "pause")
    print(
        f"{len(records)} recordings, {pauses} pauses, {len(rows) - pauses} ends -> {OUT_FILE.name}"
    )


if __name__ == "__main__":
    main()
