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
import torch
from tqdm import tqdm

from indic_eot.audio_model import AudioEOT, FeaturePrep
from indic_eot.config import REPO_ROOT, load_params
from indic_eot.streaming import iter_decision_clips
from indic_eot.text_model import encode, load_model, load_tokenizer

AUDIO_MODEL = REPO_ROOT / "models" / "audio" / "model.pt"
TEXT_MODEL = REPO_ROOT / "models" / "text"
FUSION = REPO_ROOT / "models" / "fusion.json"
OUT_FILE = REPO_ROOT / "results" / "stream_points.csv"


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def main() -> None:
    params = load_params()
    sr = params["audio"]["sample_rate"]
    max_seconds = params["audio"]["max_seconds"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    encoder = params["train_audio"]["encoder"]
    prep = FeaturePrep(encoder, max_seconds, sr)
    audio_model = AudioEOT(encoder, max_seconds, 0.0)
    audio_model.load_state_dict(torch.load(AUDIO_MODEL, map_location="cpu"))
    audio_model = audio_model.to(device).eval()
    tokenizer = load_tokenizer(TEXT_MODEL)
    text_model = load_model(TEXT_MODEL).to(device).eval()
    fusion = json.loads(FUSION.read_text())

    rows = []
    for record, split, points, clips, texts in tqdm(
        iter_decision_clips(params), desc="scoring decision points"
    ):
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
                    "split": split,
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
    print(f"{pauses} pauses, {len(rows) - pauses} ends -> {OUT_FILE.name}")


if __name__ == "__main__":
    main()
