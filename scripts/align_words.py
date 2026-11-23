"""Stage 2: find when each word is spoken in each clip (forced alignment).

We already know WHAT was said (the transcript). This stage works out WHEN each
word was said, so that later we can cut a clip exactly between two words.

How it works:
  1. The MMS forced-alignment model listens to the clip and, for every 20 ms
     frame, gives the probability of each Latin letter being spoken.
  2. The transcript is romanised (Devanagari -> Latin letters) with uroman,
     because that one model covers 1000+ languages through Latin letters.
  3. A dynamic-programming search (forced_align) finds the best way to lay the
     transcript's letters along the frames. Grouping letters back into words
     gives a start and end time for every word.

Output: data/interim/alignments.jsonl, one JSON object per clip:
  {"lang", "split", "sentence_id", "path", "text", "duration",
   "words": [{"word", "start", "end", "score"}, ...]}
"""

import argparse
import csv
import json
import re

import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
import uroman as ur
from tqdm import tqdm

from indic_eot.config import REPO_ROOT, load_params

MANIFEST = REPO_ROOT / "data" / "raw" / "fleurs" / "manifest.csv"
OUT_FILE = REPO_ROOT / "data" / "interim" / "alignments.jsonl"

# FLEURS language code -> ISO 639-3 code that uroman expects.
UROMAN_CODES = {"hi_in": "hin", "ta_in": "tam", "te_in": "tel", "bn_in": "ben", "en_in": "eng"}


def romanise_words(text: str, lang: str, romaniser: ur.Uroman) -> tuple[list[str], list[str]]:
    """Split text into words; return (original words, romanised a-z versions).

    A word with no letters after romanisation (a number, a symbol) becomes "*",
    a wildcard token the aligner can match against any sound.
    """
    originals, romans = [], []
    for word in text.split():
        roman = romaniser.romanize_string(word, lcode=UROMAN_CODES.get(lang))
        roman = re.sub(r"[^a-z']", "", roman.lower())
        originals.append(word)
        romans.append(roman or "*")
    return originals, romans


def align_clip(waveform, romans, model, dictionary, device):
    """Return [(start_frame, end_frame, score)] for each word, plus frame count."""
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
    emission = emission.cpu()

    tokens = [dictionary[ch] for word in romans for ch in word]
    targets = torch.tensor([tokens], dtype=torch.int32)
    alignment, scores = F.forced_align(emission, targets, blank=0)
    spans = F.merge_tokens(alignment[0], scores[0].exp())  # one span per letter

    words, i = [], 0
    for word in romans:
        letters = spans[i : i + len(word)]
        i += len(word)
        score = sum(s.score * (s.end - s.start) for s in letters) / max(
            1, sum(s.end - s.start for s in letters)
        )
        words.append((letters[0].start, letters[-1].end, float(score)))
    return words, emission.size(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only align the first N clips")
    args = parser.parse_args()

    sample_rate = load_params()["audio"]["sample_rate"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=True).to(device).eval()
    dictionary = bundle.get_dict(star="*")
    romaniser = ur.Uroman()

    with open(MANIFEST, encoding="utf-8") as f:
        clips = list(csv.DictReader(f))[: args.limit]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    with open(OUT_FILE, "w", encoding="utf-8") as out:
        for clip in tqdm(clips, desc=f"aligning on {device}"):
            audio, sr = sf.read(REPO_ROOT / clip["path"], dtype="float32", always_2d=True)
            waveform = torch.from_numpy(audio.mean(axis=1)).unsqueeze(0)  # (1, samples), mono
            if sr != sample_rate:
                waveform = F.resample(waveform, sr, sample_rate)

            originals, romans = romanise_words(clip["text"], clip["lang"], romaniser)
            try:
                spans, num_frames = align_clip(waveform, romans, model, dictionary, device)
            except (RuntimeError, ValueError, KeyError) as err:  # e.g. transcript too long for the audio
                skipped += 1
                tqdm.write(f"skipped {clip['path']}: {err}")
                continue

            seconds_per_frame = waveform.size(1) / num_frames / sample_rate
            record = {
                "lang": clip["lang"],
                "split": clip["split"],
                "sentence_id": clip["sentence_id"],
                "path": clip["path"],
                "text": clip["text"],
                "duration": round(waveform.size(1) / sample_rate, 3),
                "words": [
                    {
                        "word": word,
                        "start": round(start * seconds_per_frame, 3),
                        "end": round(end * seconds_per_frame, 3),
                        "score": round(score, 3),
                    }
                    for word, (start, end, score) in zip(originals, spans)
                ],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Aligned {len(clips) - skipped} clips, skipped {skipped}. Wrote {OUT_FILE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
