"""Turn aligned utterances into end-of-turn training examples.

Each example is a short audio clip plus a label:
  label 1 ("complete")   - the clip ends where the speaker really finished.
  label 0 ("incomplete") - the clip ends between two words, mid-sentence.

The important rule: both kinds of clip must END THE SAME WAY acoustically.
Every clip is cut just after a word, faded out, and followed by the same kind
of "room tone" (background hiss taken from that recording's own silence).
Otherwise a model could cheat by spotting the edit instead of listening to
the speech.
"""

from dataclasses import dataclass

import numpy as np

# Punctuation that marks a clause or sentence boundary. We never create an
# "incomplete" example by cutting right after one of these, because a speaker
# could plausibly have finished there and the label would be ambiguous.
BOUNDARY_PUNCTUATION = tuple(",;:.!?।|")


@dataclass
class Example:
    """One training example, described by word positions in the source clip."""

    label: int        # 1 = complete, 0 = incomplete
    first_word: int   # index of the first word included
    last_word: int    # index of the last word included
    tail_ms: int      # length of silence after the speech


def is_usable(record: dict, min_words: int, min_align_score: float) -> bool:
    """Keep a clip only if it is long enough and its alignment is trustworthy."""
    words = record["words"]
    if len(words) < min_words:
        return False
    mean_score = sum(w["score"] for w in words) / len(words)
    return mean_score >= min_align_score


def cut_candidates(words: list[dict], min_words_before: int, min_gap_s: float) -> list[int]:
    """Indices k where cutting after word k gives a valid "incomplete" example.

    Rules: enough words before the cut, at least one word left unsaid, the word
    does not end a clause, and there is a real gap before the next word.
    If no position has a big enough gap, fall back to the largest gap available.
    """
    allowed = [
        k
        for k in range(min_words_before - 1, len(words) - 1)
        if not words[k]["word"].endswith(BOUNDARY_PUNCTUATION)
    ]

    def gap(k: int) -> float:
        return words[k + 1]["start"] - words[k]["end"]

    with_gap = [k for k in allowed if gap(k) >= min_gap_s]
    if with_gap:
        return with_gap
    return [max(allowed, key=gap)] if allowed else []


def plan_examples(
    record: dict,
    rng: np.random.Generator,
    per_class: int,
    min_words_before: int,
    min_gap_s: float,
    tail_ms_range: tuple[int, int],
) -> list[Example]:
    """Decide which complete and incomplete examples to make from one clip."""
    words = record["words"]
    candidates = cut_candidates(words, min_words_before, min_gap_s)
    if not candidates:
        return []

    def random_tail() -> int:
        return int(rng.integers(tail_ms_range[0], tail_ms_range[1] + 1))

    # Examples are made in matched pairs with the SAME number of words:
    #   incomplete = n words ending at the cut point
    #   complete   = the last n words of the sentence
    # Without this, complete clips would be longer on average and a model could
    # just learn "long clip = finished" instead of listening to how it ends.
    examples = []
    last = len(words) - 1
    chosen = rng.choice(candidates, size=min(per_class, len(candidates)), replace=False)
    for k in sorted(int(k) for k in chosen):
        n = int(rng.integers(min_words_before, k + 2))  # between the minimum and k + 1 words
        examples.append(Example(0, k - n + 1, k, random_tail()))
        examples.append(Example(1, last - n + 1, last, random_tail()))
    return examples


def room_tone(audio: np.ndarray, words: list[dict], sr: int) -> np.ndarray:
    """Background noise from the recording's own silence (after or before speech)."""
    after = audio[int((words[-1]["end"] + 0.15) * sr) :]
    before = audio[: max(0, int((words[0]["start"] - 0.10) * sr))]
    tone = after if len(after) >= len(before) else before
    if len(tone) < int(0.1 * sr):  # too little silence in this clip: use faint noise
        tone = np.random.default_rng(0).normal(0, 1e-4, sr).astype(np.float32)
    return tone


def render(
    audio: np.ndarray,
    record: dict,
    example: Example,
    sr: int,
    max_seconds: float,
    margin_s: float = 0.03,
    lead_in_s: float = 0.05,
    fade_s: float = 0.015,
) -> np.ndarray:
    """Cut the example out of the source audio and give it the standard ending."""
    words = record["words"]
    start_s = max(0.0, words[example.first_word]["start"] - lead_in_s)

    # Cut a little after the last included word, but never into the next word.
    end_s = words[example.last_word]["end"] + margin_s
    if example.last_word + 1 < len(words):
        gap_middle = (words[example.last_word]["end"] + words[example.last_word + 1]["start"]) / 2
        end_s = min(end_s, gap_middle)
    end_s = min(end_s, len(audio) / sr)

    speech = audio[int(start_s * sr) : int(end_s * sr)].copy()
    fade = min(int(fade_s * sr), len(speech))
    speech[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)  # no click at the start
    tail_len = int(example.tail_ms / 1000 * sr)

    # Room tone long enough for the tail plus the cross-fade region.
    tone = room_tone(audio, words, sr)
    tone = np.tile(tone, int(np.ceil((tail_len + fade) / len(tone))))[: tail_len + fade].copy()

    # Cross-fade: speech fades out while room tone fades in over the same samples.
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    speech[-fade:] = speech[-fade:] * (1 - ramp) + tone[:fade] * ramp
    clip = np.concatenate([speech, tone[fade:]])

    # The model only ever sees the most recent max_seconds of audio.
    return clip[-int(max_seconds * sr) :].astype(np.float32)


def example_text(record: dict, example: Example) -> str:
    """The words spoken in this example (what a speech recogniser would have heard)."""
    words = record["words"][example.first_word : example.last_word + 1]
    return " ".join(w["word"] for w in words)
