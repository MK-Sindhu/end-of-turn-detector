"""Turn aligned utterances into end-of-turn training examples.

A live voice agent only has to decide "is the user finished?" at moments when
the user goes quiet. So every example here is one such moment, taken from a
real recording and left UNEDITED at the end:

  label 0 ("incomplete") - audio up to a natural pause inside the sentence,
                           plus the real silence that followed. The speaker
                           went on talking afterwards.
  label 1 ("complete")   - audio up to the true end of the sentence, plus the
                           real silence that followed.

Pauses at commas are included: the speaker paused, sounded as if a clause had
ended, and still carried on. Those are the hardest and most useful examples.

(An earlier version cut sentences at arbitrary word boundaries. The streaming
simulation showed that taught the model moments a live agent never faces.)

Examples come in matched pairs with the same number of words and the same
amount of trailing silence, so clip length cannot give the label away.
"""

from dataclasses import dataclass

import numpy as np

# Punctuation that marks a clause or sentence boundary.
BOUNDARY_PUNCTUATION = tuple(",;:.!?।|")


@dataclass
class Example:
    """One training example, described by word positions in the source clip."""

    label: int        # 1 = complete, 0 = incomplete
    first_word: int   # index of the first word included
    last_word: int    # index of the last word included
    tail_ms: int      # how much of the real silence after the last word is kept


def is_usable(record: dict, min_words: int, min_align_score: float) -> bool:
    """Keep a clip only if it is long enough and its alignment is trustworthy."""
    words = record["words"]
    if len(words) < min_words:
        return False
    mean_score = sum(w["score"] for w in words) / len(words)
    return mean_score >= min_align_score


def gap_after(record: dict, k: int) -> float:
    """Seconds of silence after word k (after the last word: until the file ends)."""
    words = record["words"]
    following = words[k + 1]["start"] if k + 1 < len(words) else record["duration"]
    return following - words[k]["end"]


def natural_pauses(record: dict, min_words: int, min_gap_s: float) -> list[int]:
    """Indices k where the speaker paused for at least min_gap_s after word k,
    with at least min_words spoken so far and at least one word still to come."""
    return [
        k
        for k in range(min_words - 1, len(record["words"]) - 1)
        if gap_after(record, k) >= min_gap_s
    ]


def plan_examples(
    record: dict,
    rng: np.random.Generator,
    max_pauses: int,
    min_words: int,
    min_gap_s: float,
    tail_ms_range: tuple[int, int],
    tail_guard_ms: int,
) -> list[Example]:
    """Decide which examples to make from one recording.

    For each chosen natural pause we make a matched pair:
      incomplete = n words ending at the pause
      complete   = the last n words of the sentence
    Both keep the same amount of real silence after the last word.
    """
    words = record["words"]
    last = len(words) - 1
    pauses = natural_pauses(record, min_words, min_gap_s)
    if len(pauses) > max_pauses:
        pauses = [int(k) for k in rng.choice(pauses, size=max_pauses, replace=False)]

    examples = []
    for k in sorted(pauses):
        # Keep a little less than the available silence, so the start of the
        # next word can never leak into the clip.
        available_ms = min(gap_after(record, k), gap_after(record, last)) * 1000 - tail_guard_ms
        if available_ms < tail_ms_range[0]:
            continue  # this recording ends too abruptly to give a fair pair
        tail_ms = int(rng.integers(tail_ms_range[0], min(tail_ms_range[1], int(available_ms)) + 1))
        n = int(rng.integers(min_words, k + 2))  # between min_words and k + 1 words
        examples.append(Example(0, k - n + 1, k, tail_ms))
        examples.append(Example(1, last - n + 1, last, tail_ms))
    return examples


def render(
    audio: np.ndarray,
    record: dict,
    example: Example,
    sr: int,
    max_seconds: float,
    lead_in_s: float = 0.05,
    fade_s: float = 0.015,
) -> np.ndarray:
    """Cut the example out of the source audio. The END is left untouched."""
    words = record["words"]
    start_s = max(0.0, words[example.first_word]["start"] - lead_in_s)
    end_s = words[example.last_word]["end"] + example.tail_ms / 1000

    clip = audio[int(start_s * sr) : int(end_s * sr)].copy()
    if example.first_word > 0:  # started mid-sentence: fade in so there is no click
        fade = min(int(fade_s * sr), len(clip))
        clip[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)

    # The model only ever sees the most recent max_seconds of audio.
    return clip[-int(max_seconds * sr) :].astype(np.float32)


def example_text(record: dict, example: Example) -> str:
    """The words spoken in this example (what a speech recogniser would have heard)."""
    words = record["words"][example.first_word : example.last_word + 1]
    return " ".join(w["word"] for w in words)


def ends_at_clause_boundary(record: dict, example: Example) -> bool:
    """Does the last word end with a comma or similar?"""
    return record["words"][example.last_word]["word"].endswith(BOUNDARY_PUNCTUATION)
