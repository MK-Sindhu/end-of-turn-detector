import numpy as np

from indic_eot.dataset import (
    Example,
    ends_at_clause_boundary,
    example_text,
    gap_after,
    natural_pauses,
    plan_examples,
    render,
)

SR = 16000


def make_record(n_words=12, word_s=0.3, gap_s=0.05, lead_s=0.5, trail_s=0.8, pauses=None):
    """A fake aligned recording. `pauses` maps word index -> gap after that word."""
    pauses = pauses or {}
    words, t = [], lead_s
    for i in range(n_words):
        words.append({"word": f"w{i}", "start": t, "end": t + word_s, "score": 0.9})
        t += word_s + pauses.get(i, gap_s)
    duration = words[-1]["end"] + trail_s
    return {"words": words, "duration": duration}


def make_audio(record):
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.1, int(record["duration"] * SR)).astype(np.float32)


def plan(record, seed=0, max_pauses=4):
    return plan_examples(record, np.random.default_rng(seed), max_pauses, 2, 0.2, (150, 400), 50)


def test_natural_pauses_finds_only_real_gaps():
    record = make_record(pauses={3: 0.4, 7: 0.25})
    assert natural_pauses(record, min_words=2, min_gap_s=0.2) == [3, 7]


def test_natural_pauses_needs_enough_words_and_a_word_to_come():
    record = make_record(pauses={0: 0.5, 11: 0.5})  # too early, and after the last word
    assert natural_pauses(record, min_words=2, min_gap_s=0.2) == []


def test_gap_after_last_word_is_the_trailing_silence():
    record = make_record(trail_s=0.8)
    assert abs(gap_after(record, 11) - 0.8) < 1e-9


def test_examples_come_in_matched_pairs():
    record = make_record(pauses={3: 0.4, 7: 0.6})
    examples = plan(record)
    incomplete = [e for e in examples if e.label == 0]
    complete = [e for e in examples if e.label == 1]
    assert len(incomplete) == len(complete) == 2
    for neg, pos in zip(incomplete, complete):
        assert neg.last_word in (3, 7)                       # ends at a natural pause
        assert pos.last_word == 11                           # ends at the true end
        assert neg.last_word - neg.first_word == pos.last_word - pos.first_word
        assert neg.tail_ms == pos.tail_ms
        assert neg.first_word >= 0 and pos.first_word >= 0


def test_tail_never_reaches_the_next_word():
    record = make_record(pauses={5: 0.22})  # a 220 ms pause, with a 50 ms guard
    for seed in range(20):
        for e in plan(record, seed):
            assert 150 <= e.tail_ms <= 170


def test_recording_without_pauses_gives_no_examples():
    assert plan(make_record()) == []


def test_max_pauses_is_respected():
    record = make_record(pauses={2: 0.4, 4: 0.4, 6: 0.4, 8: 0.4, 9: 0.4})
    assert len(plan(record, max_pauses=3)) == 6


def test_render_keeps_the_real_audio_after_the_last_word():
    record = make_record(pauses={5: 0.4})
    audio = make_audio(record)
    clip = render(audio, record, Example(0, 0, 5, 300), SR, max_seconds=8.0)
    end = int((record["words"][5]["end"] + 0.3) * SR)
    start = int((record["words"][0]["start"] - 0.05) * SR)
    assert len(clip) == end - start
    np.testing.assert_array_equal(clip, audio[start:end])    # byte-for-byte unedited


def test_render_keeps_only_last_max_seconds():
    record = make_record(n_words=40)
    clip = render(make_audio(record), record, Example(1, 0, 39, 200), SR, max_seconds=8.0)
    assert len(clip) == 8 * SR


def test_example_text_and_clause_boundary():
    record = make_record()
    record["words"][4]["word"] = "w4,"
    example = Example(0, 2, 4, 200)
    assert example_text(record, example) == "w2 w3 w4,"
    assert ends_at_clause_boundary(record, example)
    assert not ends_at_clause_boundary(record, Example(0, 2, 5, 200))
