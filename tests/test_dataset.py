import numpy as np

from indic_eot.dataset import (
    Example,
    choose_cuts,
    cut_candidates,
    example_text,
    plan_examples,
    render,
)

SR = 16000


def make_record(n_words=10, word_s=0.3, gap_s=0.1, lead_s=0.5, trail_s=0.8):
    words, t = [], lead_s
    for i in range(n_words):
        words.append({"word": f"w{i}", "start": t, "end": t + word_s, "score": 0.9})
        t += word_s + gap_s
    return {"words": words, "duration": t - gap_s + trail_s}


def make_audio(record):
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.1, int(record["duration"] * SR)).astype(np.float32)


def test_cut_never_after_last_word_or_too_early():
    record = make_record()
    ks = cut_candidates(record["words"], min_words_before=3, min_gap_s=0.05)
    assert min(ks) >= 2          # at least 3 words spoken before the cut
    assert max(ks) <= 8          # at least one word left unsaid


def test_cut_skips_clause_boundaries():
    record = make_record()
    record["words"][4]["word"] = "w4,"
    assert 4 not in cut_candidates(record["words"], 3, 0.05)


def test_plan_is_balanced_and_labelled_correctly():
    record = make_record()
    examples = plan_examples(record, np.random.default_rng(1), 2, 3, 0.05, (150, 400))
    assert sum(e.label == 1 for e in examples) == 2
    assert sum(e.label == 0 for e in examples) == 2
    for e in examples:
        assert 150 <= e.tail_ms <= 400
        assert e.first_word >= 0
        assert e.last_word - e.first_word + 1 >= 3
        if e.label == 1:
            assert e.last_word == len(record["words"]) - 1
        else:
            assert e.last_word < len(record["words"]) - 1


def test_complete_and_incomplete_examples_have_matching_word_counts():
    record = make_record(n_words=20)
    examples = plan_examples(record, np.random.default_rng(3), 2, 3, 0.05, (150, 400))
    count = lambda e: e.last_word - e.first_word + 1
    assert sorted(count(e) for e in examples if e.label == 1) == sorted(
        count(e) for e in examples if e.label == 0
    )


def test_render_length_and_identical_tail_for_both_classes():
    record = make_record()
    audio = make_audio(record)
    complete = render(audio, record, Example(1, 0, 9, 200), SR, max_seconds=8.0)
    incomplete = render(audio, record, Example(0, 0, 5, 200), SR, max_seconds=8.0)

    # Speech from 0.45 s (50 ms lead-in) to word end + 30 ms, then a 200 ms tail.
    expected = (record["words"][5]["end"] + 0.03 - 0.45) + 0.2
    assert abs(len(incomplete) / SR - expected) < 0.01

    # Both clips end with the very same room tone, sample for sample.
    tail = int(0.2 * SR)
    np.testing.assert_array_equal(complete[-tail:], incomplete[-tail:])


def test_render_keeps_only_last_max_seconds():
    record = make_record(n_words=40)
    clip = render(make_audio(record), record, Example(1, 0, 39, 200), SR, max_seconds=8.0)
    assert len(clip) == 8 * SR


def test_example_text_is_the_spoken_prefix():
    record = make_record()
    assert example_text(record, Example(0, 2, 4, 200)) == "w2 w3 w4"


def test_choose_cuts_prefers_real_pauses_when_asked():
    record = make_record(n_words=20)
    words = record["words"]
    # Make one long pause, after word 7.
    for w in words[8:]:
        w["start"] += 0.5
        w["end"] += 0.5
    candidates = cut_candidates(words, 3, 0.05)
    always = choose_cuts(words, candidates, np.random.default_rng(0), 2, 0.25, pause_share=1.0)
    assert always[0] == 7                  # the pause is taken first
    assert len(set(always)) == 2           # the second pick falls back to another gap
