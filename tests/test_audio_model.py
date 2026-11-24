import numpy as np

from indic_eot.audio_model import left_pad


def test_left_pad_puts_audio_at_the_right_edge():
    audio = np.ones(100, dtype=np.float32)
    padded, real = left_pad(audio, 400)
    assert real == 100
    assert padded[:300].sum() == 0
    assert padded[300:].sum() == 100


def test_left_pad_keeps_only_the_most_recent_audio():
    audio = np.arange(1000, dtype=np.float32)
    padded, real = left_pad(audio, 400)
    assert real == 400
    assert padded[0] == 600 and padded[-1] == 999
