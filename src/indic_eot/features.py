"""Audio features for the exported model, using numpy only.

Training uses the Whisper feature extractor from `transformers`. A deployed
model should not need that library (it pulls in PyTorch), so this file redoes
the same calculation in plain numpy. A test checks the two agree.

The calculation, in order:
  1. cut the audio into overlapping 25 ms windows, one every 10 ms
  2. measure how much energy each window has at each frequency (FFT)
  3. group the frequencies into 80 bands spaced the way hearing works (mel)
  4. take the logarithm and rescale
"""

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400  # 25 ms window
HOP = 160  # 10 ms step
FRAMES_PER_SECOND = 50  # the encoder halves the 100 mel frames per second


def log_mel(audio: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """Return the Whisper log-mel spectrogram, shape [80, len(audio) // 160].

    mel_filters has shape [201, 80] and is saved next to the exported model.
    """
    window = np.hanning(N_FFT + 1)[:-1]
    padded = np.pad(audio.astype(np.float64), N_FFT // 2, mode="reflect")
    num_frames = 1 + (len(padded) - N_FFT) // HOP
    index = np.arange(N_FFT)[None, :] + HOP * np.arange(num_frames)[:, None]
    power = np.abs(np.fft.rfft(padded[index] * window, axis=1)) ** 2
    mel = power[:-1] @ mel_filters  # Whisper drops the final frame
    log = np.log10(np.maximum(mel, 1e-10))
    log = np.maximum(log, log.max() - 8.0)
    return ((log + 4.0) / 4.0).T.astype(np.float32)


def prepare(
    audio: np.ndarray, mel_filters: np.ndarray, max_seconds: float
) -> tuple[np.ndarray, np.ndarray]:
    """Turn one waveform into the model's two inputs.

    Keeps the last max_seconds of audio and pads shorter audio with silence at
    the start. Returns (mel [1, 80, frames * 2], mask [1, frames]); the mask is
    True where the frame holds real audio.
    """
    num_samples = int(max_seconds * SAMPLE_RATE)
    num_frames = int(max_seconds * FRAMES_PER_SECOND)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)[-num_samples:]
    padded = np.zeros(num_samples, dtype=np.float32)
    padded[num_samples - len(audio) :] = audio

    mask = np.zeros((1, num_frames), dtype=bool)
    real_frames = max(1, int(np.ceil(len(audio) / SAMPLE_RATE * FRAMES_PER_SECOND)))
    mask[0, num_frames - real_frames :] = True
    return log_mel(padded, mel_filters)[None], mask
