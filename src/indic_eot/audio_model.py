"""The audio end-of-turn model.

Shape of the model:

    waveform (last 8 s, 16 kHz)
      -> log-mel spectrogram          a picture of the sound: 80 pitch bands x 800 time steps
      -> Whisper encoder (pretrained) turns it into 400 vectors, one per 20 ms
      -> attention pooling            learns which moments matter and averages them
      -> small classifier head        one number: the logit for "turn is complete"

Whisper was trained on 680k hours of speech in ~100 languages (Hindi included),
so its encoder already knows a lot about how speech sounds. We reuse that
knowledge and only teach it the new question "has this person finished?".

Clips shorter than 8 s are padded with silence AT THE START, so the end of the
speech (the part that matters) always sits at the same place: the right edge.
"""

import numpy as np
import torch
from torch import nn
from transformers import WhisperFeatureExtractor, WhisperModel

FRAMES_PER_SECOND = 50  # the encoder outputs one vector per 20 ms


def left_pad(audio: np.ndarray, num_samples: int) -> tuple[np.ndarray, int]:
    """Keep the last num_samples of audio, padding with zeros on the left.

    Returns the padded audio and how many samples are real (not padding).
    """
    audio = audio[-num_samples:]
    padded = np.zeros(num_samples, dtype=np.float32)
    padded[num_samples - len(audio) :] = audio
    return padded, len(audio)


class FeaturePrep:
    """Turns raw waveforms into what the model expects."""

    def __init__(self, encoder_name: str, max_seconds: float, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.num_samples = int(max_seconds * sample_rate)
        self.num_frames = int(max_seconds * FRAMES_PER_SECOND)
        self.extractor = WhisperFeatureExtractor.from_pretrained(
            encoder_name, chunk_length=int(max_seconds)
        )

    def __call__(self, clips: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mel features [batch, 80, frames*2], mask [batch, frames]).

        The mask is True for frames that hold real audio and False for padding,
        so the model can ignore the padded part.
        """
        padded, real = zip(*(left_pad(c, self.num_samples) for c in clips))
        mel = self.extractor(
            list(padded), sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features
        mask = torch.zeros(len(clips), self.num_frames, dtype=torch.bool)
        for i, n in enumerate(real):
            real_frames = max(1, int(np.ceil(n / self.sample_rate * FRAMES_PER_SECOND)))
            mask[i, self.num_frames - real_frames :] = True
        return mel, mask


class AudioEOT(nn.Module):
    def __init__(self, encoder_name: str, max_seconds: float, dropout: float = 0.1):
        super().__init__()
        self.encoder = WhisperModel.from_pretrained(encoder_name).encoder

        # Whisper normally takes 30 s of audio (1500 positions). We only use
        # 8 s, so keep just the first 400 position vectors. Less to compute.
        positions = int(max_seconds * FRAMES_PER_SECOND)
        kept = self.encoder.embed_positions.weight.data[:positions].clone()
        self.encoder.embed_positions = nn.Embedding.from_pretrained(kept, freeze=False)
        self.encoder.config.max_source_positions = positions

        dim = self.encoder.config.d_model
        self.attention = nn.Linear(dim, 1)
        self.head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1)
        )

    def forward(self, mel: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return one logit per clip. sigmoid(logit) = P(turn complete)."""
        hidden = self.encoder(mel).last_hidden_state            # [batch, frames, dim]
        scores = self.attention(hidden).squeeze(-1)             # [batch, frames]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1).unsqueeze(-1)          # attention over time
        pooled = (hidden * weights).sum(dim=1)                  # [batch, dim]
        return self.head(pooled).squeeze(-1)
