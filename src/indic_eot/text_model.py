"""The text end-of-turn model.

The audio model hears HOW something was said. This one reads WHAT was said:
"मुझे दो टिकट चाहिए" is a finished thought, "मुझे दो टिकट" is not, whatever the
voice did. It is a small multilingual transformer (MiniLM, pretrained on ~100
languages) with a classification head.

In a live agent the text comes from a speech recogniser, which gives no
reliable punctuation while the person is still talking. So punctuation is
stripped before the model sees anything. That also removes a giveaway: in our
data every complete sentence would otherwise end with "।" or ".".
"""

import re
import unicodedata

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_TOKENS = 64


def normalize_text(text: str) -> str:
    """Lowercase, drop all punctuation, collapse spaces."""
    kept = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return re.sub(r"\s+", " ", kept).strip().lower()


def load_tokenizer(name_or_path: str):
    return AutoTokenizer.from_pretrained(name_or_path)


def load_model(name_or_path: str) -> torch.nn.Module:
    """A transformer with one output: the logit for "turn is complete"."""
    return AutoModelForSequenceClassification.from_pretrained(name_or_path, num_labels=1)


def encode(tokenizer, texts: list[str]) -> dict[str, torch.Tensor]:
    """Text -> token ids. If a transcript is too long we keep its END, because
    the last few words decide whether the thought is finished."""
    tokenizer.truncation_side = "left"
    return tokenizer(
        [normalize_text(t) for t in texts],
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    )
