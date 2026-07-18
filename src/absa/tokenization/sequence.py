"""Minimal leakage-safe sequence vocabulary for v3 recurrent models."""

import re
from collections import Counter

import torch

PAD, UNK = "<pad>", "<unk>"


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def build_vocab(texts: list[str], min_frequency: int = 1) -> dict[str, int]:
    counts = Counter(token for text in texts for token in tokens(text))
    return {PAD: 0, UNK: 1, **{token: index + 2 for index, token in enumerate(sorted(token for token, count in counts.items() if count >= min_frequency))}}


def encode(texts: list[str], vocab: dict[str, int], max_length: int = 80) -> torch.Tensor:
    rows = []
    for text in texts:
        row = [vocab.get(token, vocab[UNK]) for token in tokens(text)][:max_length]
        rows.append(row + [vocab[PAD]] * (max_length - len(row)))
    return torch.tensor(rows, dtype=torch.long)
