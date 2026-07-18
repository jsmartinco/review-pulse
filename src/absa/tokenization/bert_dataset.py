"""Torch dataset for encoded ABSA review/aspect pairs."""

import torch
from torch.utils.data import Dataset

from ..labels import LABEL_TO_ID
from .transformer import encode_review_aspect_pairs


class AspectPairDataset(Dataset):
    def __init__(self, tokenizer, rows, max_length: int = 128) -> None:
        self.encoded = encode_review_aspect_pairs(tokenizer, [row.review_raw for row in rows], [row.aspect for row in rows], max_length=max_length)
        self.labels = torch.tensor([LABEL_TO_ID[row.label] for row in rows], dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.encoded.items()} | {"labels": self.labels[index]}
