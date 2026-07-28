"""Three-class review-only TextCNN used as an optional non-recurrent baseline."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class TextCNN(nn.Module):
    """Embed review tokens, pool convolution features and emit three logits."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        num_filters: int = 100,
        filter_widths: Sequence[int] = (3, 4, 5),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        widths = tuple(int(width) for width in filter_widths)
        if vocab_size < 2:
            raise ValueError("vocab_size must contain padding and unknown tokens")
        if embedding_dim < 1 or num_filters < 1:
            raise ValueError("embedding_dim and num_filters must be positive")
        if not widths or any(width < 1 for width in widths):
            raise ValueError("filter_widths must contain positive integers")
        if len(set(widths)) != len(widths):
            raise ValueError("filter_widths must be unique")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.filter_widths = widths
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, num_filters, kernel_size=width)
            for width in widths
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(widths), 3)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return logits after right-padding inputs shorter than the widest filter."""
        embedded = self.embedding(tokens).transpose(1, 2)
        missing = max(self.filter_widths) - embedded.shape[-1]
        if missing > 0:
            embedded = F.pad(embedded, (0, missing))
        pooled = [
            torch.relu(convolution(embedded)).amax(dim=2)
            for convolution in self.convolutions
        ]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))
