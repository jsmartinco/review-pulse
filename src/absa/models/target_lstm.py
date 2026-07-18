"""Three-class review-only LSTM used as the controlled ABSA comparator."""

import torch
from torch import nn


class TargetAgnosticLSTM(nn.Module):
    """Bidirectional LSTM that deliberately receives no aspect input."""

    def __init__(self, vocab_size: int, embedding_dim: int = 100, hidden_dim: int = 128, dropout: float = 0.5) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 3)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        lengths = torch.clamp((tokens != 0).sum(dim=1).cpu(), min=1)
        embedded = self.dropout(self.embedding(tokens))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1)))
