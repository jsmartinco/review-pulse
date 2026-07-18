"""Aspect-conditioned attention LSTM for three-class ABSA."""

import torch
from torch import nn


class ATAELSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 100, hidden_dim: int = 128, dropout: float = 0.5) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2 + embedding_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2 + embedding_dim, 3)

    def forward(self, review_tokens: torch.Tensor, aspect_tokens: torch.Tensor, return_attention: bool = False):
        review_embedding = self.dropout(self.embedding(review_tokens))
        aspect_embedding = self.embedding(aspect_tokens)
        aspect_mask = (aspect_tokens != 0).unsqueeze(-1)
        aspect_vector = (aspect_embedding * aspect_mask).sum(1) / aspect_mask.sum(1).clamp(min=1)
        repeated_aspect = aspect_vector.unsqueeze(1).expand(-1, review_tokens.size(1), -1)
        lengths = (review_tokens != 0).sum(1).cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(torch.cat((review_embedding, repeated_aspect), -1), lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(self.lstm(packed)[0], batch_first=True, total_length=review_tokens.size(1))
        scores = self.attention(torch.cat((encoded, repeated_aspect), -1)).squeeze(-1).masked_fill(review_tokens == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        context = (encoded * weights.unsqueeze(-1)).sum(1)
        logits = self.classifier(self.dropout(torch.cat((context, aspect_vector), -1)))
        return (logits, weights) if return_attention else logits
