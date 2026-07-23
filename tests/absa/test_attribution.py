from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.absa.interpretability.attribution import (
    align_subword_scores,
    gradient_x_input_attribution,
)


def test_subword_scores_aggregate_to_exact_visible_tokens():
    result = align_subword_scores(
        "Unbelievable food!",
        offsets=[(0, 0), (0, 5), (5, 12), (13, 17), (17, 18), (0, 4)],
        sequence_ids=[None, 0, 0, 0, 0, 1],
        scores=[0.0, 0.2, 0.3, 0.4, 0.1, 99.0],
    )
    assert result == [
        {
            "token": "Unbelievable",
            "start": 0,
            "end": 12,
            "subword_count": 2,
            "score": pytest.approx(0.5),
        },
        {
            "token": "food",
            "start": 13,
            "end": 17,
            "subword_count": 1,
            "score": pytest.approx(0.4),
        },
        {
            "token": "!",
            "start": 17,
            "end": 18,
            "subword_count": 1,
            "score": pytest.approx(0.1),
        },
    ]


def test_subword_alignment_rejects_inconsistent_inputs():
    with pytest.raises(ValueError, match="equal length"):
        align_subword_scores("food", [(0, 4)], [0], [])


class _Encoding(dict):
    def sequence_ids(self, _batch_index):
        return [None, 0, 0, None, 1, None]


class _Tokenizer:
    def __call__(self, review, aspect, **kwargs):
        assert review == "Great food"
        assert aspect == "food"
        assert kwargs["return_offsets_mapping"] is True
        return _Encoding(
            input_ids=torch.tensor([[0, 1, 2, 3, 2, 3]]),
            attention_mask=torch.ones((1, 6), dtype=torch.long),
            offset_mapping=torch.tensor(
                [[[0, 0], [0, 5], [6, 10], [0, 0], [0, 4], [0, 0]]]
            ),
        )


class _TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.embedding = nn.Embedding(4, 5)
        self.classifier = nn.Linear(5, 3, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, attention_mask):
        masked = inputs_embeds * attention_mask.unsqueeze(-1)
        return SimpleNamespace(logits=self.classifier(masked.sum(dim=1)))


def test_gradient_x_input_is_deterministic_and_returns_aligned_review_tokens():
    model = _TinyClassifier().eval()
    first_logits, first_tokens = gradient_x_input_attribution(
        model, _Tokenizer(), "Great food", "food"
    )
    second_logits, second_tokens = gradient_x_input_attribution(
        model, _Tokenizer(), "Great food", "food"
    )
    assert torch.equal(first_logits, second_logits)
    assert first_tokens == second_tokens
    assert [item["token"] for item in first_tokens] == ["Great", "food"]
    assert sum(item["score"] for item in first_tokens) == pytest.approx(1.0)
