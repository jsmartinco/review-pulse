import torch

from src.absa.models.atae_lstm import ATAELSTM


def test_atae_lstm_returns_three_logits_and_normalised_attention() -> None:
    model = ATAELSTM(30, embedding_dim=8, hidden_dim=4, dropout=0)
    logits, attention = model(torch.tensor([[1, 2, 3, 0]]), torch.tensor([[2, 0]]), return_attention=True)
    assert logits.shape == (1, 3)
    assert torch.allclose(attention.sum(1), torch.ones(1))
    assert attention[0, 3] == 0


def test_atae_lstm_accepts_different_aspects_for_the_same_review() -> None:
    model = ATAELSTM(30, embedding_dim=8, hidden_dim=4, dropout=0)
    review = torch.tensor([[1, 2, 3, 0]])
    assert model(review, torch.tensor([[2, 0]])).shape == model(review, torch.tensor([[3, 0]])).shape == (1, 3)
