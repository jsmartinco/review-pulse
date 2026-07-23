import torch

from src.absa.models.atae_lstm import ATAELSTM


def test_atae_lstm_returns_three_logits_and_normalised_attention() -> None:
    model = ATAELSTM(30, embedding_dim=8, hidden_dim=4, dropout=0)
    logits, attention = model(torch.tensor([[1, 2, 3, 0]]), torch.tensor([[2, 0]]), return_attention=True)
    assert logits.shape == (1, 3)
    assert torch.allclose(attention.sum(1), torch.ones(1))
    assert attention[0, 3] == 0


def test_atae_lstm_accepts_different_aspects_for_the_same_review() -> None:
    torch.manual_seed(13)
    model = ATAELSTM(30, embedding_dim=8, hidden_dim=4, dropout=0)
    review = torch.tensor([[1, 2, 3, 0]])
    first_logits, first_attention = model(
        review, torch.tensor([[2, 0]]), return_attention=True
    )
    second_logits, second_attention = model(
        review, torch.tensor([[3, 0]]), return_attention=True
    )
    assert first_logits.shape == second_logits.shape == (1, 3)
    assert not torch.allclose(first_attention, second_attention)
