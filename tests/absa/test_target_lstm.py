import torch

from src.absa.models.target_lstm import TargetAgnosticLSTM


def test_target_agnostic_lstm_emits_three_logits_and_backpropagates() -> None:
    model = TargetAgnosticLSTM(vocab_size=30, embedding_dim=8, hidden_dim=4, dropout=0)
    logits = model(torch.tensor([[1, 2, 0], [3, 4, 5]]))
    assert logits.shape == (2, 3)
    torch.nn.CrossEntropyLoss()(logits, torch.tensor([0, 2])).backward()
    assert model.classifier.weight.grad is not None
