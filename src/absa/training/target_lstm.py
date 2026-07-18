"""Train the controlled review-only three-class LSTM."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import ABSA_OUTPUTS_DIR
from ..data.splits import split_official_data
from ..evaluation import compute_metrics
from ..labels import LABEL_TO_ID
from ..models.target_lstm import TargetAgnosticLSTM
from ..tokenization.sequence import build_vocab, encode


def train_target_lstm(train_rows, test_rows, *, epochs: int = 8, batch_size: int = 64, seed: int = 42):
    torch.manual_seed(seed)
    splits = split_official_data(train_rows, test_rows, seed=seed)
    vocab = build_vocab([row.review_raw for row in splits.train])
    model = TargetAgnosticLSTM(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(
        TensorDataset(
            encode([row.review_raw for row in splits.train], vocab),
            torch.tensor([LABEL_TO_ID[row.label] for row in splits.train]),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    loss_function = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_function(model(x), y)
            loss.backward()
            optimizer.step()

    def score(rows):
        model.eval()
        with torch.no_grad():
            prediction = model(encode([row.review_raw for row in rows], vocab)).argmax(1).tolist()
        labels = [r.label for r in rows]
        inverse = {value: key for key, value in LABEL_TO_ID.items()}
        return compute_metrics(labels, [inverse[value] for value in prediction])
    return model, vocab, {"development": score(splits.development), "test": score(splits.test)}


def save_artifact(model, vocab, metrics, output_dir: Path = ABSA_OUTPUTS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "labels": list(LABEL_TO_ID)}, output_dir / "target_lstm.pt")
    (output_dir / "target_lstm_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
