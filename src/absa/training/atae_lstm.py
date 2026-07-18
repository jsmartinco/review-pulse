"""Train ATAE-LSTM with the same recurrent budget as the review-only LSTM."""

import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from ..data.splits import split_official_data
from ..evaluation import compute_metrics
from ..labels import LABEL_TO_ID
from ..models.atae_lstm import ATAELSTM
from ..tokenization.sequence import build_vocab, encode
from ..config import ABSA_OUTPUTS_DIR


def train_atae_lstm(train_rows, test_rows, *, epochs: int = 8, batch_size: int = 64, seed: int = 42):
    torch.manual_seed(seed)
    splits = split_official_data(train_rows, test_rows, seed=seed)
    vocab = build_vocab([row.review_raw for row in splits.train])
    model = ATAELSTM(len(vocab))
    loader = DataLoader(TensorDataset(encode([r.review_raw for r in splits.train], vocab), encode([r.aspect for r in splits.train], vocab, 12), torch.tensor([LABEL_TO_ID[r.label] for r in splits.train])), batch_size=batch_size, shuffle=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for review, aspect, label in loader:
            optimiser.zero_grad(); loss=torch.nn.CrossEntropyLoss()(model(review, aspect), label); loss.backward(); optimiser.step()
    inverse={value:key for key,value in LABEL_TO_ID.items()}
    def score(rows):
        model.eval()
        with torch.no_grad(): predicted=model(encode([r.review_raw for r in rows],vocab),encode([r.aspect for r in rows],vocab,12)).argmax(1).tolist()
        return compute_metrics([r.label for r in rows],[inverse[x] for x in predicted])
    return model, vocab, {"development":score(splits.development),"test":score(splits.test)}


def save_artifact(model, vocab, metrics, output_dir: Path = ABSA_OUTPUTS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "labels": list(LABEL_TO_ID)}, output_dir / "atae_lstm.pt")
    (output_dir / "atae_lstm_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
