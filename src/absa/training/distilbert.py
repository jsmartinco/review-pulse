"""Fine-tune the three-class ABSA DistilBERT sentence-pair model."""

import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..data.splits import split_official_data
from ..models.distilbert import ABSADistilBERT
from ..tokenization.bert_dataset import AspectPairDataset
from ..evaluation import compute_metrics
from ..labels import LABEL_TO_ID
from ..config import ABSA_OUTPUTS_DIR


def preferred_device() -> torch.device:
    """Prefer CUDA, then Apple Metal, while retaining CPU compatibility."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_distilbert(train_rows, test_rows, *, epochs: int = 2, batch_size: int = 8, device: torch.device | None = None):
    splits = split_official_data(train_rows, test_rows)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    active_device = device or preferred_device()
    model = ABSADistilBERT.from_pretrained_absa().to(active_device)
    loader = DataLoader(AspectPairDataset(tokenizer, splits.train), batch_size=batch_size, shuffle=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_function = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for batch in loader:
            labels = batch.pop("labels").to(active_device)
            features = {name: tensor.to(active_device) for name, tensor in batch.items()}
            optimiser.zero_grad()
            loss = loss_function(model(**features).logits, labels)
            loss.backward()
            optimiser.step()

    inverse = {value: key for key, value in LABEL_TO_ID.items()}

    def score(rows):
        model.eval()
        loader = DataLoader(AspectPairDataset(tokenizer, rows), batch_size=batch_size)
        predicted = []
        with torch.no_grad():
            for batch in loader:
                batch.pop("labels")
                features = {name: tensor.to(active_device) for name, tensor in batch.items()}
                predicted.extend(model(**features).logits.argmax(1).cpu().tolist())
        return compute_metrics([row.label for row in rows], [inverse[item] for item in predicted])

    return model, tokenizer, {"development": score(splits.development), "test": score(splits.test)}


def save_artifact(model, tokenizer, metrics, output_dir: Path = ABSA_OUTPUTS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "distilbert")
    tokenizer.save_pretrained(output_dir / "distilbert")
    (output_dir / "distilbert_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
