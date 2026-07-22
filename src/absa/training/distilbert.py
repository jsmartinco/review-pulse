import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..config import ABSA_OUTPUTS_DIR
from ..data.splits import split_official_data
from ..evaluation import compute_metrics
from ..labels import LABEL_TO_ID
from ..models.distilbert import ABSADistilBERT
from ..tokenization.bert_dataset import AspectPairDataset
from .common import (
    BestCheckpoint,
    build_run_result,
    checkpoint_metadata,
    seed_everything,
    validate_training_parameters,
)


def preferred_device() -> torch.device:
    """Prefer CUDA, then Apple Metal, while retaining CPU compatibility."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_distilbert(
    train_rows,
    test_rows,
    *,
    epochs: int = 2,
    batch_size: int = 8,
    seed: int = 42,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    patience: int = 2,
    max_length: int = 128,
    model_name: str = "distilbert-base-uncased",
    device: torch.device | None = None,
):
    validate_training_parameters(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_length=max_length,
        patience=patience,
    )
    loader_generator = seed_everything(seed)
    splits = split_official_data(train_rows, test_rows, seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    active_device = device or preferred_device()
    model = ABSADistilBERT.from_pretrained_absa(model_name).to(active_device)
    loader = DataLoader(
        AspectPairDataset(tokenizer, splits.train, max_length=max_length),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = torch.nn.CrossEntropyLoss()

    inverse = {value: key for key, value in LABEL_TO_ID.items()}

    def score(rows):
        model.eval()
        evaluation_loader = DataLoader(
            AspectPairDataset(tokenizer, rows, max_length=max_length),
            batch_size=batch_size,
        )
        predicted = []
        with torch.no_grad():
            for batch in evaluation_loader:
                batch.pop("labels")
                features = {name: tensor.to(active_device) for name, tensor in batch.items()}
                predicted.extend(model(**features).logits.argmax(1).cpu().tolist())
        return compute_metrics([row.label for row in rows], [inverse[item] for item in predicted])

    checkpoint = BestCheckpoint(patience=patience)
    history: list[dict[str, float | int]] = []
    stopped_early = False
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            labels = batch.pop("labels").to(active_device)
            features = {name: tensor.to(active_device) for name, tensor in batch.items()}
            optimiser.zero_grad()
            loss = loss_function(model(**features).logits, labels)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.detach().cpu().item()
        development = score(splits.development)
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / len(loader),
                "development_macro_f1": float(development["macro_f1"]),
            }
        )
        if checkpoint.update(model, float(development["macro_f1"]), epoch):
            stopped_early = epoch < epochs
            break

    checkpoint.restore(model)
    development = score(splits.development)
    test = score(splits.test)
    config = {
        "model": "distilbert",
        "pretrained_model": model_name,
        "seed": seed,
        "device": str(active_device),
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "AdamW",
        "weight_decay": weight_decay,
        "patience": patience,
        "max_length": max_length,
    }
    return model, tokenizer, build_run_result(
        development=development,
        test=test,
        history=history,
        checkpoint=checkpoint,
        config=config,
        stopped_early=stopped_early,
    )


def save_artifact(model, tokenizer, metrics, output_dir: Path = ABSA_OUTPUTS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "distilbert"
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    (checkpoint_dir / "training_run.json").write_text(
        json.dumps(checkpoint_metadata(metrics), indent=2) + "\n"
    )
    (output_dir / "distilbert_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
