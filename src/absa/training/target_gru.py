"""Train the optional review-only three-class GRU under the #91 protocol."""

import json
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import ABSA_OUTPUTS_DIR
from ..data.splits import split_official_data
from ..evaluation import compute_metrics
from ..labels import LABEL_TO_ID
from ..models.target_gru import TargetAgnosticGRU
from ..tokenization.sequence import build_vocab, encode
from .common import (
    BestCheckpoint,
    build_run_result,
    checkpoint_metadata,
    seed_everything,
    validate_training_parameters,
)


def train_target_gru(
    train_rows,
    test_rows,
    *,
    epochs: int = 8,
    batch_size: int = 64,
    seed: int = 42,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 2,
    max_length: int = 80,
    embedding_dim: int = 100,
    hidden_dim: int = 128,
    dropout: float = 0.5,
):
    """Train a deterministic GRU matched to the target-agnostic LSTM budget."""
    validate_training_parameters(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_length=max_length,
        patience=patience,
    )
    if embedding_dim < 1 or hidden_dim < 1:
        raise ValueError("embedding_dim and hidden_dim must be positive")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")

    loader_generator = seed_everything(seed)
    splits = split_official_data(train_rows, test_rows, seed=seed)
    vocab = build_vocab([row.review_raw for row in splits.train])
    model = TargetAgnosticGRU(
        len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loader = DataLoader(
        TensorDataset(
            encode(
                [row.review_raw for row in splits.train],
                vocab,
                max_length,
            ),
            torch.tensor([LABEL_TO_ID[row.label] for row in splits.train]),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    loss_function = torch.nn.CrossEntropyLoss()
    inverse = {value: key for key, value in LABEL_TO_ID.items()}

    def score(rows):
        model.eval()
        with torch.no_grad():
            prediction = (
                model(
                    encode(
                        [row.review_raw for row in rows],
                        vocab,
                        max_length,
                    )
                )
                .argmax(1)
                .tolist()
            )
        labels = [row.label for row in rows]
        return compute_metrics(
            labels,
            [inverse[value] for value in prediction],
        )

    checkpoint = BestCheckpoint(patience=patience)
    history: list[dict[str, float | int]] = []
    stopped_early = False
    training_started = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for review_tokens, labels in loader:
            optimizer.zero_grad()
            loss = loss_function(model(review_tokens), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        development = score(splits.development)
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / len(loader),
                "development_macro_f1": float(development["macro_f1"]),
            }
        )
        if checkpoint.update(
            model,
            float(development["macro_f1"]),
            epoch,
        ):
            stopped_early = epoch < epochs
            break
    training_seconds = perf_counter() - training_started

    checkpoint.restore(model)
    development = score(splits.development)
    test = score(splits.test)
    config = {
        "model": "target_gru",
        "seed": seed,
        "device": "cpu",
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "weight_decay": weight_decay,
        "patience": patience,
        "max_length": max_length,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "bidirectional": True,
        "recurrent_layers": 1,
        "controlled_against": "target_lstm",
        "deliberate_differences": [
            "GRU recurrent cell replaces LSTM recurrent and cell state gates",
            "Parameter count differs because GRU has three gates instead of four",
        ],
    }
    result = build_run_result(
        development=development,
        test=test,
        history=history,
        checkpoint=checkpoint,
        config=config,
        stopped_early=stopped_early,
        training_seconds=training_seconds,
    )
    result["parameter_count"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    return model, vocab, result


def save_artifact(
    model,
    vocab,
    metrics,
    output_dir: Path = ABSA_OUTPUTS_DIR,
) -> None:
    """Persist the clean-load checkpoint and complete JSON run record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "target_gru.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "labels": list(LABEL_TO_ID),
            "parameter_count": metrics["parameter_count"],
            **checkpoint_metadata(metrics),
        },
        checkpoint_path,
    )
    metrics["artifact_bytes"] = checkpoint_path.stat().st_size
    metrics["artifact_megabytes"] = metrics["artifact_bytes"] / (1024 * 1024)
    (output_dir / "target_gru_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
