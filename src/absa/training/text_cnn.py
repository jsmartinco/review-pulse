"""Train the optional review-only three-class TextCNN under the #91 protocol."""

import json
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import ABSA_OUTPUTS_DIR
from ..data.splits import split_official_data
from ..evaluation import compute_metrics
from ..labels import LABEL_TO_ID
from ..models.text_cnn import TextCNN
from ..tokenization.sequence import build_vocab, encode
from .common import (
    BestCheckpoint,
    build_run_result,
    checkpoint_metadata,
    seed_everything,
    validate_training_parameters,
)


def train_text_cnn(
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
    num_filters: int = 100,
    filter_widths: Sequence[int] = (3, 4, 5),
    dropout: float = 0.5,
    evaluate_official_test: bool = True,
):
    """Train a deterministic TextCNN selected only on development macro-F1."""
    validate_training_parameters(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_length=max_length,
        patience=patience,
    )
    widths = tuple(int(width) for width in filter_widths)
    if embedding_dim < 1 or num_filters < 1:
        raise ValueError("embedding_dim and num_filters must be positive")
    if not widths or any(width < 1 for width in widths):
        raise ValueError("filter_widths must contain positive integers")
    if len(set(widths)) != len(widths):
        raise ValueError("filter_widths must be unique")
    if max(widths) > max_length:
        raise ValueError("filter_widths cannot exceed max_length")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")

    loader_generator = seed_everything(seed)
    splits = split_official_data(train_rows, test_rows, seed=seed)
    vocab = build_vocab([row.review_raw for row in splits.train])
    model = TextCNN(
        len(vocab),
        embedding_dim=embedding_dim,
        num_filters=num_filters,
        filter_widths=widths,
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
        return compute_metrics(
            [row.label for row in rows],
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
    test = (
        score(splits.test)
        if evaluate_official_test
        else {"status": "not_evaluated_during_configuration_selection"}
    )
    config = {
        "model": "text_cnn",
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
        "num_filters": num_filters,
        "filter_widths": list(widths),
        "dropout": dropout,
        "pooling": "global_max",
        "padding": "right_pad_to_max_filter_width",
        "review_only": True,
        "configuration_selection": "development_macro_f1_only",
        "official_test_evaluated": evaluate_official_test,
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
    checkpoint_path = output_dir / "text_cnn.pt"
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
    (output_dir / "text_cnn_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
