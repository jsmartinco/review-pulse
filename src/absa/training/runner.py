"""Train and persist verified ReviewPulse v3 artifacts from one command."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..config import ABSA_DATA_DIR, ABSA_OUTPUTS_DIR
from ..data.parser import parse_aspect_examples
from .atae_lstm import save_artifact as save_atae_lstm
from .atae_lstm import train_atae_lstm
from .baseline import save_artifact as save_baseline
from .baseline import train_baseline
from .distilbert import preferred_device, save_artifact as save_distilbert
from .distilbert import train_distilbert
from .target_lstm import save_artifact as save_target_lstm
from .target_lstm import train_target_lstm


MODEL_ORDER = ("tfidf", "target_lstm", "atae_lstm", "distilbert")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _device(value: str) -> torch.device:
    if value == "auto":
        return preferred_device()
    device = torch.device(value)
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if value == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def _record(metrics: dict[str, object], provenance: dict[str, object]) -> dict[str, object]:
    metrics["provenance"] = provenance
    return metrics


def train_models(
    train_rows,
    test_rows,
    output_dir: Path,
    *,
    models: tuple[str, ...] = MODEL_ORDER,
    seed: int = 42,
    lstm_epochs: int = 8,
    distilbert_epochs: int = 2,
    recurrent_batch_size: int = 64,
    transformer_batch_size: int = 8,
    patience: int = 2,
    distilbert_device: torch.device | None = None,
    provenance: dict[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    """Train selected models and save only artifacts carrying the #91 record."""
    invalid = sorted(set(models) - set(MODEL_ORDER))
    if invalid:
        raise ValueError(f"Unknown training models: {invalid}")
    output_dir.mkdir(parents=True, exist_ok=True)
    common_provenance = provenance or {
        "git_commit": _git_commit(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    completed: dict[str, dict[str, object]] = {}

    if "tfidf" in models:
        model, metrics = train_baseline(train_rows, test_rows, seed=seed)
        completed["tfidf"] = _record(metrics, common_provenance)
        save_baseline(model, completed["tfidf"], output_dir)

    if "target_lstm" in models:
        model, vocab, metrics = train_target_lstm(
            train_rows,
            test_rows,
            epochs=lstm_epochs,
            batch_size=recurrent_batch_size,
            seed=seed,
            patience=patience,
        )
        completed["target_lstm"] = _record(metrics, common_provenance)
        save_target_lstm(model, vocab, completed["target_lstm"], output_dir)

    if "atae_lstm" in models:
        model, vocab, metrics = train_atae_lstm(
            train_rows,
            test_rows,
            epochs=lstm_epochs,
            batch_size=recurrent_batch_size,
            seed=seed,
            patience=patience,
        )
        completed["atae_lstm"] = _record(metrics, common_provenance)
        save_atae_lstm(model, vocab, completed["atae_lstm"], output_dir)

    if "distilbert" in models:
        model, tokenizer, metrics = train_distilbert(
            train_rows,
            test_rows,
            epochs=distilbert_epochs,
            batch_size=transformer_batch_size,
            seed=seed,
            patience=patience,
            device=distilbert_device or preferred_device(),
        )
        completed["distilbert"] = _record(metrics, common_provenance)
        save_distilbert(model, tokenizer, completed["distilbert"], output_dir)

    return completed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train verified ReviewPulse v3 artifacts on SemEval Restaurants."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ABSA_DATA_DIR / "restaurants",
    )
    parser.add_argument("--output-dir", type=Path, default=ABSA_OUTPUTS_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=list(MODEL_ORDER),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lstm-epochs", type=int, default=8)
    parser.add_argument("--distilbert-epochs", type=int, default=2)
    parser.add_argument("--recurrent-batch-size", type=int, default=64)
    parser.add_argument("--transformer-batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    train_path = args.data_dir / "restaurants_train.xml"
    test_path = args.data_dir / "restaurants_test.xml"
    provenance = {
        "git_commit": _git_commit(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_file": str(train_path),
        "train_sha256": _sha256(train_path),
        "test_file": str(test_path),
        "test_sha256": _sha256(test_path),
    }
    completed = train_models(
        parse_aspect_examples(train_path, "train"),
        parse_aspect_examples(test_path, "test"),
        args.output_dir,
        models=tuple(args.models),
        seed=args.seed,
        lstm_epochs=args.lstm_epochs,
        distilbert_epochs=args.distilbert_epochs,
        recurrent_batch_size=args.recurrent_batch_size,
        transformer_batch_size=args.transformer_batch_size,
        patience=args.patience,
        distilbert_device=_device(args.device),
        provenance=provenance,
    )
    summary = {
        key: {
            "training_seconds": metrics["training_seconds"],
            "best_epoch": metrics.get("best_epoch"),
            "test_accuracy": metrics["test"]["accuracy"],
            "test_macro_f1": metrics["test"]["macro_f1"],
        }
        for key, metrics in completed.items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
