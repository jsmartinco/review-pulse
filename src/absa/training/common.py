"""Shared reproducibility and checkpoint-selection utilities for ABSA trainers."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> torch.Generator:
    """Seed Python, NumPy and PyTorch and return a seeded loader generator."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return torch.Generator().manual_seed(seed)


def validate_training_parameters(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_length: int,
    patience: int = 2,
) -> None:
    """Fail early when a run configuration cannot produce a valid checkpoint."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    if patience < 1:
        raise ValueError("patience must be at least 1")


@dataclass
class BestCheckpoint:
    """Track and restore the model state with the best development score."""

    patience: int = 2
    min_delta: float = 0.0
    best_score: float = float("-inf")
    best_epoch: int | None = None
    epochs_without_improvement: int = 0
    _state_dict: dict[str, torch.Tensor] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")

    def update(self, model: torch.nn.Module, score: float, epoch: int) -> bool:
        """Capture a new best state and return whether training should stop."""
        if score > self.best_score + self.min_delta:
            self.best_score = float(score)
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            self._state_dict = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience

    def restore(self, model: torch.nn.Module) -> None:
        """Restore the selected state before official test evaluation or saving."""
        if self._state_dict is None or self.best_epoch is None:
            raise RuntimeError("No development checkpoint has been selected")
        model.load_state_dict(self._state_dict)


def training_diagnostic(history: list[dict[str, float | int]], threshold: float = 0.02) -> dict[str, Any]:
    """Record whether post-best behaviour warrants the multi-seed contingency."""
    if not history:
        raise ValueError("history must contain at least one epoch")
    best = max(history, key=lambda item: float(item["development_macro_f1"]))
    final = history[-1]
    score_drop = float(best["development_macro_f1"]) - float(final["development_macro_f1"])
    loss_decreased = float(final["train_loss"]) < float(best["train_loss"])
    material_overfitting = score_drop >= threshold and loss_decreased
    return {
        "material_overfitting_observed": material_overfitting,
        "development_macro_f1_drop_after_best": max(0.0, score_drop),
        "threshold": threshold,
        "multi_seed_recommended": material_overfitting,
        "decision": (
            "Run multiple seeds before final reporting because development performance "
            "materially declined while training loss improved."
            if material_overfitting
            else "Retain the fixed seed unless later runs show instability."
        ),
    }


def build_run_result(
    *,
    development: dict[str, object],
    test: dict[str, object],
    history: list[dict[str, float | int]],
    checkpoint: BestCheckpoint,
    config: dict[str, object],
    stopped_early: bool,
) -> dict[str, object]:
    """Build the JSON-safe training record shared by all neural models."""
    if checkpoint.best_epoch is None:
        raise RuntimeError("A best development epoch is required")
    return {
        "development": development,
        "test": test,
        "history": history,
        "config": config,
        "best_epoch": checkpoint.best_epoch,
        "selection_metric": "development_macro_f1",
        "best_development_macro_f1": checkpoint.best_score,
        "stopped_early": stopped_early,
        "overfitting_diagnostic": training_diagnostic(history),
    }


def checkpoint_metadata(run_result: dict[str, object]) -> dict[str, object]:
    """Select training metadata that must travel with a neural checkpoint."""
    keys = (
        "config",
        "history",
        "best_epoch",
        "selection_metric",
        "best_development_macro_f1",
        "stopped_early",
        "overfitting_diagnostic",
    )
    return {key: run_result[key] for key in keys}
