"""Reproduce the four-model ReviewPulse v3 comparison from verified artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch

from ..config import ABSA_DATA_DIR, ABSA_OUTPUTS_DIR
from ..data.parser import parse_aspect_examples
from ..data.schema import AspectExample
from ..data.splits import retained_examples
from ..labels import LABELS
from .artifact_evaluators import (
    LoadedEvaluator,
    UnverifiedArtifactError,
    load_artifact_evaluators,
)
from .metrics import compute_metrics
from .subsets import mixed_polarity_multi_aspect


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluate_models(
    rows: list[AspectExample],
    evaluators: list[LoadedEvaluator],
) -> tuple[dict[str, dict[str, object]], dict[str, list[str]], set[str]]:
    if not rows:
        raise ValueError("Official test evaluation requires retained examples")
    if {evaluator.key for evaluator in evaluators} != {
        "tfidf",
        "target_lstm",
        "atae_lstm",
        "distilbert",
    }:
        raise ValueError("The comparison requires exactly the four A2 model families")

    gold = [row.label for row in rows]
    mixed_ids = {row.sentence_id for row in mixed_polarity_multi_aspect(rows)}
    mixed_indices = [index for index, row in enumerate(rows) if row.sentence_id in mixed_ids]
    if not mixed_indices:
        raise ValueError("The official test set contains no mixed-polarity multi-aspect rows")

    results: dict[str, dict[str, object]] = {}
    predictions: dict[str, list[str]] = {}
    for evaluator in evaluators:
        first_started = perf_counter()
        first_prediction = evaluator.predict_batch(rows[:1])
        first_seconds = perf_counter() - first_started
        if len(first_prediction) != 1:
            raise ValueError(f"{evaluator.key} did not return one first prediction")

        warm_started = perf_counter()
        model_predictions = evaluator.predict_batch(rows)
        warm_seconds = perf_counter() - warm_started
        if len(model_predictions) != len(rows):
            raise ValueError(
                f"{evaluator.key} returned {len(model_predictions)} predictions for {len(rows)} rows"
            )
        invalid = sorted(set(model_predictions) - set(LABELS))
        if invalid:
            raise ValueError(f"{evaluator.key} returned invalid labels: {invalid}")

        predictions[evaluator.key] = model_predictions
        mixed_gold = [gold[index] for index in mixed_indices]
        mixed_predictions = [model_predictions[index] for index in mixed_indices]
        results[evaluator.key] = {
            "display_name": evaluator.display_name,
            "training": {
                "config": evaluator.training_config,
                "provenance": evaluator.provenance,
            },
            "full_test": compute_metrics(gold, model_predictions),
            "mixed_polarity_multi_aspect": compute_metrics(
                mixed_gold,
                mixed_predictions,
            ),
            "efficiency": {
                "training_seconds": evaluator.training_seconds,
                "artifact_bytes": evaluator.artifact_bytes,
                "artifact_megabytes": evaluator.artifact_bytes / (1024 * 1024),
                "device": evaluator.device,
                "load_seconds": evaluator.load_seconds,
                "first_prediction_ms": first_seconds * 1000,
                "cold_start_prediction_ms": (evaluator.load_seconds + first_seconds) * 1000,
                "warm_total_seconds": warm_seconds,
                "warm_latency_ms_per_example": warm_seconds * 1000 / len(rows),
                "warm_examples_per_second": len(rows) / warm_seconds,
            },
        }
    return results, predictions, mixed_ids


def _write_predictions(
    path: Path,
    rows: list[AspectExample],
    predictions: dict[str, list[str]],
    mixed_ids: set[str],
) -> None:
    model_keys = ["tfidf", "target_lstm", "atae_lstm", "distilbert"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sentence_id",
                "review",
                "aspect",
                "aspect_from",
                "aspect_to",
                "gold",
                "is_mixed_polarity_multi_aspect",
                *model_keys,
            ],
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "sentence_id": row.sentence_id,
                    "review": row.review_raw,
                    "aspect": row.aspect,
                    "aspect_from": row.aspect_from,
                    "aspect_to": row.aspect_to,
                    "gold": row.label,
                    "is_mixed_polarity_multi_aspect": row.sentence_id in mixed_ids,
                    **{key: predictions[key][index] for key in model_keys},
                }
            )


def _example(
    row: AspectExample,
    index: int,
    predictions: dict[str, list[str]],
) -> dict[str, object]:
    return {
        "sentence_id": row.sentence_id,
        "review": row.review_raw,
        "aspect": row.aspect,
        "gold": row.label,
        "predictions": {key: values[index] for key, values in predictions.items()},
    }


def _error_analysis(
    rows: list[AspectExample],
    predictions: dict[str, list[str]],
    mixed_ids: set[str],
    example_limit: int = 25,
) -> dict[str, object]:
    review_only = ("tfidf", "target_lstm")
    conditioned = ("atae_lstm", "distilbert")
    categories: dict[str, list[dict[str, object]]] = {
        "conditioned_wins_on_mixed_subset": [],
        "review_only_wins_on_mixed_subset": [],
        "all_models_wrong": [],
        "cross_model_disagreement": [],
    }
    counts = {key: 0 for key in categories}

    for index, row in enumerate(rows):
        correct_review = [predictions[key][index] == row.label for key in review_only]
        correct_conditioned = [predictions[key][index] == row.label for key in conditioned]
        predicted_labels = {predictions[key][index] for key in predictions}
        matched: list[str] = []
        if row.sentence_id in mixed_ids and any(correct_conditioned) and not any(correct_review):
            matched.append("conditioned_wins_on_mixed_subset")
        if row.sentence_id in mixed_ids and any(correct_review) and not any(correct_conditioned):
            matched.append("review_only_wins_on_mixed_subset")
        if not any(correct_review + correct_conditioned):
            matched.append("all_models_wrong")
        if len(predicted_labels) > 1:
            matched.append("cross_model_disagreement")

        for category in matched:
            counts[category] += 1
            if len(categories[category]) < example_limit:
                categories[category].append(_example(row, index, predictions))

    return {
        "counts": counts,
        "example_limit_per_category": example_limit,
        "examples": categories,
        "selection_note": (
            "Examples retain official-test order. They are analysis candidates, not cherry-picked claims."
        ),
    }


def _comparison_markdown(results: dict[str, dict[str, object]]) -> str:
    rows = [
        "| Model | Test accuracy | Test macro-F1 | Mixed accuracy | Mixed macro-F1 | Training s | Cold ms | Warm ms/example | Artifact MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("tfidf", "target_lstm", "atae_lstm", "distilbert"):
        result = results[key]
        full = result["full_test"]
        mixed = result["mixed_polarity_multi_aspect"]
        efficiency = result["efficiency"]
        training = efficiency["training_seconds"]
        training_text = f"{training:.2f}" if training is not None else "n/a"
        rows.append(
            "| {name} | {accuracy:.4f} | {macro:.4f} | {mixed_accuracy:.4f} | "
            "{mixed_macro:.4f} | {training} | {cold:.2f} | {warm:.3f} | {size:.2f} |".format(
                name=result["display_name"],
                accuracy=full["accuracy"],
                macro=full["macro_f1"],
                mixed_accuracy=mixed["accuracy"],
                mixed_macro=mixed["macro_f1"],
                training=training_text,
                cold=efficiency["cold_start_prediction_ms"],
                warm=efficiency["warm_latency_ms_per_example"],
                size=efficiency["artifact_megabytes"],
            )
        )
    return "\n".join(rows) + "\n"


def _plot_confusion_matrices(path: Path, results: dict[str, dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for axis, key in zip(axes.flat, ("tfidf", "target_lstm", "atae_lstm", "distilbert")):
        result = results[key]
        matrix = result["full_test"]["confusion_matrix"]
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(result["display_name"])
        axis.set_xticks(range(len(LABELS)), LABELS, rotation=25, ha="right")
        axis.set_yticks(range(len(LABELS)), LABELS)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Gold")
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                axis.text(column_index, row_index, str(value), ha="center", va="center")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("ReviewPulse v3 — official Restaurants test confusion matrices")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_evaluation(
    rows: list[AspectExample],
    evaluators: list[LoadedEvaluator],
    output_dir: Path,
    *,
    dataset_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate all models once and persist every A3 comparison artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results, predictions, mixed_ids = _evaluate_models(rows, evaluators)

    predictions_path = output_dir / "predictions.csv"
    _write_predictions(predictions_path, rows, predictions, mixed_ids)
    error_analysis = _error_analysis(rows, predictions, mixed_ids)
    (output_dir / "error_analysis.json").write_text(
        json.dumps(error_analysis, indent=2) + "\n",
        encoding="utf-8",
    )
    comparison = _comparison_markdown(results)
    (output_dir / "comparison.md").write_text(comparison, encoding="utf-8")
    _plot_confusion_matrices(output_dir / "confusion_matrices.png", results)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset": {
            "domain": "SemEval-2014 Task 4 Restaurants",
            "official_test_examples": len(rows),
            "mixed_polarity_examples": sum(row.sentence_id in mixed_ids for row in rows),
            "mixed_polarity_sentences": len(mixed_ids),
            **(dataset_metadata or {}),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "scikit_learn": version("scikit-learn"),
            "transformers": version("transformers"),
        },
        "measurement": {
            "clock": "time.perf_counter",
            "cold_start_prediction_ms": "artifact load plus first single-example prediction",
            "warm_latency_ms_per_example": "full official test batch after one warm-up prediction",
            "artifact_bytes": "recursive on-disk bytes for the loaded artifact",
        },
        "predictions_file": predictions_path.name,
        "predictions_sha256": _sha256(predictions_path),
        "models": results,
        "error_analysis_file": "error_analysis.json",
        "comparison_file": "comparison.md",
        "confusion_matrices_file": "confusion_matrices.png",
    }
    (output_dir / "results.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _device(value: str) -> torch.device | None:
    if value == "auto":
        return None
    device = torch.device(value)
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if value == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the four ReviewPulse v3 models on the official Restaurants test set."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ABSA_DATA_DIR / "restaurants",
    )
    parser.add_argument("--artifact-dir", type=Path, default=ABSA_OUTPUTS_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ABSA_OUTPUTS_DIR / "evaluation",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--recurrent-batch-size", type=int, default=64)
    parser.add_argument("--transformer-batch-size", type=int, default=16)
    parser.add_argument(
        "--allow-unverified-artifacts",
        action="store_true",
        help="Diagnostic only: allow pre-#91 artifacts with missing training metadata.",
    )
    args = parser.parse_args()

    test_path = args.data_dir / "restaurants_test.xml"
    rows = retained_examples(parse_aspect_examples(test_path, "test"))
    try:
        evaluators = load_artifact_evaluators(
            args.artifact_dir,
            require_verified=not args.allow_unverified_artifacts,
            recurrent_batch_size=args.recurrent_batch_size,
            transformer_batch_size=args.transformer_batch_size,
            device=_device(args.device),
        )
    except (FileNotFoundError, UnverifiedArtifactError) as error:
        parser.error(str(error))
    report = run_evaluation(
        rows,
        evaluators,
        args.output_dir,
        dataset_metadata={
            "test_file": str(test_path),
            "test_sha256": _sha256(test_path),
        },
    )
    print((args.output_dir / report["comparison_file"]).read_text())
    print(f"Wrote reproducible evaluation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
