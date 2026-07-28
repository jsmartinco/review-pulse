"""Reproduce canonical four-model or exploratory six-model comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
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
from ..training.provenance import file_sha256, git_commit
from .artifact_evaluators import (
    CORE_MODEL_ORDER,
    SIX_MODEL_ORDER,
    LoadedEvaluator,
    UnverifiedArtifactError,
    load_artifact_evaluators,
)
from .metrics import compute_metrics
from .subsets import mixed_polarity_multi_aspect


def _evaluate_models(
    rows: list[AspectExample],
    evaluators: list[LoadedEvaluator],
    model_keys: tuple[str, ...],
) -> tuple[dict[str, dict[str, object]], dict[str, list[str]], set[str]]:
    if not rows:
        raise ValueError("Official test evaluation requires retained examples")
    if tuple(evaluator.key for evaluator in evaluators) != model_keys:
        raise ValueError(
            f"The comparison requires evaluators in this order: {model_keys}"
        )

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
                "parameter_count": evaluator.parameter_count,
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
    model_keys: tuple[str, ...],
) -> None:
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
    model_keys: tuple[str, ...],
    example_limit: int = 25,
) -> dict[str, object]:
    review_only = tuple(
        key
        for key in ("tfidf", "target_lstm", "target_gru", "text_cnn")
        if key in model_keys
    )
    conditioned = tuple(
        key for key in ("atae_lstm", "distilbert") if key in model_keys
    )
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
        "model_groups": {
            "review_only": list(review_only),
            "aspect_conditioned": list(conditioned),
        },
        "counts": counts,
        "example_limit_per_category": example_limit,
        "examples": categories,
        "selection_note": (
            "Examples retain official-test order. They are analysis candidates, not cherry-picked claims."
        ),
    }


def _comparison_markdown(
    results: dict[str, dict[str, object]],
    model_keys: tuple[str, ...],
) -> str:
    rows = [
        "| Model | Scope | Test accuracy | Test macro-F1 | Mixed accuracy | Mixed macro-F1 | Training s | Cold ms | Warm ms/example | Throughput ex/s | Parameters | Artifact MB | Device |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in model_keys:
        result = results[key]
        full = result["full_test"]
        mixed = result["mixed_polarity_multi_aspect"]
        efficiency = result["efficiency"]
        training = efficiency["training_seconds"]
        training_text = f"{training:.2f}" if training is not None else "n/a"
        parameters = efficiency["parameter_count"]
        parameters_text = f"{parameters:,}" if parameters is not None else "n/a"
        rows.append(
            "| {name} | {scope} | {accuracy:.4f} | {macro:.4f} | {mixed_accuracy:.4f} | "
            "{mixed_macro:.4f} | {training} | {cold:.2f} | {warm:.3f} | "
            "{throughput:.2f} | {parameters} | {size:.2f} | {device} |".format(
                name=result["display_name"],
                scope="exploratory" if key in {"target_gru", "text_cnn"} else "A2 core",
                accuracy=full["accuracy"],
                macro=full["macro_f1"],
                mixed_accuracy=mixed["accuracy"],
                mixed_macro=mixed["macro_f1"],
                training=training_text,
                cold=efficiency["cold_start_prediction_ms"],
                warm=efficiency["warm_latency_ms_per_example"],
                throughput=efficiency["warm_examples_per_second"],
                parameters=parameters_text,
                size=efficiency["artifact_megabytes"],
                device=efficiency["device"],
            )
        )

    for heading, result_key in (
        ("Full-test per-class evidence", "full_test"),
        ("Mixed-polarity per-class evidence", "mixed_polarity_multi_aspect"),
    ):
        first_per_class = results[model_keys[0]][result_key]["per_class"]
        supports = {
            label: int(first_per_class[label]["support"])
            for label in LABELS
        }
        rows.extend(
            [
                "",
                f"### {heading}",
                "",
                "Cells report precision / recall / F1.",
                "",
                "| Model | "
                + " | ".join(
                    f"{label.title()} (n={supports[label]})"
                    for label in LABELS
                )
                + " |",
                "|---|" + "---:|" * len(LABELS),
            ]
        )
        for key in model_keys:
            per_class = results[key][result_key]["per_class"]
            cells = [
                "{precision:.4f} / {recall:.4f} / {f1:.4f}".format(
                    precision=per_class[label]["precision"],
                    recall=per_class[label]["recall"],
                    f1=per_class[label]["f1-score"],
                )
                for label in LABELS
            ]
            rows.append(
                f"| {results[key]['display_name']} | "
                + " | ".join(cells)
                + " |"
            )
    return "\n".join(rows) + "\n"


def _plot_confusion_matrices(
    path: Path,
    results: dict[str, dict[str, object]],
    model_keys: tuple[str, ...],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 2 if len(model_keys) == 4 else 3
    rows = math.ceil(len(model_keys) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4.5 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, key in zip(axes.flat, model_keys):
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
    for axis in tuple(axes.flat)[len(model_keys) :]:
        axis.set_visible(False)
    figure.suptitle("ReviewPulse v3 — official Restaurants test confusion matrices")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_evaluation(
    rows: list[AspectExample],
    evaluators: list[LoadedEvaluator],
    output_dir: Path,
    *,
    model_keys: tuple[str, ...] = CORE_MODEL_ORDER,
    dataset_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate all models once and persist every A3 comparison artifact."""
    if model_keys not in (CORE_MODEL_ORDER, SIX_MODEL_ORDER):
        raise ValueError("Official comparison mode must contain four or six models")
    output_dir.mkdir(parents=True, exist_ok=True)
    results, predictions, mixed_ids = _evaluate_models(
        rows,
        evaluators,
        model_keys,
    )

    predictions_path = output_dir / "predictions.csv"
    _write_predictions(
        predictions_path,
        rows,
        predictions,
        mixed_ids,
        model_keys,
    )
    error_analysis = _error_analysis(
        rows,
        predictions,
        mixed_ids,
        model_keys,
    )
    (output_dir / "error_analysis.json").write_text(
        json.dumps(error_analysis, indent=2) + "\n",
        encoding="utf-8",
    )
    comparison = _comparison_markdown(results, model_keys)
    (output_dir / "comparison.md").write_text(comparison, encoding="utf-8")
    _plot_confusion_matrices(
        output_dir / "confusion_matrices.png",
        results,
        model_keys,
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "comparison_mode": (
            "canonical_four_model"
            if model_keys == CORE_MODEL_ORDER
            else "exploratory_six_model"
        ),
        "model_order": list(model_keys),
        "scope_note": (
            "GRU and TextCNN are exploratory extensions; the submitted A2 "
            "four-model comparison remains canonical."
        ),
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
            "warm_examples_per_second": "official test examples divided by warm total seconds",
            "parameter_count": "fitted classifier coefficients or stored neural parameters",
            "artifact_bytes": "recursive on-disk bytes for the loaded artifact",
        },
        "predictions_file": predictions_path.name,
        "predictions_sha256": file_sha256(predictions_path),
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
        description="Evaluate the canonical four or exploratory six ReviewPulse v3 models."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ABSA_DATA_DIR / "restaurants",
    )
    parser.add_argument("--artifact-dir", type=Path, default=ABSA_OUTPUTS_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SIX_MODEL_ORDER,
        default=list(CORE_MODEL_ORDER),
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
    model_keys = tuple(args.models)
    if model_keys not in (CORE_MODEL_ORDER, SIX_MODEL_ORDER):
        parser.error(
            "Use the canonical four-model order or the complete exploratory "
            f"six-model order: {SIX_MODEL_ORDER}"
        )
    output_dir = args.output_dir or ABSA_OUTPUTS_DIR / (
        "evaluation"
        if model_keys == CORE_MODEL_ORDER
        else "evaluation-six-model"
    )

    test_path = args.data_dir / "restaurants_test.xml"
    rows = retained_examples(parse_aspect_examples(test_path, "test"))
    try:
        evaluators = load_artifact_evaluators(
            args.artifact_dir,
            model_keys=model_keys,
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
        output_dir,
        model_keys=model_keys,
        dataset_metadata={
            "test_file": str(test_path),
            "test_sha256": file_sha256(test_path),
        },
    )
    print((output_dir / report["comparison_file"]).read_text())
    print(f"Wrote reproducible evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
