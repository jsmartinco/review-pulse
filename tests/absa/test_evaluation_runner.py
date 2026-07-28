import csv
import json

import numpy as np

from src.absa.data.schema import AspectExample
from src.absa.evaluation.artifact_evaluators import (
    CORE_MODEL_ORDER,
    SIX_MODEL_ORDER,
    LoadedEvaluator,
    UnverifiedArtifactError,
    _normalise_model_keys,
    _assert_labels,
    artifact_size,
    load_artifact_evaluators,
)
from src.absa.evaluation.runner import run_evaluation


def _row(sentence_id: str, review: str, aspect: str, label: str) -> AspectExample:
    start = review.index(aspect)
    return AspectExample(
        sentence_id=sentence_id,
        review_raw=review,
        aspect=aspect,
        aspect_from=start,
        aspect_to=start + len(aspect),
        label=label,
        source_split="test",
        offset_valid=True,
    )


def _evaluator(key: str, predictions: dict[tuple[str, str], str]) -> LoadedEvaluator:
    return LoadedEvaluator(
        key=key,
        display_name={
            "tfidf": "TF-IDF review-only",
            "target_lstm": "LSTM review-only",
            "target_gru": "GRU review-only (exploratory)",
            "text_cnn": "Text CNN review-only (exploratory)",
            "atae_lstm": "ATAE-LSTM",
            "distilbert": "DistilBERT sentence-pair",
        }[key],
        device="cpu",
        artifact_bytes=1024,
        parameter_count=100,
        training_seconds=2.5,
        training_config={"seed": 42},
        provenance={"git_commit": "fixture"},
        load_seconds=0.01,
        predict_batch=lambda rows: [
            predictions[(row.sentence_id, row.aspect)] for row in rows
        ],
    )


def test_runner_writes_common_predictions_metrics_efficiency_and_errors(tmp_path) -> None:
    rows = [
        _row("mixed", "great food but slow service", "food", "positive"),
        _row("mixed", "great food but slow service", "service", "negative"),
        _row("neutral", "ordinary menu", "menu", "neutral"),
        _row("positive", "clean restaurant", "restaurant", "positive"),
    ]
    gold = {(row.sentence_id, row.aspect): row.label for row in rows}
    review_only = gold | {("mixed", "service"): "positive"}
    evaluators = [
        _evaluator("tfidf", review_only),
        _evaluator("target_lstm", review_only),
        _evaluator("atae_lstm", gold),
        _evaluator("distilbert", gold),
    ]

    output_dir = tmp_path / "evaluation"
    report = run_evaluation(
        rows,
        evaluators,
        output_dir,
        dataset_metadata={"test_sha256": "fixture"},
    )

    assert list(report["models"]) == ["tfidf", "target_lstm", "atae_lstm", "distilbert"]
    assert report["dataset"]["official_test_examples"] == 4
    assert report["dataset"]["mixed_polarity_examples"] == 2
    assert report["models"]["tfidf"]["mixed_polarity_multi_aspect"]["accuracy"] == 0.5
    assert report["models"]["atae_lstm"]["mixed_polarity_multi_aspect"]["accuracy"] == 1.0
    assert report["models"]["distilbert"]["efficiency"]["training_seconds"] == 2.5

    with (output_dir / "predictions.csv").open(newline="") as handle:
        prediction_rows = list(csv.DictReader(handle))
    assert len(prediction_rows) == len(rows)
    assert prediction_rows[1]["gold"] == "negative"
    assert prediction_rows[1]["tfidf"] == "positive"
    assert prediction_rows[1]["atae_lstm"] == "negative"

    errors = json.loads((output_dir / "error_analysis.json").read_text())
    assert errors["counts"]["conditioned_wins_on_mixed_subset"] == 1
    comparison = (output_dir / "comparison.md").read_text()
    assert comparison.count("| TF-IDF review-only |") == 1
    assert "| Scope |" not in comparison
    assert "### Full-test per-class evidence" not in comparison
    assert (output_dir / "confusion_matrices.png").stat().st_size > 0
    assert report["predictions_sha256"]
    assert report["comparison_mode"] == "canonical_four_model"
    assert tuple(report["model_order"]) == CORE_MODEL_ORDER


def test_runner_writes_explicit_six_model_supplement(tmp_path) -> None:
    rows = [
        _row("mixed", "great food but slow service", "food", "positive"),
        _row("mixed", "great food but slow service", "service", "negative"),
        _row("neutral", "ordinary menu", "menu", "neutral"),
    ]
    gold = {(row.sentence_id, row.aspect): row.label for row in rows}
    review_only = gold | {("mixed", "service"): "positive"}
    evaluators = [
        _evaluator(
            key,
            review_only
            if key in {"tfidf", "target_lstm", "target_gru", "text_cnn"}
            else gold,
        )
        for key in SIX_MODEL_ORDER
    ]

    output_dir = tmp_path / "evaluation-six-model"
    report = run_evaluation(
        rows,
        evaluators,
        output_dir,
        model_keys=SIX_MODEL_ORDER,
    )

    assert report["comparison_mode"] == "exploratory_six_model"
    assert list(report["models"]) == list(SIX_MODEL_ORDER)
    assert report["models"]["text_cnn"]["efficiency"]["parameter_count"] == 100
    comparison = (output_dir / "comparison.md").read_text()
    assert "| GRU review-only (exploratory) | exploratory |" in comparison
    assert "| Text CNN review-only (exploratory) | exploratory |" in comparison
    assert "Throughput ex/s" in comparison
    assert "| Device |" in comparison
    assert "### Full-test per-class evidence" in comparison
    assert "### Mixed-polarity per-class evidence" in comparison
    assert "Negative (n=1)" in comparison
    with (output_dir / "predictions.csv").open(newline="") as handle:
        prediction_rows = list(csv.DictReader(handle))
    assert list(prediction_rows[0])[-6:] == list(SIX_MODEL_ORDER)
    errors = json.loads((output_dir / "error_analysis.json").read_text())
    assert errors["model_groups"]["review_only"] == [
        "tfidf",
        "target_lstm",
        "target_gru",
        "text_cnn",
    ]
    assert (output_dir / "confusion_matrices.png").stat().st_size > 0


def test_artifact_size_sums_files_recursively(tmp_path) -> None:
    artifact = tmp_path / "artifact"
    nested = artifact / "nested"
    nested.mkdir(parents=True)
    (artifact / "one.bin").write_bytes(b"123")
    (nested / "two.bin").write_bytes(b"4567")
    assert artifact_size(artifact) == 7


def test_artifact_preflight_rejects_mixed_training_runs_before_model_loading(tmp_path) -> None:
    filenames = {
        "tfidf_baseline_metrics.json": "commit-a",
        "target_lstm_metrics.json": "commit-a",
        "atae_lstm_metrics.json": "commit-b",
        "distilbert_metrics.json": "commit-a",
    }
    for filename, commit in filenames.items():
        (tmp_path / filename).write_text(
            json.dumps(
                {
                    "config": {"seed": 42},
                    "training_seconds": 1.0,
                    "provenance": {
                        "git_commit": commit,
                        "train_sha256": "train",
                        "test_sha256": "test",
                    },
                }
            )
        )

    try:
        load_artifact_evaluators(tmp_path)
    except UnverifiedArtifactError as error:
        assert "same commit" in str(error)
    else:
        raise AssertionError("A comparison must not mix artifacts from different runs")


def test_artifact_selection_accepts_ordered_subsets_and_rejects_reordering() -> None:
    assert _normalise_model_keys(("target_gru", "text_cnn")) == (
        "target_gru",
        "text_cnn",
    )
    try:
        _normalise_model_keys(("text_cnn", "target_gru"))
    except ValueError as error:
        assert "order" in str(error)
    else:
        raise AssertionError("Selected models must retain the shared order")


def test_artifact_label_validation_accepts_numpy_classifier_classes(tmp_path) -> None:
    _assert_labels(
        np.array(["negative", "neutral", "positive"]),
        tmp_path / "tfidf.joblib",
        True,
    )
