import csv
import json

from src.absa.data.schema import AspectExample
from src.absa.evaluation.artifact_evaluators import (
    LoadedEvaluator,
    UnverifiedArtifactError,
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
            "atae_lstm": "ATAE-LSTM",
            "distilbert": "DistilBERT sentence-pair",
        }[key],
        device="cpu",
        artifact_bytes=1024,
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
    assert (output_dir / "comparison.md").read_text().count("| TF-IDF review-only |") == 1
    assert (output_dir / "confusion_matrices.png").stat().st_size > 0
    assert report["predictions_sha256"]


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
