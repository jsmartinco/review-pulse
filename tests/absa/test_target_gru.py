import json

import pytest
import torch

from src.absa.data.schema import AspectExample
from src.absa.evaluation import compute_metrics, mixed_polarity_multi_aspect
from src.absa.evaluation.artifact_evaluators import load_target_gru_evaluator
from src.absa.inference.predictors import (
    MODEL_OPTIONS,
    OPTIONAL_MODEL_OPTIONS,
    TargetGruAspectPredictor,
)
from src.absa.models.target_gru import TargetAgnosticGRU
from src.absa.models.target_lstm import TargetAgnosticLSTM
from src.absa.training.target_gru import save_artifact, train_target_gru


def _row(
    sentence_id: str,
    review: str,
    aspect: str,
    label: str,
    source: str,
) -> AspectExample:
    start = review.index(aspect)
    return AspectExample(
        sentence_id=sentence_id,
        review_raw=review,
        aspect=aspect,
        aspect_from=start,
        aspect_to=start + len(aspect),
        label=label,
        source_split=source,
        offset_valid=True,
    )


def _tiny_rows() -> tuple[list[AspectExample], list[AspectExample]]:
    train = [
        _row("train-1", "great food", "food", "positive", "train"),
        _row("train-2", "slow service", "service", "negative", "train"),
        _row("train-3", "average menu", "menu", "neutral", "train"),
        _row("train-4", "clean restaurant", "restaurant", "positive", "train"),
        _row("train-5", "noisy room", "room", "negative", "train"),
        _row("train-6", "ordinary desserts", "desserts", "neutral", "train"),
    ]
    test = [
        _row(
            "mixed",
            "great food but slow service",
            "food",
            "positive",
            "test",
        ),
        _row(
            "mixed",
            "great food but slow service",
            "service",
            "negative",
            "test",
        ),
        _row("test-2", "standard menu", "menu", "neutral", "test"),
    ]
    return train, test


def test_target_gru_emits_three_logits_and_backpropagates() -> None:
    model = TargetAgnosticGRU(
        vocab_size=30,
        embedding_dim=8,
        hidden_dim=4,
        dropout=0,
    )
    logits = model(torch.tensor([[1, 2, 0], [3, 4, 5]]))
    assert logits.shape == (2, 3)
    torch.nn.CrossEntropyLoss()(logits, torch.tensor([0, 2])).backward()
    assert model.classifier.weight.grad is not None


def test_target_gru_is_a_matched_lower_parameter_recurrent_ablation() -> None:
    gru = TargetAgnosticGRU(40, embedding_dim=8, hidden_dim=4, dropout=0)
    lstm = TargetAgnosticLSTM(40, embedding_dim=8, hidden_dim=4, dropout=0)
    assert sum(parameter.numel() for parameter in gru.parameters()) < sum(
        parameter.numel() for parameter in lstm.parameters()
    )


def test_target_gru_training_is_deterministic_and_persists_differences(
    tmp_path,
) -> None:
    train_rows, test_rows = _tiny_rows()
    kwargs = {
        "epochs": 2,
        "batch_size": 2,
        "seed": 19,
        "patience": 1,
        "max_length": 8,
        "embedding_dim": 8,
        "hidden_dim": 4,
        "dropout": 0.0,
    }
    first_model, first_vocab, first = train_target_gru(
        train_rows,
        test_rows,
        **kwargs,
    )
    second_model, _, second = train_target_gru(
        train_rows,
        test_rows,
        **kwargs,
    )
    assert first["history"] == second["history"]
    assert first["development"] == second["development"]
    assert first["test"] == second["test"]
    assert all(
        torch.equal(
            first_model.state_dict()[name],
            second_model.state_dict()[name],
        )
        for name in first_model.state_dict()
    )
    assert first["config"]["controlled_against"] == "target_lstm"
    assert first["config"]["embedding_dim"] == 8
    assert first["config"]["hidden_dim"] == 4
    assert first["config"]["dropout"] == 0.0
    assert len(first["config"]["deliberate_differences"]) == 2
    assert first["parameter_count"] == sum(
        parameter.numel() for parameter in first_model.parameters()
    )

    first["provenance"] = {
        "git_commit": "fixture",
        "generated_at_utc": "2026-07-25T00:00:00+10:00",
        "train_file": "restaurants_train.xml",
        "train_sha256": "train-fixture",
        "test_file": "restaurants_test.xml",
        "test_sha256": "test-fixture",
    }
    save_artifact(first_model, first_vocab, first, tmp_path)
    artifact = torch.load(tmp_path / "target_gru.pt", weights_only=True)
    metrics = json.loads((tmp_path / "target_gru_metrics.json").read_text())
    assert artifact["labels"] == ["negative", "neutral", "positive"]
    assert artifact["parameter_count"] == first["parameter_count"]
    assert artifact["config"] == first["config"]
    assert metrics["provenance"] == first["provenance"]
    assert metrics["artifact_bytes"] == (tmp_path / "target_gru.pt").stat().st_size
    assert metrics["artifact_megabytes"] == pytest.approx(
        metrics["artifact_bytes"] / (1024 * 1024)
    )


def test_target_gru_predictor_is_aspect_invariant_and_unsupported(
    tmp_path,
) -> None:
    train_rows, test_rows = _tiny_rows()
    model, vocab, result = train_target_gru(
        train_rows,
        test_rows,
        epochs=1,
        batch_size=2,
        max_length=8,
        embedding_dim=8,
        hidden_dim=4,
        dropout=0,
    )
    result["provenance"] = {"git_commit": "fixture"}
    save_artifact(model, vocab, result, tmp_path)

    predictor = TargetGruAspectPredictor(tmp_path / "target_gru.pt")
    food = predictor.predict(
        "great food but slow service",
        "food",
        "absa_target_gru",
    )
    service = predictor.predict(
        "great food but slow service",
        "service",
        "absa_target_gru",
    )
    assert food["label"] == service["label"]
    assert food["confidence"] == service["confidence"]
    assert food["token_evidence"]["status"] == "unsupported"
    assert service["token_evidence"]["status"] == "unsupported"
    assert "absa_target_gru" not in MODEL_OPTIONS
    assert "absa_target_gru" in OPTIONAL_MODEL_OPTIONS


def test_target_gru_missing_artifact_fails_explicitly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        TargetGruAspectPredictor(tmp_path / "missing.pt")


def test_target_gru_evaluator_uses_shared_full_and_mixed_metrics(
    tmp_path,
) -> None:
    train_rows, test_rows = _tiny_rows()
    model, vocab, result = train_target_gru(
        train_rows,
        test_rows,
        epochs=1,
        batch_size=2,
        max_length=8,
        embedding_dim=8,
        hidden_dim=4,
        dropout=0,
    )
    result["provenance"] = {
        "git_commit": "fixture",
        "generated_at_utc": "2026-07-25T00:00:00+10:00",
        "train_file": "restaurants_train.xml",
        "train_sha256": "train-fixture",
        "test_file": "restaurants_test.xml",
        "test_sha256": "test-fixture",
    }
    save_artifact(model, vocab, result, tmp_path)
    evaluator = load_target_gru_evaluator(tmp_path)
    predictions = evaluator.predict_batch(test_rows)
    full = compute_metrics(
        [row.label for row in test_rows],
        predictions,
    )
    mixed_rows = mixed_polarity_multi_aspect(test_rows)
    mixed_predictions = evaluator.predict_batch(mixed_rows)
    mixed = compute_metrics(
        [row.label for row in mixed_rows],
        mixed_predictions,
    )
    assert full["label_order"] == ["negative", "neutral", "positive"]
    assert mixed["label_order"] == ["negative", "neutral", "positive"]
    assert evaluator.key == "target_gru"
    assert evaluator.artifact_bytes == (tmp_path / "target_gru.pt").stat().st_size
