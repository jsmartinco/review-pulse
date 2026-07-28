import json

import pytest
import torch

from src.absa.data.schema import AspectExample
from src.absa.evaluation import compute_metrics, mixed_polarity_multi_aspect
from src.absa.evaluation.artifact_evaluators import load_text_cnn_evaluator
from src.absa.inference.predictors import (
    MODEL_OPTIONS,
    OPTIONAL_MODEL_OPTIONS,
    TextCnnAspectPredictor,
)
from src.absa.models.text_cnn import TextCNN
from src.absa.tokenization.sequence import build_vocab, encode, tokens
from src.absa.training.text_cnn import save_artifact, train_text_cnn


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


def _provenance() -> dict[str, str]:
    return {
        "git_commit": "fixture",
        "generated_at_utc": "2026-07-29T00:00:00+10:00",
        "train_file": "restaurants_train.xml",
        "train_sha256": "train-fixture",
        "test_file": "restaurants_test.xml",
        "test_sha256": "test-fixture",
    }


def test_text_cnn_emits_three_logits_and_backpropagates() -> None:
    model = TextCNN(
        vocab_size=30,
        embedding_dim=8,
        num_filters=4,
        filter_widths=(2, 3),
        dropout=0,
    )
    logits = model(torch.tensor([[1, 2, 0], [3, 4, 5]]))
    assert logits.shape == (2, 3)
    torch.nn.CrossEntropyLoss()(logits, torch.tensor([0, 2])).backward()
    assert model.classifier.weight.grad is not None


def test_text_cnn_right_pads_short_inputs_and_tokenizer_retains_punctuation() -> None:
    model = TextCNN(
        vocab_size=10,
        embedding_dim=4,
        num_filters=2,
        filter_widths=(3, 4, 5),
        dropout=0,
    )
    assert model(torch.tensor([[1]])).shape == (1, 3)
    assert tokens("Great, but slow!") == ["great", ",", "but", "slow", "!"]
    vocab = build_vocab(["Great, but slow!"])
    assert encode(["!"], vocab, max_length=1).item() == vocab["!"]


def test_text_cnn_rejects_invalid_convolution_configuration() -> None:
    with pytest.raises(ValueError, match="unique"):
        TextCNN(10, filter_widths=(3, 3))
    train_rows, test_rows = _tiny_rows()
    with pytest.raises(ValueError, match="max_length"):
        train_text_cnn(
            train_rows,
            test_rows,
            max_length=4,
            filter_widths=(3, 5),
        )


def test_text_cnn_training_is_deterministic_and_persists_configuration(
    tmp_path,
) -> None:
    train_rows, test_rows = _tiny_rows()
    kwargs = {
        "epochs": 2,
        "batch_size": 2,
        "seed": 23,
        "patience": 1,
        "max_length": 8,
        "embedding_dim": 8,
        "num_filters": 4,
        "filter_widths": (2, 3),
        "dropout": 0.0,
    }
    first_model, first_vocab, first = train_text_cnn(
        train_rows,
        test_rows,
        **kwargs,
    )
    second_model, _, second = train_text_cnn(
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
    assert first["config"]["filter_widths"] == [2, 3]
    assert first["config"]["num_filters"] == 4
    assert first["config"]["padding"] == "right_pad_to_max_filter_width"
    assert first["config"]["pooling"] == "global_max"
    assert first["config"]["configuration_selection"] == "development_macro_f1_only"
    assert first["config"]["official_test_evaluated"] is True
    assert first["parameter_count"] == sum(
        parameter.numel() for parameter in first_model.parameters()
    )

    first["provenance"] = _provenance()
    save_artifact(first_model, first_vocab, first, tmp_path)
    artifact = torch.load(tmp_path / "text_cnn.pt", weights_only=True)
    metrics = json.loads((tmp_path / "text_cnn_metrics.json").read_text())
    assert artifact["labels"] == ["negative", "neutral", "positive"]
    assert artifact["parameter_count"] == first["parameter_count"]
    assert artifact["config"] == first["config"]
    assert metrics["provenance"] == first["provenance"]
    assert metrics["artifact_bytes"] == (tmp_path / "text_cnn.pt").stat().st_size
    assert metrics["artifact_megabytes"] == pytest.approx(
        metrics["artifact_bytes"] / (1024 * 1024)
    )


def test_text_cnn_predictor_is_aspect_invariant_and_unsupported(
    tmp_path,
) -> None:
    train_rows, test_rows = _tiny_rows()
    model, vocab, result = train_text_cnn(
        train_rows,
        test_rows,
        epochs=1,
        batch_size=2,
        max_length=8,
        embedding_dim=8,
        num_filters=4,
        filter_widths=(2, 3),
        dropout=0,
    )
    result["provenance"] = _provenance()
    save_artifact(model, vocab, result, tmp_path)

    predictor = TextCnnAspectPredictor(tmp_path / "text_cnn.pt")
    food = predictor.predict(
        "great food but slow service",
        "food",
        "absa_text_cnn",
    )
    service = predictor.predict(
        "great food but slow service",
        "service",
        "absa_text_cnn",
    )
    assert food["label"] == service["label"]
    assert food["confidence"] == service["confidence"]
    assert food["token_evidence"]["status"] == "unsupported"
    assert service["token_evidence"]["status"] == "unsupported"
    assert "absa_text_cnn" not in MODEL_OPTIONS
    assert "absa_text_cnn" in OPTIONAL_MODEL_OPTIONS


def test_text_cnn_missing_artifact_fails_explicitly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        TextCnnAspectPredictor(tmp_path / "missing.pt")


def test_text_cnn_evaluator_uses_shared_full_and_mixed_metrics(
    tmp_path,
) -> None:
    train_rows, test_rows = _tiny_rows()
    model, vocab, result = train_text_cnn(
        train_rows,
        test_rows,
        epochs=1,
        batch_size=2,
        max_length=8,
        embedding_dim=8,
        num_filters=4,
        filter_widths=(2, 3),
        dropout=0,
    )
    result["provenance"] = _provenance()
    save_artifact(model, vocab, result, tmp_path)
    evaluator = load_text_cnn_evaluator(tmp_path)
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
    assert evaluator.key == "text_cnn"
    assert evaluator.artifact_bytes == (tmp_path / "text_cnn.pt").stat().st_size


def test_text_cnn_configuration_selection_does_not_evaluate_official_test() -> None:
    train_rows, test_rows = _tiny_rows()
    _model, _vocab, result = train_text_cnn(
        train_rows,
        test_rows,
        epochs=1,
        batch_size=2,
        max_length=8,
        embedding_dim=8,
        num_filters=4,
        filter_widths=(2, 3),
        dropout=0,
        evaluate_official_test=False,
    )
    assert result["config"]["official_test_evaluated"] is False
    assert result["test"] == {
        "status": "not_evaluated_during_configuration_selection"
    }
