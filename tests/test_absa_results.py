"""Unit tests for the Streamlit-facing v3 aspect result helpers."""

import pytest

from src.absa.inference.predictors import (
    ALL_MODEL_OPTIONS,
    ASPECT_CONDITIONED_MODELS,
    REVIEW_ONLY_MODELS,
    exposes_token_evidence,
)
from src.app.absa_results import (
    LABEL_STYLE,
    chunk,
    first_evidence,
    format_confidence,
    has_token_evidence,
    label_colour,
)


def _supported(aspect: str) -> dict:
    return {
        "aspect": aspect,
        "label": "positive",
        "confidence": 0.9,
        "token_evidence": {
            "status": "supported",
            "method": "ATAE-LSTM aspect-conditioned attention",
            "tokens": [{"token": "Great", "start": 0, "end": 5, "score": 1.0}],
            "caveat": "Token scores are indicative evidence.",
            "limitations": "Attention weights show where this model concentrated mass.",
        },
    }


def _unsupported(aspect: str) -> dict:
    return {
        "aspect": aspect,
        "label": "neutral",
        "confidence": 0.5,
        "token_evidence": {"status": "unsupported", "method": None, "tokens": []},
    }


def test_label_style_covers_exactly_the_three_classes():
    assert set(LABEL_STYLE) == {"positive", "neutral", "negative"}


@pytest.mark.parametrize("label", ["positive", "Positive", "NEGATIVE", "neutral"])
def test_label_colour_is_case_insensitive(label):
    assert label_colour(label) == LABEL_STYLE[label.lower()]


def test_label_colour_falls_back_for_unknown_label():
    assert label_colour("mixed") == LABEL_STYLE["neutral"]


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [(0.825, "82.5%"), (1.0, "100.0%"), (0.0, "0.0%"), (0.5361, "53.6%")],
)
def test_format_confidence_always_uses_one_decimal(confidence, expected):
    assert format_confidence(confidence) == expected


def test_chunk_wraps_after_three_entries():
    assert [len(row) for row in chunk(list(range(5)))] == [3, 2]
    assert [len(row) for row in chunk(list(range(3)))] == [3]
    assert chunk([]) == []


def test_chunk_rejects_non_positive_size():
    with pytest.raises(ValueError, match="at least 1"):
        chunk([1, 2], size=0)


def test_has_token_evidence_requires_supported_status_and_tokens():
    assert has_token_evidence(_supported("food")) is True
    assert has_token_evidence(_unsupported("food")) is False
    empty = _supported("food")
    empty["token_evidence"]["tokens"] = []
    assert has_token_evidence(empty) is False
    assert has_token_evidence({"aspect": "food"}) is False


def test_first_evidence_returns_one_payload_for_the_whole_result_set():
    results = [_unsupported("food"), _supported("service"), _supported("price")]
    evidence = first_evidence(results)
    assert evidence is not None
    assert evidence["method"] == "ATAE-LSTM aspect-conditioned attention"
    assert first_evidence([_unsupported("food")]) is None
    assert first_evidence([]) is None


def test_model_evidence_partition_is_exhaustive():
    """Every selectable model must be classified as review-only or aspect-conditioned."""
    assert REVIEW_ONLY_MODELS.isdisjoint(ASPECT_CONDITIONED_MODELS)
    assert REVIEW_ONLY_MODELS | ASPECT_CONDITIONED_MODELS == set(ALL_MODEL_OPTIONS)


@pytest.mark.parametrize("model_name", sorted(ASPECT_CONDITIONED_MODELS))
def test_aspect_conditioned_models_expose_evidence(model_name):
    assert exposes_token_evidence(model_name) is True


@pytest.mark.parametrize("model_name", sorted(REVIEW_ONLY_MODELS))
def test_review_only_models_do_not_expose_evidence(model_name):
    assert exposes_token_evidence(model_name) is False
