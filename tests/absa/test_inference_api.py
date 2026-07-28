from pathlib import Path

from src.absa.inference.api import predict_aspects
from src.absa.inference.predictors import (
    ALL_MODEL_OPTIONS,
    MODEL_OPTIONS,
    OPTIONAL_MODEL_OPTIONS,
    get_predictor,
)


class _Predictor:
    def __init__(self):
        self.reviews = []

    def predict(self, review, aspect, model):
        self.reviews.append(review)
        return {"aspect": aspect, "label": "positive", "model": model}


def test_predict_aspects_deduplicates_and_preserves_order():
    result = predict_aspects("great food", ["food", " Service ", "FOOD"], "absa_tfidf", _Predictor())
    assert [item["aspect"] for item in result] == ["food", "Service"]


def test_predict_aspects_preserves_visible_review_whitespace_for_offsets():
    predictor = _Predictor()
    predict_aspects("  Great food!  ", ["food"], "absa_tfidf", predictor)
    assert predictor.reviews == ["  Great food!  "]


def test_predictor_registry_rejects_unknown_model_without_loading_an_artifact():
    assert "absa_atae_lstm" in MODEL_OPTIONS
    assert "absa_target_gru" in OPTIONAL_MODEL_OPTIONS
    assert "absa_text_cnn" in OPTIONAL_MODEL_OPTIONS
    assert list(ALL_MODEL_OPTIONS) == [
        "absa_tfidf",
        "absa_target_lstm",
        "absa_target_gru",
        "absa_text_cnn",
        "absa_atae_lstm",
        "absa_distilbert",
    ]
    try:
        get_predictor("not-a-model")
    except ValueError as error:
        assert "Unknown v3 model" in str(error)
    else:
        raise AssertionError("Unknown models must not silently select a predictor")


def test_v3_page_uses_the_explicit_six_model_registry():
    page = Path("pages/2_ReviewPulse_v3_0_0.py").read_text()
    assert "ALL_MODEL_OPTIONS" in page
    assert "OPTIONAL_MODEL_OPTIONS" not in page
