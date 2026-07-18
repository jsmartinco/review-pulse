from src.absa.inference.api import predict_aspects
from src.absa.inference.predictors import MODEL_OPTIONS, get_predictor


class _Predictor:
    def predict(self, review, aspect, model): return {"aspect": aspect, "label": "positive", "model": model}


def test_predict_aspects_deduplicates_and_preserves_order():
    result = predict_aspects("great food", ["food", " Service ", "FOOD"], "absa_tfidf", _Predictor())
    assert [item["aspect"] for item in result] == ["food", "Service"]


def test_predictor_registry_rejects_unknown_model_without_loading_an_artifact():
    assert "absa_atae_lstm" in MODEL_OPTIONS
    try:
        get_predictor("not-a-model")
    except ValueError as error:
        assert "Unknown v3 model" in str(error)
    else:
        raise AssertionError("Unknown models must not silently select a predictor")
