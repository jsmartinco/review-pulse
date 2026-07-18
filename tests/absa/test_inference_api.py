from src.absa.inference.api import predict_aspects


class _Predictor:
    def predict(self, review, aspect, model): return {"aspect": aspect, "label": "positive", "model": model}


def test_predict_aspects_deduplicates_and_preserves_order():
    result = predict_aspects("great food", ["food", " Service ", "FOOD"], "absa_tfidf", _Predictor())
    assert [item["aspect"] for item in result] == ["food", "Service"]
