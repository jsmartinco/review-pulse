"""Small adapters that normalise v3 prediction payloads."""

import joblib

from ..config import ABSA_OUTPUTS_DIR


class TfidfAspectPredictor:
    def __init__(self, path=ABSA_OUTPUTS_DIR / "tfidf_baseline.joblib") -> None:
        self.model = joblib.load(path)

    def predict(self, review: str, aspect: str, model_name: str) -> dict:
        probabilities = self.model.predict_proba([review])[0]
        labels = list(self.model.classes_)
        index = probabilities.argmax()
        return {"aspect": aspect, "label": labels[index], "confidence": float(probabilities[index]), "model": model_name, "token_evidence": None}
