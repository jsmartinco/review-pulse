"""Clean-load smoke test for locally prepared ReviewPulse v3 artifacts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.absa.inference.api import predict_aspects
from src.absa.inference.predictors import MODEL_OPTIONS, get_predictor


def main() -> None:
    review = "The food was great but the service was slow."
    aspects = ["food", "service"]
    for model_name in MODEL_OPTIONS:
        predictions = predict_aspects(
            review, aspects, model_name, get_predictor(model_name)
        )
        assert len(predictions) == len(aspects)
        assert all(item["label"] in {"negative", "neutral", "positive"} for item in predictions)
        expected_status = (
            "supported"
            if model_name in {"absa_atae_lstm", "absa_distilbert"}
            else "unsupported"
        )
        assert all(item["token_evidence"]["status"] == expected_status for item in predictions)
        if expected_status == "supported":
            assert all(item["token_evidence"]["tokens"] for item in predictions)
        print(
            f"{model_name}: "
            + ", ".join(
                f"{item['aspect']}={item['label']}" for item in predictions
            )
        )


if __name__ == "__main__":
    main()
