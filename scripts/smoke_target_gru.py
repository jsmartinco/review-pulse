"""Clean-load smoke test for the optional ReviewPulse v3 GRU artifact."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.absa.inference.api import predict_aspects
from src.absa.inference.predictors import get_predictor


def main() -> None:
    """Verify review-only invariance and the unsupported-evidence contract."""
    model_name = "absa_target_gru"
    results = predict_aspects(
        "Great food but the service was dreadful!",
        ["food", "service"],
        model_name,
        get_predictor(model_name),
    )
    assert len(results) == 2
    assert {result["label"] for result in results} <= {
        "negative",
        "neutral",
        "positive",
    }
    assert len({result["label"] for result in results}) == 1
    assert len({result["confidence"] for result in results}) == 1
    assert all(
        result["token_evidence"]["status"] == "unsupported"
        for result in results
    )
    print(
        "absa_target_gru: "
        + ", ".join(
            f"{result['aspect']}={result['label']}"
            for result in results
        )
    )


if __name__ == "__main__":
    main()
