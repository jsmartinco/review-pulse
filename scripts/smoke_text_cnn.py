"""Clean-load smoke test for the optional ReviewPulse v3 TextCNN artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.absa.inference.api import predict_aspects
from src.absa.inference.predictors import get_predictor


def main() -> None:
    """Verify repeated review-only predictions and unsupported evidence."""
    model_name = "absa_text_cnn"
    results = predict_aspects(
        "The menu is limited, but the desserts are excellent and the restaurant is clean.",
        ["menu", "desserts", "restaurant"],
        model_name,
        get_predictor(model_name),
    )
    if len(results) != 3:
        raise RuntimeError("TextCNN smoke expected one result per supplied aspect")
    labels = {result["label"] for result in results}
    confidences = {result["confidence"] for result in results}
    if len(labels) != 1 or len(confidences) != 1:
        raise RuntimeError("TextCNN unexpectedly changed prediction by aspect")
    if any(
        result["token_evidence"]["status"] != "unsupported"
        for result in results
    ):
        raise RuntimeError("TextCNN must not claim token-evidence support")
    print(
        json.dumps(
            {
                "model": model_name,
                "aspects": [result["aspect"] for result in results],
                "label": results[0]["label"],
                "confidence": results[0]["confidence"],
                "token_evidence": "unsupported",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
