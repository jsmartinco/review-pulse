"""Export deterministic, report-ready RQ3 token-evidence data."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.absa.inference.api import predict_aspects
from src.absa.inference.predictors import get_predictor


DEFAULT_REVIEW = "Great food but the service was dreadful!"
DEFAULT_ASPECTS = ("food", "service")
SUPPORTED_MODELS = ("absa_atae_lstm", "absa_distilbert")


def export_evidence(
    output: Path,
    *,
    review: str = DEFAULT_REVIEW,
    aspects: tuple[str, ...] = DEFAULT_ASPECTS,
) -> dict:
    """Write one stable mixed-polarity evidence bundle for the A3 report."""
    payload = {
        "purpose": "RQ3 representative indicative token evidence",
        "review": review,
        "aspects": list(aspects),
        "models": {},
    }
    for model_name in SUPPORTED_MODELS:
        payload["models"][model_name] = predict_aspects(
            review,
            list(aspects),
            model_name,
            get_predictor(model_name),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/absa/evidence/rq3_food_service.json"),
    )
    args = parser.parse_args()
    payload = export_evidence(args.output)
    print(
        f"Wrote {args.output} with "
        f"{sum(len(results) for results in payload['models'].values())} aspect views."
    )


if __name__ == "__main__":
    main()
