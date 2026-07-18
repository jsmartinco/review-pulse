"""Train and persist the v3 review-only TF-IDF baseline."""

import argparse
import json
from pathlib import Path

import joblib

from ..config import ABSA_DATA_DIR, ABSA_OUTPUTS_DIR
from ..data.parser import parse_aspect_examples
from ..data.splits import split_official_data
from ..evaluation.metrics import compute_metrics
from ..models.baseline import build_baseline


def train_baseline(train_rows, test_rows):
    splits = split_official_data(train_rows, test_rows)
    model = build_baseline()
    model.fit([row.review_raw for row in splits.train], [row.label for row in splits.train])
    return model, {"development": compute_metrics([row.label for row in splits.development], model.predict([row.review_raw for row in splits.development]).tolist()), "test": compute_metrics([row.label for row in splits.test], model.predict([row.review_raw for row in splits.test]).tolist())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ABSA_OUTPUTS_DIR)
    args = parser.parse_args()
    root = ABSA_DATA_DIR / "restaurants"
    model, metrics = train_baseline(parse_aspect_examples(root / "restaurants_train.xml", "train"), parse_aspect_examples(root / "restaurants_test.xml", "test"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "tfidf_baseline.joblib")
    (args.output_dir / "tfidf_baseline_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
