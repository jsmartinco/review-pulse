"""Train and persist the v3 review-only TF-IDF baseline."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib

from ..config import ABSA_DATA_DIR, ABSA_OUTPUTS_DIR
from ..data.parser import parse_aspect_examples
from ..data.splits import split_official_data
from ..evaluation.metrics import compute_metrics
from ..models.baseline import build_baseline


def train_baseline(train_rows, test_rows, *, seed: int = 42):
    splits = split_official_data(train_rows, test_rows, seed=seed)
    model = build_baseline(seed=seed)
    training_started = perf_counter()
    model.fit([row.review_raw for row in splits.train], [row.label for row in splits.train])
    training_seconds = perf_counter() - training_started
    return model, {
        "development": compute_metrics(
            [row.label for row in splits.development],
            model.predict([row.review_raw for row in splits.development]).tolist(),
        ),
        "test": compute_metrics(
            [row.label for row in splits.test],
            model.predict([row.review_raw for row in splits.test]).tolist(),
        ),
        "config": {
            "model": "tfidf_baseline",
            "seed": seed,
            "device": "cpu",
            "estimator": "LogisticRegression",
            "solver": model.named_steps["classifier"].solver,
            "max_iter": model.named_steps["classifier"].max_iter,
            "ngram_range": list(model.named_steps["tfidf"].ngram_range),
        },
        "training_seconds": training_seconds,
    }


def save_artifact(model, metrics, output_dir: Path = ABSA_OUTPUTS_DIR) -> None:
    """Persist the fitted baseline and its complete reproducibility record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "tfidf_baseline.joblib")
    (output_dir / "tfidf_baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ABSA_OUTPUTS_DIR)
    args = parser.parse_args()
    root = ABSA_DATA_DIR / "restaurants"
    model, metrics = train_baseline(parse_aspect_examples(root / "restaurants_train.xml", "train"), parse_aspect_examples(root / "restaurants_test.xml", "test"))
    save_artifact(model, metrics, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
