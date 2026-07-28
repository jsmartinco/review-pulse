"""Run the bounded development-only configuration gate for the optional TextCNN."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.absa.config import ABSA_DATA_DIR, ABSA_OUTPUTS_DIR
from src.absa.data.parser import parse_aspect_examples
from src.absa.training.text_cnn import train_text_cnn


CANDIDATES = (
    {"filter_widths": (2, 3, 4), "num_filters": 64},
    {"filter_widths": (3, 4, 5), "num_filters": 64},
    {"filter_widths": (3, 4, 5), "num_filters": 100},
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def main() -> None:
    """Select one bounded configuration without evaluating the official test."""
    parser = argparse.ArgumentParser(
        description="Select the optional TextCNN configuration on development macro-F1."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ABSA_DATA_DIR / "restaurants",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ABSA_OUTPUTS_DIR / "text_cnn_config_search.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    train_path = args.data_dir / "restaurants_train.xml"
    test_path = args.data_dir / "restaurants_test.xml"
    train_rows = parse_aspect_examples(train_path, "train")
    test_rows = parse_aspect_examples(test_path, "test")
    records = []
    for candidate in CANDIDATES:
        _model, _vocab, result = train_text_cnn(
            train_rows,
            test_rows,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            patience=args.patience,
            filter_widths=candidate["filter_widths"],
            num_filters=candidate["num_filters"],
            evaluate_official_test=False,
        )
        records.append(
            {
                **candidate,
                "filter_widths": list(candidate["filter_widths"]),
                "best_epoch": result["best_epoch"],
                "development_macro_f1": result["development"]["macro_f1"],
                "training_seconds": result["training_seconds"],
                "parameter_count": result["parameter_count"],
                "official_test_evaluated": result["config"][
                    "official_test_evaluated"
                ],
            }
        )

    selected = max(
        enumerate(records),
        key=lambda item: (item[1]["development_macro_f1"], -item[0]),
    )[1]
    payload = {
        "selection_metric": "development_macro_f1",
        "official_test_evaluated": False,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "candidate_limit": len(CANDIDATES),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": _git_commit(),
            "train_file": str(train_path),
            "train_sha256": _sha256(train_path),
            "test_file": str(test_path),
            "test_sha256": _sha256(test_path),
        },
        "candidates": records,
        "selected": {
            "filter_widths": selected["filter_widths"],
            "num_filters": selected["num_filters"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
