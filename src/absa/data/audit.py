"""Reproducible SemEval Restaurants audit for the v3 data contract."""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ..config import ABSA_DATA_DIR
from .parser import parse_aspect_examples


def audit_examples(examples: list) -> dict[str, object]:
    invalid = [asdict(item) for item in examples if not item.offset_valid]
    return {
        "aspect_examples": len(examples),
        "sentences_with_aspects": len({item.sentence_id for item in examples}),
        "polarity_counts": dict(sorted(Counter(item.label for item in examples).items())),
        "offset_valid": len(examples) - len(invalid),
        "offset_invalid": len(invalid),
        "invalid_offsets": invalid,
    }


def run_audit(train_path: Path, test_path: Path) -> dict[str, object]:
    train = parse_aspect_examples(train_path, "train")
    test = parse_aspect_examples(test_path, "test")
    return {"train": audit_examples(train), "test": audit_examples(test)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ABSA_DATA_DIR / "restaurants" / "restaurants_train.xml")
    parser.add_argument("--test", type=Path, default=ABSA_DATA_DIR / "restaurants" / "restaurants_test.xml")
    parser.add_argument("--output", type=Path, default=ABSA_DATA_DIR / "restaurants" / "audit.json")
    args = parser.parse_args()
    report = run_audit(args.train, args.test)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
