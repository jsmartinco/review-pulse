"""Train the TextCNN configuration selected by the development-only gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.absa.config import ABSA_DATA_DIR, ABSA_OUTPUTS_DIR
from src.absa.model_order import SIX_MODEL_ORDER
from src.absa.training.provenance import file_sha256, git_commit


def selected_configuration(
    selection_path: Path,
    data_dir: Path,
    *,
    expected_commit: str | None = None,
) -> tuple[tuple[int, ...], int]:
    """Validate the selection record and return its filter widths and count."""
    payload = json.loads(selection_path.read_text())
    if payload.get("official_test_evaluated") is not False:
        raise ValueError(
            "TextCNN configuration selection must not evaluate the official test"
        )
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("TextCNN selection record has no selected configuration")
    width_values = selected.get("filter_widths")
    filter_value = selected.get("num_filters")
    if not isinstance(width_values, list) or not isinstance(filter_value, int):
        raise ValueError("TextCNN selected configuration is invalid")
    widths = tuple(int(width) for width in width_values)
    num_filters = filter_value
    if not widths or any(width < 1 for width in widths) or num_filters < 1:
        raise ValueError("TextCNN selected configuration is invalid")

    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not any(
        isinstance(candidate, dict)
        and candidate.get("filter_widths") == list(widths)
        and candidate.get("num_filters") == num_filters
        for candidate in candidates
    ):
        raise ValueError("TextCNN selected configuration is not a gate candidate")

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("TextCNN selection record has no provenance")
    current_commit = expected_commit or git_commit()
    if provenance.get("git_commit") != current_commit:
        raise ValueError(
            "TextCNN selection and training must run from the same Git commit"
        )
    for split in ("train", "test"):
        path = data_dir / f"restaurants_{split}.xml"
        if provenance.get(f"{split}_sha256") != file_sha256(path):
            raise ValueError(
                f"TextCNN selection {split} checksum does not match {path}"
            )
    return widths, num_filters


def main() -> None:
    """Forward the verified winning configuration to the common trainer."""
    parser = argparse.ArgumentParser(
        description="Train the TextCNN configuration selected by its gate."
    )
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=ABSA_OUTPUTS_DIR / "text_cnn_config_search.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ABSA_DATA_DIR / "restaurants",
    )
    parser.add_argument("--output-dir", type=Path, default=ABSA_OUTPUTS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--all-six",
        action="store_true",
        help="Train the four A2 core models plus exploratory GRU and selected TextCNN.",
    )
    args = parser.parse_args()

    widths, num_filters = selected_configuration(
        args.selection_file,
        args.data_dir,
    )
    model_keys = SIX_MODEL_ORDER if args.all_six else ("text_cnn",)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.absa.training.runner",
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(args.output_dir),
            "--models",
            *model_keys,
            "--seed",
            str(args.seed),
            "--cnn-epochs",
            str(args.epochs),
            "--cnn-batch-size",
            str(args.batch_size),
            "--cnn-filter-widths",
            *(str(width) for width in widths),
            "--cnn-num-filters",
            str(num_filters),
            "--patience",
            str(args.patience),
            "--device",
            args.device,
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
