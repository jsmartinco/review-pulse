"""Build a reproducible four-model ABSA comparison table from metric artifacts."""

import json
from pathlib import Path

from ..config import ABSA_OUTPUTS_DIR


FILES = {"TF-IDF review-only": "tfidf_baseline_metrics.json", "LSTM review-only": "target_lstm_metrics.json", "ATAE-LSTM": "atae_lstm_metrics.json", "DistilBERT sentence-pair": "distilbert_metrics.json"}


def build_comparison(output_dir: Path = ABSA_OUTPUTS_DIR) -> str:
    rows = ["| Model | Test accuracy | Test macro-F1 |", "|---|---:|---:|"]
    for name, filename in FILES.items():
        metrics = json.loads((output_dir / filename).read_text())["test"]
        rows.append(f"| {name} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} |")
    result = "\n".join(rows) + "\n"
    (output_dir / "four_model_comparison.md").write_text(result)
    return result
