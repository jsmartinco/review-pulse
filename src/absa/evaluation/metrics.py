"""Shared deterministic metrics for every ReviewPulse v3 model."""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from ..labels import LABELS


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, object]:
    """Return JSON-serialisable three-class metrics in canonical label order."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must be non-empty and have equal length")
    report = classification_report(y_true, y_pred, labels=list(LABELS), output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0),
        "per_class": {label: report[label] for label in LABELS},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist(),
        "label_order": list(LABELS),
    }
