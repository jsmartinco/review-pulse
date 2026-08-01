"""Aspect-by-model comparison matrix for the ReviewPulse v3 interface.

The builder is Streamlit-free and receives its predictor factory by injection, so
the interface can supply its cached, lock-guarded loader while tests supply
fakes. Models are evaluated sequentially: the shared predictors are not safe to
run in parallel, and the comparison is small enough that sequencing is not a
constraint.
"""

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import pandas as pd

from .api import normalise_aspects, predict_aspects
from .predictors import MODEL_OPTIONS


ARTIFACT_MISSING = "artifact missing"
MISSING_LABEL = ""

# Errors that mean "this artifact could not be loaded or run". A ValueError is
# deliberately excluded: it signals an invalid request, such as an unknown model
# name or an empty aspect list, and must reach the caller.
ARTIFACT_ERRORS = (FileNotFoundError, OSError, RuntimeError)


class Comparison(NamedTuple):
    """Two aligned frames: display text and the label backing each cell.

    Keeping them separate avoids parsing display strings back into labels for
    styling, and keeps both frames as plain strings so they serialise cleanly.
    """

    display: pd.DataFrame
    labels: pd.DataFrame


def format_cell(label: str, confidence: float) -> str:
    """Format one prediction for a matrix cell."""
    return f"{label.title()} · {confidence * 100:.1f}%"


def build_comparison(
    review: str,
    aspects: Sequence[str],
    model_names: Sequence[str],
    get_predictor: Callable[[str], Any],
) -> Comparison:
    """Build an aspects-by-models comparison of one review.

    Raises:
        ValueError: when the review is empty or no usable aspect is supplied.
    """
    if not review.strip():
        raise ValueError("Review must not be empty")
    resolved_aspects = normalise_aspects(list(aspects))
    if not model_names:
        raise ValueError("Provide at least one model")

    display: dict[str, list[str]] = {}
    labels: dict[str, list[str]] = {}
    for model_name in model_names:
        column = MODEL_OPTIONS.get(model_name, model_name)
        try:
            predictor = get_predictor(model_name)
            results = predict_aspects(review, resolved_aspects, model_name, predictor)
        except ARTIFACT_ERRORS:
            display[column] = [ARTIFACT_MISSING] * len(resolved_aspects)
            labels[column] = [MISSING_LABEL] * len(resolved_aspects)
            continue
        display[column] = [format_cell(item["label"], item["confidence"]) for item in results]
        labels[column] = [item["label"] for item in results]

    return Comparison(
        display=pd.DataFrame(display, index=resolved_aspects),
        labels=pd.DataFrame(labels, index=resolved_aspects),
    )
