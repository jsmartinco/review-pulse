"""Align ATAE attention to exact, visible review spans."""

import math
import re

from .evidence import EVIDENCE_CAVEAT


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def visible_token_spans(review: str, *, max_length: int | None = None) -> list[dict[str, int | str]]:
    """Return model-compatible token spans without changing visible text."""
    matches = list(TOKEN_PATTERN.finditer(review))
    if max_length is not None:
        matches = matches[:max_length]
    return [
        {"token": match.group(0), "start": match.start(), "end": match.end()}
        for match in matches
    ]


def align_attention(review: str, weights, *, max_length: int = 80) -> list[dict[str, float | int | str]]:
    """Align non-padding attention weights to exact review token offsets."""
    visible = visible_token_spans(review, max_length=max_length)
    values = weights.detach().cpu().reshape(-1).tolist() if hasattr(weights, "detach") else list(weights)
    aligned = []
    for token, weight in zip(visible, values):
        value = float(weight)
        if not math.isfinite(value):
            value = 0.0
        aligned.append({**token, "score": value, "weight": value})
    return aligned
