"""Safe HTML rendering for exact-offset token evidence."""

import html
import math
from collections.abc import Sequence


def render_token_heatmap_html(review: str, tokens: Sequence[dict]) -> str:
    """Render aligned token scores while preserving punctuation and whitespace."""
    maximum = max(
        (
            float(token.get("score", token.get("weight", 0.0)))
            for token in tokens
            if math.isfinite(float(token.get("score", token.get("weight", 0.0))))
        ),
        default=0.0,
    )
    cursor = 0
    fragments: list[str] = []
    for token in tokens:
        start, end = int(token["start"]), int(token["end"])
        surface = str(token["token"])
        if start < cursor or end < start or end > len(review) or review[start:end] != surface:
            raise ValueError("Token evidence is not aligned to the visible review")
        fragments.append(html.escape(review[cursor:start]))
        score = float(token.get("score", token.get("weight", 0.0)))
        relative = max(0.0, min(1.0, score / maximum)) if maximum > 0 else 0.0
        alpha = 0.10 + (0.70 * relative)
        fragments.append(
            '<span style="background: rgba(255, 126, 95, '
            f'{alpha:.3f}); border-radius: 0.25rem; padding: 0.08rem 0.12rem;" '
            f'title="token score: {score:.4f}">{html.escape(surface)}</span>'
        )
        cursor = end
    fragments.append(html.escape(review[cursor:]))
    return (
        '<div aria-label="Indicative token evidence" '
        'style="line-height: 2.1; white-space: pre-wrap; padding: 0.75rem; '
        'border: 1px solid rgba(128,128,128,0.25); border-radius: 0.5rem;">'
        + "".join(fragments)
        + "</div>"
    )
