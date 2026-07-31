"""Safe HTML rendering for exact-offset token evidence."""

import html
import math
from collections.abc import Sequence


ALPHA_FLOOR = 0.06
ALPHA_RANGE = 0.74


def _score_of(token: dict) -> float:
    """Read a token score, tolerating the legacy ``weight`` key."""
    return float(token.get("score", token.get("weight", 0.0)))


def _relative_scores(tokens: Sequence[dict]) -> list[float]:
    """Scale scores to 0.0-1.0 across the view so the weakest token reads as unshaded.

    Attention is a softmax over the whole review, so raw weights cluster in a narrow
    band around ``1 / len(tokens)``. Dividing by the maximum alone never approaches
    zero and shades every token, which hides the contrast between aspect views.
    Rescaling between the observed minimum and maximum restores that contrast.
    """
    finite = [_score_of(token) for token in tokens if math.isfinite(_score_of(token))]
    if not finite:
        return [0.0] * len(tokens)
    minimum, maximum = min(finite), max(finite)
    span = maximum - minimum
    if span <= 0:
        return [0.0] * len(tokens)
    relatives = []
    for token in tokens:
        score = _score_of(token)
        if not math.isfinite(score):
            relatives.append(0.0)
            continue
        relatives.append(max(0.0, min(1.0, (score - minimum) / span)))
    return relatives


def render_token_heatmap_html(review: str, tokens: Sequence[dict]) -> str:
    """Render aligned token scores while preserving punctuation and whitespace."""
    relatives = _relative_scores(tokens)
    cursor = 0
    fragments: list[str] = []
    for token, relative in zip(tokens, relatives):
        start, end = int(token["start"]), int(token["end"])
        surface = str(token["token"])
        if start < cursor or end < start or end > len(review) or review[start:end] != surface:
            raise ValueError("Token evidence is not aligned to the visible review")
        fragments.append(html.escape(review[cursor:start]))
        alpha = ALPHA_FLOOR + (ALPHA_RANGE * relative)
        fragments.append(
            '<span style="background: rgba(255, 126, 95, '
            f'{alpha:.3f}); border-radius: 0.25rem; padding: 0.08rem 0.12rem;" '
            f'title="token score: {_score_of(token):.4f}">{html.escape(surface)}</span>'
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
