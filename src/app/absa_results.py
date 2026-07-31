"""Streamlit presentation helpers for ReviewPulse v3 aspect results.

This module owns the Streamlit-facing rendering of the aspect prediction payload
produced by :mod:`src.absa.inference.api`. It lives outside ``src/absa`` so that
package stays free of Streamlit imports and remains testable without the UI stack.
"""

from collections.abc import Sequence
from typing import Any

import streamlit as st

from src.absa.inference.comparison import Comparison
from src.absa.interpretability.heatmap import render_token_heatmap_html


MAX_CARDS_PER_ROW = 3

# Single source of truth for sentiment colour. Chosen to stay legible on both the
# light and dark Streamlit themes.
LABEL_STYLE: dict[str, str] = {
    "positive": "#2ea043",
    "neutral": "#8b949e",
    "negative": "#f85149",
}
_FALLBACK_COLOUR = "#8b949e"
_MISSING_CSS = f"color: {_FALLBACK_COLOUR}; font-style: italic"


def format_confidence(confidence: float) -> str:
    """Format a 0.0-1.0 confidence as a one-decimal percentage."""
    return f"{confidence * 100:.1f}%"


def label_colour(label: str) -> str:
    """Return the display colour for a three-class sentiment label."""
    return LABEL_STYLE.get(label.lower(), _FALLBACK_COLOUR)


def chunk(items: Sequence[Any], size: int = MAX_CARDS_PER_ROW) -> list[list[Any]]:
    """Split *items* into consecutive groups of at most *size* entries."""
    if size < 1:
        raise ValueError("Chunk size must be at least 1")
    return [list(items[start:start + size]) for start in range(0, len(items), size)]


def has_token_evidence(result: dict) -> bool:
    """Return True when *result* carries renderable aspect-specific evidence."""
    evidence = result.get("token_evidence") or {}
    return evidence.get("status") == "supported" and bool(evidence.get("tokens"))


def first_evidence(results: Sequence[dict]) -> dict | None:
    """Return the first supported evidence payload, or None when none is present.

    The method, caveat and limitation text describe the selected model rather than
    any single aspect, so the page renders them once from this payload.
    """
    for result in results:
        if has_token_evidence(result):
            return result["token_evidence"]
    return None


def comparison_css(comparison: Comparison) -> Any:
    """Build the per-cell CSS frame for a comparison matrix.

    Cells are coloured from the label frame rather than by parsing the display
    text, and an unavailable artifact is muted and italicised instead of being
    coloured as a sentiment.
    """
    return comparison.labels.map(
        lambda label: f"color: {label_colour(label)}" if label else _MISSING_CSS
    )


def style_comparison(comparison: Comparison) -> Any:
    """Return the display frame styled with one colour per sentiment."""
    return comparison.display.style.apply(lambda _frame: comparison_css(comparison), axis=None)


def render_aspect_card(container, result: dict, review: str, show_evidence: bool) -> None:
    """Render one aspect prediction as a compact card inside *container*."""
    with container:
        with st.container(border=True):
            st.caption(result["aspect"])
            st.markdown(
                f'<span style="color: {label_colour(result["label"])}; font-size: 1.35rem; '
                f'font-weight: 600;">{result["label"].title()}</span>'
                f'<span style="opacity: 0.7; font-size: 1.05rem;"> · '
                f'{format_confidence(result["confidence"])}</span>',
                unsafe_allow_html=True,
            )
            if show_evidence and has_token_evidence(result):
                st.markdown(
                    render_token_heatmap_html(review, result["token_evidence"]["tokens"]),
                    unsafe_allow_html=True,
                )


def render_result_grid(results: Sequence[dict], review: str, show_evidence: bool) -> None:
    """Lay out every aspect result as cards, wrapping after `MAX_CARDS_PER_ROW`.

    Rows always allocate the full column count so a partial final row keeps the
    same card width as the rows above it.
    """
    rows = chunk(results)
    columns_per_row = min(len(results), MAX_CARDS_PER_ROW) or 1
    for row in rows:
        columns = st.columns(columns_per_row)
        # A partial final row has fewer results than columns, so this zip is
        # intentionally non-strict.
        for column, result in zip(columns, row, strict=False):
            render_aspect_card(column, result, review, show_evidence)
