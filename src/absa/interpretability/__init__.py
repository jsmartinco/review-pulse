"""Indicative token-evidence alignment for supported ABSA models."""

from .attention import align_attention, visible_token_spans
from .attribution import align_subword_scores, gradient_x_input_attribution
from .evidence import EVIDENCE_CAVEAT, supported_evidence, unsupported_evidence
from .heatmap import render_token_heatmap_html

__all__ = [
    "EVIDENCE_CAVEAT",
    "align_attention",
    "align_subword_scores",
    "gradient_x_input_attribution",
    "render_token_heatmap_html",
    "supported_evidence",
    "unsupported_evidence",
    "visible_token_spans",
]
