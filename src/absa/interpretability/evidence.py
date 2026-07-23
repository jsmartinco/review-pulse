"""Shared, intentionally limited token-evidence payloads."""

from collections.abc import Sequence


EVIDENCE_CAVEAT = (
    "Token scores are indicative evidence, not model reasoning or a causal explanation."
)
ATTENTION_METHOD = "ATAE-LSTM aspect-conditioned attention"
ATTRIBUTION_METHOD = "Gradient × input attribution for the predicted class"
ATTENTION_LIMITATIONS = (
    "Attention weights show where this model concentrated probability mass; "
    "they do not establish which tokens caused the prediction."
)
ATTRIBUTION_LIMITATIONS = (
    "Gradient × input is local to this checkpoint, input and predicted class; "
    "it can be sensitive to model gradients and is not a faithful causal explanation."
)


def supported_evidence(
    *,
    aspect: str,
    method: str,
    tokens: Sequence[dict],
    limitations: str,
) -> dict:
    """Build the stable supported-evidence contract used by inference and UI."""
    return {
        "status": "supported",
        "aspect": aspect,
        "method": method,
        "tokens": list(tokens),
        "caveat": EVIDENCE_CAVEAT,
        "limitations": limitations,
    }


def unsupported_evidence(model_name: str) -> dict:
    """Make absence of token evidence explicit for review-only baselines."""
    return {
        "status": "unsupported",
        "method": None,
        "tokens": [],
        "caveat": EVIDENCE_CAVEAT,
        "limitations": (
            f"{model_name} is a review-only baseline and does not expose "
            "aspect-specific token evidence."
        ),
    }
