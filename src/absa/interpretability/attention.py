"""Render ATAE attention as caveated, indicative token-level evidence."""

from ..tokenization.sequence import tokens


EVIDENCE_CAVEAT = "Attention weights are indicative token-level evidence, not causal model reasoning."


def align_attention(review: str, weights, *, max_length: int = 80) -> list[dict[str, float | str]]:
    """Align visible tokenizer tokens with non-padding attention weights."""
    visible = tokens(review)[:max_length]
    values = weights.detach().cpu().tolist() if hasattr(weights, "detach") else list(weights)
    return [{"token": token, "weight": float(values[index])} for index, token in enumerate(visible)]
