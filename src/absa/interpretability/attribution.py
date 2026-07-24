"""Deterministic DistilBERT gradient × input attribution and offset alignment."""

import math
from collections.abc import Sequence

import torch

from .attention import visible_token_spans


def _normalise(values: Sequence[float]) -> list[float]:
    cleaned = [value if math.isfinite(value) and value > 0 else 0.0 for value in values]
    total = sum(cleaned)
    return [value / total for value in cleaned] if total else cleaned


def align_subword_scores(
    review: str,
    offsets: Sequence[Sequence[int]],
    sequence_ids: Sequence[int | None],
    scores: Sequence[float],
) -> list[dict[str, float | int | str]]:
    """Aggregate review wordpieces onto exact visible review token spans."""
    if not (len(offsets) == len(sequence_ids) == len(scores)):
        raise ValueError("offsets, sequence_ids and scores must have equal length")

    review_pieces = [
        (int(offset[0]), int(offset[1]), float(score))
        for offset, sequence_id, score in zip(offsets, sequence_ids, scores)
        if sequence_id == 0 and int(offset[1]) > int(offset[0])
    ]

    aligned: list[dict[str, float | int | str]] = []
    raw_scores: list[float] = []
    for token in visible_token_spans(review):
        start, end = int(token["start"]), int(token["end"])
        overlaps = [
            score
            * (
                max(0, min(piece_end, end) - max(piece_start, start))
                / (piece_end - piece_start)
            )
            for piece_start, piece_end, score in review_pieces
            if piece_start < end and piece_end > start
        ]
        if not overlaps:
            continue
        raw_scores.append(sum(overlaps))
        aligned.append({**token, "subword_count": len(overlaps)})

    for token, score in zip(aligned, _normalise(raw_scores)):
        token["score"] = score
    return aligned


def gradient_x_input_attribution(
    model,
    tokenizer,
    review: str,
    aspect: str,
    *,
    max_length: int = 128,
) -> tuple[torch.Tensor, list[dict[str, float | int | str]]]:
    """Return logits and review-token attribution for the predicted class."""
    encoded = tokenizer(
        review,
        aspect,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    if "offset_mapping" not in encoded or not hasattr(encoded, "sequence_ids"):
        raise ValueError("A fast tokenizer with offset mappings is required for token evidence")

    sequence_ids = encoded.sequence_ids(0)
    offsets = encoded.pop("offset_mapping")[0].detach().cpu().tolist()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    input_ids = encoded.pop("input_ids").to(device)
    attention_mask = encoded.pop("attention_mask").to(device)
    embeddings = model.get_input_embeddings()(input_ids).detach()
    embeddings.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_class = int(logits[0].detach().argmax())
    gradients = torch.autograd.grad(logits[0, predicted_class], embeddings)[0]
    wordpiece_scores = torch.linalg.vector_norm(gradients * embeddings, dim=-1)[0]

    tokens = align_subword_scores(
        review,
        offsets,
        sequence_ids,
        wordpiece_scores.detach().cpu().tolist(),
    )
    return logits.detach().cpu(), tokens
