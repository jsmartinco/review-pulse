"""Canonical records retained between SemEval parsing and v3 modelling."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AspectExample:
    sentence_id: str
    review_raw: str
    aspect: str
    aspect_from: int
    aspect_to: int
    label: str
    source_split: str
    offset_valid: bool
