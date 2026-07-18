"""Contracts for the fixed ABSA three-class mapping."""

from src.absa import ID_TO_LABEL, LABELS, LABEL_TO_ID


def test_labels_have_the_canonical_order() -> None:
    assert LABELS == ("negative", "neutral", "positive")


def test_label_mappings_are_bijective() -> None:
    assert LABEL_TO_ID == {"negative": 0, "neutral": 1, "positive": 2}
    assert ID_TO_LABEL == {0: "negative", 1: "neutral", 2: "positive"}
