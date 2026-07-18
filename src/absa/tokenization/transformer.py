"""Sentence-pair encoding for aspect-conditioned Transformers."""


def encode_review_aspect_pairs(tokenizer, reviews: list[str], aspects: list[str], *, max_length: int = 128):
    if len(reviews) != len(aspects):
        raise ValueError("reviews and aspects must have equal length")
    return tokenizer(reviews, aspects, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
