from src.absa.tokenization.transformer import encode_review_aspect_pairs


class _Tokenizer:
    def __call__(self, reviews, aspects, **kwargs):
        return {"reviews": reviews, "aspects": aspects, **kwargs}


def test_pair_encoder_passes_review_and_aspect_as_distinct_sequences() -> None:
    result = encode_review_aspect_pairs(_Tokenizer(), ["food good"], ["food"])
    assert result["reviews"] == ["food good"] and result["aspects"] == ["food"]
    assert result["return_tensors"] == "pt"
