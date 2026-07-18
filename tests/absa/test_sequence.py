from src.absa.tokenization.sequence import UNK, build_vocab, encode


def test_vocab_is_built_only_from_supplied_training_text() -> None:
    vocab = build_vocab(["known token"])
    assert "heldout" not in vocab
    assert encode(["heldout"], vocab)[0, 0].item() == vocab[UNK]
