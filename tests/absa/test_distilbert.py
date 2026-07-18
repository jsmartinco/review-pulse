from src.absa.models.distilbert import ABSADistilBERT


def test_absa_distilbert_declares_three_class_factory():
    assert callable(ABSADistilBERT.from_pretrained_absa)
