"""Three-class DistilBERT sentence-pair classifier for v3 ABSA."""

from transformers import DistilBertForSequenceClassification


class ABSADistilBERT(DistilBertForSequenceClassification):
    """DistilBERT configured for tokenizer(review, aspect) pair inputs."""

    @classmethod
    def from_pretrained_absa(cls, model_name: str = "distilbert-base-uncased", **kwargs):
        return cls.from_pretrained(model_name, num_labels=3, **kwargs)
