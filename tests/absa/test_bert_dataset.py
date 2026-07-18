import torch

from src.absa.data.schema import AspectExample
from src.absa.tokenization.bert_dataset import AspectPairDataset


class _Tokenizer:
    def __call__(self, reviews, aspects, **kwargs):
        return {"input_ids": torch.ones((len(reviews), 2), dtype=torch.long), "attention_mask": torch.ones((len(reviews), 2), dtype=torch.long)}


def test_pair_dataset_returns_integer_label():
    row = AspectExample("s", "great food", "food", 6, 10, "positive", "train", True)
    item = AspectPairDataset(_Tokenizer(), [row])[0]
    assert item["labels"].item() == 2
