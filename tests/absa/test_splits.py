import pytest

from src.absa.data.schema import AspectExample
from src.absa.data.splits import assert_disjoint_sentence_ids, split_official_data


def _example(sentence_id: str, label: str = "positive") -> AspectExample:
    return AspectExample(sentence_id, "food and service", "food", 0, 4, label, "train", True)


def test_grouped_split_is_deterministic_and_keeps_sentence_aspects_together() -> None:
    train = [_example(f"s{index}", label) for index, label in enumerate(("positive", "negative", "neutral") * 4)]
    train.extend([AspectExample("s0", "food and service", "service", 9, 16, "negative", "train", True)])
    test = [_example("official-test", "positive")]
    first = split_official_data(train, test, seed=7, development_fraction=0.25)
    second = split_official_data(train, test, seed=7, development_fraction=0.25)
    assert first == second
    train_ids = {item.sentence_id for item in first.train}
    dev_ids = {item.sentence_id for item in first.development}
    assert ("s0" in train_ids) != ("s0" in dev_ids)
    assert first.test == tuple(test)
    assert set(first.label_distributions["development"]) == {"negative", "neutral", "positive"}


def test_overlap_assertion_fails_loudly() -> None:
    row = _example("duplicate")
    with pytest.raises(ValueError, match="duplicate"):
        assert_disjoint_sentence_ids((row,), (row,))
