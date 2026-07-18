from src.absa.data.schema import AspectExample
from src.absa.evaluation import compute_metrics, mixed_polarity_multi_aspect


def _row(sentence_id: str, aspect: str, label: str) -> AspectExample:
    return AspectExample(sentence_id, "x", aspect, 0, 1, label, "test", True)


def test_metrics_use_fixed_three_class_order() -> None:
    result = compute_metrics(["negative", "neutral", "positive"], ["negative", "positive", "positive"])
    assert result["label_order"] == ["negative", "neutral", "positive"]
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
    assert 0 <= result["macro_f1"] <= 1


def test_mixed_polarity_subset_uses_gold_labels_not_conflict() -> None:
    rows = [_row("mixed", "food", "positive"), _row("mixed", "service", "negative"), _row("same", "food", "positive"), _row("same", "staff", "positive"), _row("original-conflict", "room", "conflict")]
    assert [row.sentence_id for row in mixed_polarity_multi_aspect(rows)] == ["mixed", "mixed"]
