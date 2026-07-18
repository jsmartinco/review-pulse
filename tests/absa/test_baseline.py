from src.absa.models.baseline import build_baseline


def test_baseline_is_multiclass() -> None:
    model = build_baseline()
    model.fit(["good food", "bad service", "plain room"], ["positive", "negative", "neutral"])
    assert set(model.predict(["good service"])) <= {"negative", "neutral", "positive"}
