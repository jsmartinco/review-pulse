import torch

from src.absa.training import runner as training_runner


def _metrics(model: str) -> dict[str, object]:
    return {
        "config": {"model": model},
        "training_seconds": 1.25,
        "development": {"accuracy": 0.5, "macro_f1": 0.4},
        "test": {"accuracy": 0.6, "macro_f1": 0.5},
    }


def test_training_runner_saves_all_models_with_shared_provenance(monkeypatch, tmp_path) -> None:
    saved: dict[str, dict[str, object]] = {}
    monkeypatch.setattr(
        training_runner,
        "train_baseline",
        lambda *_args, **_kwargs: (object(), _metrics("tfidf_baseline")),
    )
    monkeypatch.setattr(
        training_runner,
        "train_target_lstm",
        lambda *_args, **_kwargs: (object(), {"token": 1}, _metrics("target_lstm")),
    )
    monkeypatch.setattr(
        training_runner,
        "train_atae_lstm",
        lambda *_args, **_kwargs: (object(), {"token": 1}, _metrics("atae_lstm")),
    )
    monkeypatch.setattr(
        training_runner,
        "train_distilbert",
        lambda *_args, **_kwargs: (object(), object(), _metrics("distilbert")),
    )
    monkeypatch.setattr(
        training_runner,
        "save_baseline",
        lambda _model, metrics, _output: saved.update(tfidf=metrics),
    )
    monkeypatch.setattr(
        training_runner,
        "save_target_lstm",
        lambda _model, _vocab, metrics, _output: saved.update(target_lstm=metrics),
    )
    monkeypatch.setattr(
        training_runner,
        "save_atae_lstm",
        lambda _model, _vocab, metrics, _output: saved.update(atae_lstm=metrics),
    )
    monkeypatch.setattr(
        training_runner,
        "save_distilbert",
        lambda _model, _tokenizer, metrics, _output: saved.update(distilbert=metrics),
    )

    provenance = {"git_commit": "abc123", "test_sha256": "fixture"}
    completed = training_runner.train_models(
        [object()],
        [object()],
        tmp_path,
        distilbert_device=torch.device("cpu"),
        provenance=provenance,
    )

    assert list(completed) == list(training_runner.MODEL_ORDER)
    assert saved.keys() == completed.keys()
    assert all(metrics["provenance"] == provenance for metrics in completed.values())


def test_training_runner_rejects_unknown_model_before_training(tmp_path) -> None:
    try:
        training_runner.train_models([], [], tmp_path, models=("cnn",))
    except ValueError as error:
        assert "cnn" in str(error)
    else:
        raise AssertionError("Unknown models must not silently enter the comparison")
