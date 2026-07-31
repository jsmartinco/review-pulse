"""Tests for the cached v3 aspect predictor loader in src.app.absa_service.

These exercise the real ``st.cache_resource`` decorator rather than a no-op stub,
because the behaviour under test is the caching itself: successful loads must be
reused, and failed loads must keep raising instead of being cached.
"""

import pytest

from src.absa.inference.predictors import ALL_MODEL_OPTIONS
from src.app import absa_service


@pytest.fixture(autouse=True)
def _clear_cache():
    """Isolate each test from cache entries left by the previous one."""
    absa_service.load_aspect_predictor.clear()
    yield
    absa_service.load_aspect_predictor.clear()


def test_successful_load_is_reused_for_the_same_model(monkeypatch):
    calls: list[str] = []

    def _fake_get_predictor(model_name: str) -> object:
        calls.append(model_name)
        return object()

    monkeypatch.setattr(absa_service, "get_predictor", _fake_get_predictor)
    first = absa_service.load_aspect_predictor("absa_atae_lstm")
    second = absa_service.load_aspect_predictor("absa_atae_lstm")
    assert first is second
    assert calls == ["absa_atae_lstm"]


def test_cache_is_keyed_per_model_name(monkeypatch):
    monkeypatch.setattr(absa_service, "get_predictor", lambda model_name: object())
    atae = absa_service.load_aspect_predictor("absa_atae_lstm")
    tfidf = absa_service.load_aspect_predictor("absa_tfidf")
    assert atae is not tfidf


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("missing artifact"), OSError("unreadable"), RuntimeError("corrupt")],
)
def test_failed_load_is_not_cached_and_keeps_raising(monkeypatch, error):
    """A missing artifact must fail loudly every time, never silently succeed later."""
    calls: list[str] = []

    def _always_fails(model_name: str) -> object:
        calls.append(model_name)
        raise error

    monkeypatch.setattr(absa_service, "get_predictor", _always_fails)
    for _ in range(3):
        with pytest.raises(type(error)):
            absa_service.load_aspect_predictor("absa_distilbert")
    assert len(calls) == 3


def test_unknown_model_still_raises_the_controlled_value_error():
    with pytest.raises(ValueError, match="Unknown v3 model"):
        absa_service.load_aspect_predictor("absa_not_a_model")


def test_every_selectable_model_is_a_valid_cache_key(monkeypatch):
    """The page offers ALL_MODEL_OPTIONS, so each name must reach the loader."""
    seen: list[str] = []
    monkeypatch.setattr(
        absa_service,
        "get_predictor",
        lambda model_name: seen.append(model_name) or object(),
    )
    for model_name in ALL_MODEL_OPTIONS:
        absa_service.load_aspect_predictor(model_name)
    assert seen == list(ALL_MODEL_OPTIONS)
