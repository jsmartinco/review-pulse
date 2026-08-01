"""Tests for the cached v3 aspect predictor loader in src.app.absa_service.

These exercise the real ``st.cache_resource`` decorator rather than a no-op stub,
because the behaviour under test is the caching itself: successful loads must be
reused, and failed loads must keep raising instead of being cached.
"""

import threading

import pytest

from src.absa.inference.predictors import ALL_MODEL_OPTIONS
from src.app import absa_service
from src.app.absa_service import LockedPredictor


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


# ---------------------------------------------------------------------------
# Thread safety of the shared cached predictor
# ---------------------------------------------------------------------------

THREADS = 4


class _BarrierProbe:
    """Reports whether callers were ever inside predict() at the same time.

    Every call waits on a barrier sized for the full thread count. Without
    serialisation all threads arrive and the barrier releases, so `concurrent`
    becomes True. Under a lock only one thread can be inside at a time, the
    barrier can never fill, and it breaks on timeout instead. This makes the
    check deterministic in both directions rather than relying on a race
    happening to occur.
    """

    def __init__(self, threads: int = THREADS) -> None:
        self.barrier = threading.Barrier(threads)
        self.concurrent = False

    def predict(self, review: str, aspect: str, model_name: str) -> dict:
        try:
            self.barrier.wait(timeout=0.75)
            self.concurrent = True
        except threading.BrokenBarrierError:
            pass
        return {"aspect": aspect, "label": "positive", "confidence": 1.0, "model": model_name}


def _hammer(predictor, threads: int = THREADS) -> list[dict]:
    results: list[dict] = []
    lock = threading.Lock()

    def _call() -> None:
        outcome = predictor.predict("Great food", "food", "absa_distilbert")
        with lock:
            results.append(outcome)

    workers = [threading.Thread(target=_call) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    return results


def test_unguarded_predictor_is_entered_concurrently():
    """Negative control: without the lock the probe observes real overlap."""
    probe = _BarrierProbe()
    _hammer(probe)
    assert probe.concurrent is True


def test_locked_predictor_serialises_concurrent_sessions():
    """st.cache_resource shares one object across sessions, so predict must serialise."""
    probe = _BarrierProbe()
    _hammer(LockedPredictor(probe))
    assert probe.concurrent is False


def test_locked_predictor_returns_every_result_under_concurrency():
    probe = _BarrierProbe()
    results = _hammer(LockedPredictor(probe))
    assert len(results) == THREADS
    assert all(item["label"] == "positive" for item in results)


def test_locked_predictor_lock_is_reentrant():
    """A predictor that re-enters the wrapper on one thread must not deadlock."""
    wrapper: dict[str, LockedPredictor] = {}

    class _Reentrant:
        def __init__(self) -> None:
            self.depth = 0

        def predict(self, review: str, aspect: str, model_name: str) -> dict:
            self.depth += 1
            if self.depth == 1:
                wrapper["value"].predict(review, aspect, model_name)
            return {"aspect": aspect, "label": "neutral", "confidence": 0.5, "model": model_name}

    inner = _Reentrant()
    wrapper["value"] = LockedPredictor(inner)
    wrapper["value"].predict("Great food", "food", "absa_atae_lstm")
    assert inner.depth == 2


def test_locked_predictor_exposes_the_wrapped_object():
    inner = _BarrierProbe()
    assert LockedPredictor(inner).wrapped is inner


def test_cached_loader_returns_a_locked_predictor(monkeypatch):
    sentinel = _BarrierProbe()
    monkeypatch.setattr(absa_service, "get_predictor", lambda model_name: sentinel)
    loaded = absa_service.load_aspect_predictor("absa_distilbert")
    assert isinstance(loaded, LockedPredictor)
    assert loaded.wrapped is sentinel


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
