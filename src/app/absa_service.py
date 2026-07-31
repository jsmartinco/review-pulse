"""Streamlit service layer for ReviewPulse v3 aspect model artifacts.

Mirrors :mod:`src.app.service` for the DLE602 models. Artifact loading is cached
per model name so switching models or reclassifying does not re-read checkpoints
from disk on every interaction.

The cache decorator lives here rather than in ``src/absa`` so that package stays
free of Streamlit imports and remains testable without the UI stack.
"""

import threading
from typing import Any

import streamlit as st

from src.absa.inference.predictors import get_predictor


class LockedPredictor:
    """Serialise predictions on a predictor shared across Streamlit sessions.

    ``st.cache_resource`` hands the same object to every session and thread. The
    DistilBERT adapter runs a backward pass to produce gradient attribution, so
    two concurrent sessions would mutate one module's gradients at the same
    time. Every cached predictor is wrapped uniformly rather than only the
    transformer, so no adapter silently loses the guarantee later.

    The lock is reentrant so a predictor that re-enters this wrapper on the same
    thread cannot deadlock. It is uncontended in single-user use, and the
    evaluation runner is unaffected because it loads predictors directly from
    :mod:`src.absa.inference.predictors` rather than through this layer.
    """

    def __init__(self, predictor: Any) -> None:
        self._predictor = predictor
        self._lock = threading.RLock()

    @property
    def wrapped(self) -> Any:
        """Return the underlying predictor, for inspection and tests."""
        return self._predictor

    def predict(self, review: str, aspect: str, model_name: str) -> dict:
        """Predict one (review, aspect) pair while holding the shared lock."""
        with self._lock:
            return self._predictor.predict(review, aspect, model_name)


@st.cache_resource(show_spinner="Loading model…")
def load_aspect_predictor(model_name: str) -> LockedPredictor:
    """Return a cached, lock-guarded predictor for *model_name*, keyed by name.

    Streamlit stores return values only. A raised exception propagates to the
    caller and is not cached, so a missing or unreadable artifact keeps failing
    loudly on every attempt instead of poisoning the cache with a broken object.
    Verified against Streamlit 1.59.2: three consecutive failing loads invoke the
    underlying loader three times.
    """
    return LockedPredictor(get_predictor(model_name))
