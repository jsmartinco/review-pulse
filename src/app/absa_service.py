"""Streamlit service layer for ReviewPulse v3 aspect model artifacts.

Mirrors :mod:`src.app.service` for the DLE602 models. Artifact loading is cached
per model name so switching models or reclassifying does not re-read checkpoints
from disk on every interaction.

The cache decorator lives here rather than in ``src/absa`` so that package stays
free of Streamlit imports and remains testable without the UI stack.
"""

from typing import Any

import streamlit as st

from src.absa.inference.predictors import get_predictor


@st.cache_resource(show_spinner="Loading model…")
def load_aspect_predictor(model_name: str) -> Any:
    """Return a cached predictor for *model_name*, keyed by that name.

    Streamlit stores return values only. A raised exception propagates to the
    caller and is not cached, so a missing or unreadable artifact keeps failing
    loudly on every attempt instead of poisoning the cache with a broken object.
    Verified against Streamlit 1.59.2: three consecutive failing loads invoke the
    underlying loader three times.
    """
    return get_predictor(model_name)
