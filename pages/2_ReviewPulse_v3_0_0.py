"""DLE602 aspect-based sentiment application."""

import streamlit as st

from src.absa.inference.api import predict_aspects
from src.absa.inference.predictors import ALL_MODEL_OPTIONS, exposes_token_evidence
from src.absa.interpretability.evidence import unsupported_evidence
from src.absa.samples import get_random_sample
from src.app.absa_results import first_evidence, render_result_grid
from src.app.absa_service import load_aspect_predictor

st.title("ReviewPulse v3.0.0")
st.caption("DLE602 · three-class aspect-based sentiment analysis")
st.info(
    "Enter one or more aspects manually. Review-only models intentionally receive "
    "only the review; ATAE-LSTM and DistilBERT receive the review-and-aspect pair. "
    "GRU and Text CNN are exploratory extensions and do not support token evidence."
)

model_name = st.selectbox(
    "Model",
    list(ALL_MODEL_OPTIONS),
    format_func=ALL_MODEL_OPTIONS.__getitem__,
)

# The review-only limitation describes the selected model, not any single aspect,
# so it renders once here instead of repeating under every result card.
model_exposes_evidence = exposes_token_evidence(model_name)
if not model_exposes_evidence:
    st.caption(unsupported_evidence(ALL_MODEL_OPTIONS[model_name])["limitations"])


def load_sample() -> None:
    """Fill both inputs with a curated multi-aspect demonstration review."""
    sample = get_random_sample(st.session_state.get("absa_review", ""))
    st.session_state["absa_review"] = sample.review
    st.session_state["absa_aspects"] = sample.aspects


st.button("💡 Generate sample", on_click=load_sample)
review = st.text_area(
    "Review",
    placeholder="The food was great but the service was slow.",
    height=150,
    key="absa_review",
)
aspects = st.text_input(
    "Aspects, separated by commas",
    placeholder="food, service",
    key="absa_aspects",
)
if st.button(
    "Classify aspects",
    type="primary",
    disabled=not review.strip() or not aspects.strip(),
):
    try:
        predictor = load_aspect_predictor(model_name)
        results = predict_aspects(review, aspects.split(","), model_name, predictor)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        st.error(f"The selected model is unavailable: {error}")
    else:
        render_result_grid(results, review, model_exposes_evidence)
        evidence = first_evidence(results) if model_exposes_evidence else None
        if evidence is not None:
            st.caption(f"**{evidence['method']}** · darker shading indicates a larger score within this aspect view.")
            st.caption(evidence["caveat"])
            st.caption(f"Limitation: {evidence['limitations']}")
