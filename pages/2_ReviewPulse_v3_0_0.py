"""DLE602 aspect-based sentiment application."""

import streamlit as st

from src.absa.inference.api import predict_aspects
from src.absa.inference.predictors import MODEL_OPTIONS, get_predictor
from src.absa.interpretability.heatmap import render_token_heatmap_html
from src.absa.samples import get_random_sample

st.title("ReviewPulse v3.0.0")
st.caption("DLE602 · three-class aspect-based sentiment analysis")
st.info(
    "Enter one or more aspects manually. Review-only models intentionally receive "
    "only the review; ATAE-LSTM and DistilBERT receive the review-and-aspect pair."
)

model_name = st.selectbox("Model", list(MODEL_OPTIONS), format_func=MODEL_OPTIONS.__getitem__)


def load_sample() -> None:
    sample = get_random_sample(st.session_state.get("absa_review", ""))
    st.session_state["absa_review"] = sample.review
    st.session_state["absa_aspects"] = sample.aspects


st.button("💡 Generate sample", on_click=load_sample)
review = st.text_area("Review", placeholder="The food was great but the service was slow.", height=150, key="absa_review")
aspects = st.text_input("Aspects, separated by commas", placeholder="food, service", key="absa_aspects")
if st.button(
    "Classify aspects",
    type="primary",
    disabled=not review.strip() or not aspects.strip(),
):
    try:
        results = predict_aspects(review, aspects.split(","), model_name, get_predictor(model_name))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        st.error(f"The selected model is unavailable: {error}")
    else:
        for result in results:
            st.metric(result["aspect"], f"{result['label'].title()} · {result['confidence']:.1%}")
            evidence = result["token_evidence"]
            if evidence["status"] == "supported":
                st.markdown(f"**{evidence['method']}**")
                st.markdown(
                    render_token_heatmap_html(review, evidence["tokens"]),
                    unsafe_allow_html=True,
                )
                st.caption("Darker shading indicates a larger score within this aspect view.")
                st.caption(evidence["caveat"])
                st.caption(f"Limitation: {evidence['limitations']}")
            else:
                st.info(evidence["limitations"])
