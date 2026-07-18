"""DLE602 aspect-based sentiment application."""

import streamlit as st

from src.absa.inference.api import predict_aspects
from src.absa.inference.predictors import TfidfAspectPredictor

st.set_page_config(page_title="ReviewPulse v3.0.0", page_icon="favicon.ico", layout="centered")
st.sidebar.image("logo.png", width="content")
st.title("ReviewPulse v3.0.0")
st.caption("DLE602 · three-class aspect-based sentiment analysis")
st.info("Enter one or more aspects manually. The current UI exposes the review-only TF-IDF baseline; aspect-conditioned models are evaluated in the experiment pipeline.")

review = st.text_area("Review", placeholder="The food was great but the service was slow.", height=150)
aspects = st.text_input("Aspects, separated by commas", placeholder="food, service")
if st.button("Classify aspects", type="primary", disabled=not review.strip() or not aspects.strip()):
    results = predict_aspects(review, aspects.split(","), "absa_tfidf", TfidfAspectPredictor())
    for result in results:
        st.metric(result["aspect"], f"{result['label'].title()} · {result['confidence']:.1%}")
    st.caption("TF-IDF uses only the review text. It is intentionally unable to condition on the selected aspect. Token evidence is only shown for supported attention/attribution-capable models.")
