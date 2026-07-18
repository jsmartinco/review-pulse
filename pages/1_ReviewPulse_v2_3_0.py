"""ISY503 legacy review-sentiment application."""

import streamlit as st

from src.app.service import DISTILBERT_UNAVAILABLE_MSG, MODEL_OPTIONS, is_distilbert_available, warm_up_model
from src.config import MODEL_DISTILBERT
from src.app.samples import get_random_sample

st.title("ReviewPulse v2.3.0")
st.caption("ISY503 · binary sentiment for a complete Amazon product review")

model_name = st.radio("Model", list(MODEL_OPTIONS), format_func=lambda key: MODEL_OPTIONS[key])
available = model_name != MODEL_DISTILBERT or is_distilbert_available()
if model_name == MODEL_DISTILBERT and not available: st.warning(DISTILBERT_UNAVAILABLE_MSG, icon="⚠️")
if model_name != MODEL_DISTILBERT: warm_up_model(model_name)

if "v2_review" not in st.session_state: st.session_state.v2_review = ""
def sample(): st.session_state.v2_review = get_random_sample(st.session_state.v2_review)
st.button("💡 Generate sample", on_click=sample)
review = st.text_area("Review text", key="v2_review", height=160)
if st.button("Classify review", type="primary", disabled=not review.strip() or not available):
    from src.inference import predict_sentiment
    result = predict_sentiment(review.strip(), model_name=model_name)
    st.success(result["label"])
    st.metric("Confidence", f"{result['confidence']:.1%}")
    st.json(result)
