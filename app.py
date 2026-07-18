"""ReviewPulse project introduction and navigation landing page."""

from PIL import Image
import streamlit as st

st.set_page_config(page_title="ReviewPulse", page_icon="favicon.ico", layout="centered")

st.sidebar.image("logo.png", width="content")
st.title("ReviewPulse")
st.caption("A phased NLP and Deep Learning project · Torrens University Australia")
st.image(Image.open("logo.png"), width=180)

st.markdown("""
ReviewPulse evolves deliberately across assessments: reusable engineering is retained, while each phase introduces a distinct learning task and evaluation contract. Use the sidebar to open either working application.
""")

st.subheader("Project roadmap")
st.table([
    {"Version": "v1.0", "Subject": "ISY503", "Capability": "Binary review sentiment", "Models": "TF-IDF + BiLSTM"},
    {"Version": "v2.3.0", "Subject": "ISY503", "Capability": "Hardened review sentiment", "Models": "TF-IDF + BiLSTM + DistilBERT"},
    {"Version": "v3.0.0", "Subject": "DLE602", "Capability": "Three-class aspect sentiment", "Models": "TF-IDF + LSTM + ATAE-LSTM + DistilBERT"},
])

st.subheader("Choose a workspace")
left, right = st.columns(2)
with left:
    st.markdown("### ReviewPulse v2.3.0")
    st.write("The current ISY503 product: one binary sentiment label for an entire Amazon review.")
with right:
    st.markdown("### ReviewPulse v3.0.0")
    st.write("The DLE602 implementation: one positive, neutral or negative label per supplied aspect.")

st.info("v3 uses SemEval Restaurants and is a new ABSA learning task; it is not a relabelled v2 model.")
