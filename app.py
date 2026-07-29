"""ReviewPulse project introduction and navigation landing page."""

import streamlit as st

st.set_page_config(page_title="ReviewPulse", page_icon="favicon.svg", layout="centered")

st.sidebar.image("logo.png", width="content")


def render_introduction() -> None:
    st.title("ReviewPulse")
    st.caption("A phased NLP and Deep Learning project · Torrens University Australia")
    st.markdown("""
    ReviewPulse evolves deliberately across assessments: reusable engineering is retained, while each phase introduces a distinct learning task and evaluation contract. Use the sidebar to open either working application.
    """)

    st.subheader("Project roadmap")
    st.table([
        {"Version": "v1.0", "Subject": "ISY503", "Capability": "Binary review sentiment", "Models": "TF-IDF + BiLSTM"},
        {"Version": "v2.3.0", "Subject": "ISY503", "Capability": "Hardened review sentiment", "Models": "TF-IDF + BiLSTM + DistilBERT"},
        {"Version": "v3.0.0", "Subject": "DLE602", "Capability": "Three-class aspect sentiment", "Models": "4 core + exploratory GRU/CNN"},
    ])

    st.subheader("Choose a workspace")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.caption("ISY503 · REVIEW-LEVEL SENTIMENT")
            st.markdown("### ReviewPulse v2.3.0")
            st.write("One binary sentiment label for an entire Amazon review.")
    with right:
        with st.container(border=True):
            st.caption("DLE602 · ASPECT-LEVEL SENTIMENT")
            st.markdown("### ReviewPulse v3.0.0")
            st.write("One positive, neutral or negative label per supplied aspect.")

    st.info("v3 uses SemEval Restaurants and is a new ABSA learning task; it is not a relabelled v2 model.")


navigation = st.navigation([
    st.Page(render_introduction, title="Introduction", icon="🏠", default=True),
    st.Page("pages/1_ReviewPulse_v2_3_0.py", title="ReviewPulse v2.3.0", icon="📈"),
    st.Page("pages/2_ReviewPulse_v3_0_0.py", title="ReviewPulse v3.0.0", icon="🔎"),
])
navigation.run()
