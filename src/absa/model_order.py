"""Canonical and exploratory ReviewPulse v3 model ordering contracts."""

CORE_MODEL_ORDER = ("tfidf", "target_lstm", "atae_lstm", "distilbert")
OPTIONAL_MODEL_ORDER = ("target_gru", "text_cnn")
SIX_MODEL_ORDER = (
    "tfidf",
    "target_lstm",
    "target_gru",
    "text_cnn",
    "atae_lstm",
    "distilbert",
)
