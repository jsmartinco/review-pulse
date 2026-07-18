"""Fixed three-class label contract for ReviewPulse v3."""

LABELS: tuple[str, str, str] = ("negative", "neutral", "positive")
LABEL_TO_ID: dict[str, int] = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL: dict[int, str] = {index: label for label, index in LABEL_TO_ID.items()}
