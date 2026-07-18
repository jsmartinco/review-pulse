"""Aspect-based sentiment analysis (ABSA) namespace for ReviewPulse v3.

The v3 implementation is intentionally isolated from the legacy binary
sentiment pipeline until shared contracts are proven stable.
"""

from .labels import ID_TO_LABEL, LABELS, LABEL_TO_ID

__all__ = ["ID_TO_LABEL", "LABELS", "LABEL_TO_ID"]
