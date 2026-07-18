"""Manual aspect-input samples for the ReviewPulse v3 demonstration."""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class AspectSample:
    review: str
    aspects: str


SAMPLES = (
    AspectSample("The food was wonderful, but the service was slow and the prices were fair.", "food, service, prices"),
    AspectSample("The pasta was bland, the staff were friendly, and the atmosphere was lovely.", "pasta, staff, atmosphere"),
    AspectSample("The menu is limited, but the desserts are excellent and the restaurant is clean.", "menu, desserts, restaurant"),
)


def get_random_sample(current_review: str = "") -> AspectSample:
    """Return a sample different from the visible review where possible."""
    candidates = [sample for sample in SAMPLES if sample.review != current_review.strip()]
    return random.choice(candidates or list(SAMPLES))
