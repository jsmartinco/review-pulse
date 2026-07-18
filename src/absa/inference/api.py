"""Public multi-aspect inference contract for ReviewPulse v3."""


def normalise_aspects(aspects: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for aspect in aspects:
        cleaned = aspect.strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    if not result:
        raise ValueError("Provide at least one non-empty aspect")
    return result


def predict_aspects(review: str, aspects: list[str], model_name: str, predictor) -> list[dict]:
    cleaned_review = review.strip()
    if not cleaned_review:
        raise ValueError("Review must not be empty")
    return [predictor.predict(cleaned_review, aspect, model_name) for aspect in normalise_aspects(aspects)]
