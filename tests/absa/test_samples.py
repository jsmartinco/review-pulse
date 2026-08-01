"""Tests for the v3 demonstration samples.

The integration tests below verify each sample against the frozen evaluation
outputs, so a sample whose text, aspects or gold polarity drifts from the
official test split fails rather than silently misleading a reader.
"""

import csv
from collections import defaultdict
from pathlib import Path

import pytest

from src.absa.labels import LABELS
from src.absa.samples import SAMPLES, AspectSample, find_sample, get_random_sample


PREDICTIONS = Path("outputs/absa/evaluation/predictions.csv")


def _aspect_list(sample: AspectSample) -> list[str]:
    return [aspect.strip() for aspect in sample.aspects.split(",")]


def test_v3_sample_has_review_and_aspects():
    sample = get_random_sample()
    assert sample in SAMPLES
    assert sample.review
    assert "," in sample.aspects


def test_v3_sample_avoids_visible_review_when_possible():
    sample = get_random_sample(SAMPLES[0].review)
    assert sample.review != SAMPLES[0].review


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s.sentence_id)
def test_gold_covers_every_listed_aspect(sample):
    assert set(_aspect_list(sample)) == set(sample.gold)


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s.sentence_id)
def test_gold_uses_only_the_three_retained_classes(sample):
    assert set(sample.gold.values()) <= set(LABELS)


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s.sentence_id)
def test_scenario_describes_the_sentence_not_an_expected_outcome(sample):
    """Outcome-shaped labels go stale when artifacts are retrained."""
    lowered = sample.scenario.lower()
    for forbidden in ("model", "wrong", "correct", "wins", "fails", "beats", "accuracy"):
        assert forbidden not in lowered, f"{sample.sentence_id} names an outcome: {sample.scenario}"


def test_samples_have_unique_sentence_ids_and_reviews():
    assert len({sample.sentence_id for sample in SAMPLES}) == len(SAMPLES)
    assert len({sample.review for sample in SAMPLES}) == len(SAMPLES)


def test_at_least_one_sample_carries_each_polarity():
    polarities = {label for sample in SAMPLES for label in sample.gold.values()}
    assert polarities == set(LABELS)


def test_find_sample_matches_verbatim_text():
    assert find_sample(SAMPLES[0].review) is SAMPLES[0]
    assert find_sample(f"  {SAMPLES[0].review}  ") is SAMPLES[0]


def test_find_sample_drops_gold_once_the_review_is_edited():
    assert find_sample(SAMPLES[0].review + " Truly awful.") is None
    assert find_sample("") is None
    assert find_sample("A review that was never annotated.") is None


# ---------------------------------------------------------------------------
# Provenance: samples must match the frozen official test split
# ---------------------------------------------------------------------------

def _frozen_gold() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    if not PREDICTIONS.exists():
        pytest.skip("Frozen evaluation predictions are not available locally")
    by_sentence: dict[str, dict[str, str]] = defaultdict(dict)
    reviews: dict[str, str] = {}
    with PREDICTIONS.open() as handle:
        for row in csv.DictReader(handle):
            by_sentence[row["sentence_id"]][row["aspect"]] = row["gold"]
            reviews[row["sentence_id"]] = row["review"]
    return by_sentence, reviews


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s.sentence_id)
def test_sample_matches_the_official_test_split(sample):
    by_sentence, reviews = _frozen_gold()
    assert sample.sentence_id in by_sentence, "sentence_id is not in the official test split"
    assert sample.review == reviews[sample.sentence_id], "review text drifted from the dataset"
    for aspect, gold in sample.gold.items():
        assert aspect in by_sentence[sample.sentence_id], f"{aspect} is not a gold aspect term"
        assert by_sentence[sample.sentence_id][aspect] == gold, f"gold polarity drifted for {aspect}"
