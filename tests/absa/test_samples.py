from src.absa.samples import SAMPLES, get_random_sample


def test_v3_sample_has_review_and_aspects():
    sample = get_random_sample()
    assert sample in SAMPLES
    assert sample.review
    assert "," in sample.aspects


def test_v3_sample_avoids_visible_review_when_possible():
    sample = get_random_sample(SAMPLES[0].review)
    assert sample.review != SAMPLES[0].review
