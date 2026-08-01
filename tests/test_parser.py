# .venv/bin/pytest tests/test_parser.py -v

from pathlib import Path

import pandas as pd
import pytest

from src.data.parser import load_all_domains, load_unlabeled_domains, parse_review_file

# Minimal pseudo-XML fixture — mirrors the real .review file format
SAMPLE_VALID = """
<review>
<unique_id>abc:positive_test</unique_id>
<asin>B00001</asin>
<rating>5.0</rating>
<title>Great product</title>
<review_text>This is an excellent product. I loved every bit of it.</review_text>
</review>
"""

SAMPLE_MISSING_TEXT = """
<review>
<unique_id>abc:no_text</unique_id>
<rating>3.0</rating>
</review>
"""

SAMPLE_MISSING_RATING = """
<review>
<unique_id>abc:no_rating</unique_id>
<review_text>Good but the rating field is absent.</review_text>
</review>
"""


def test_parse_extracts_text_and_rating(tmp_path: Path) -> None:
    f = tmp_path / "positive.review"
    f.write_text(SAMPLE_VALID)
    records = parse_review_file(f, label=1)
    assert len(records) == 1
    assert records[0]["text"] == "This is an excellent product. I loved every bit of it."
    assert records[0]["rating"] == 5.0
    assert records[0]["label"] == 1
    assert records[0]["source_file"] == "positive.review"


def test_parse_skips_review_without_text(tmp_path: Path) -> None:
    f = tmp_path / "negative.review"
    f.write_text(SAMPLE_MISSING_TEXT)
    records = parse_review_file(f, label=0)
    assert records == []


def test_parse_handles_missing_rating(tmp_path: Path) -> None:
    f = tmp_path / "positive.review"
    f.write_text(SAMPLE_MISSING_RATING)
    records = parse_review_file(f, label=1)
    assert len(records) == 1
    assert records[0]["rating"] is None


def test_parse_multiple_reviews(tmp_path: Path) -> None:
    content = SAMPLE_VALID + SAMPLE_MISSING_TEXT + SAMPLE_MISSING_RATING
    f = tmp_path / "positive.review"
    f.write_text(content)
    records = parse_review_file(f, label=1)
    # SAMPLE_MISSING_TEXT has no review_text — skipped; other two are valid
    assert len(records) == 2


# --- aggregate loading against a synthetic four-domain corpus ---

@pytest.fixture
def synthetic_legacy_dataset(tmp_path: Path) -> pd.DataFrame:
    for domain in ("books", "dvd", "electronics", "kitchen_&_housewares"):
        domain_dir = tmp_path / domain
        domain_dir.mkdir()
        for filename, rating, text in (
            ("positive.review", 5.0, f"Excellent {domain} product."),
            ("negative.review", 1.0, f"Terrible {domain} product."),
        ):
            (domain_dir / filename).write_text(
                "<review>"
                f"<rating>{rating}</rating>"
                f"<review_text>{text}</review_text>"
                "</review>",
                encoding="utf-8",
            )

    dataset = load_all_domains(data_dir=tmp_path)
    assert not dataset.empty, "Synthetic review files exist but the parser returned no records"
    return dataset


def test_load_all_domains_returns_dataframe(synthetic_legacy_dataset: pd.DataFrame) -> None:
    df = synthetic_legacy_dataset
    assert isinstance(df, pd.DataFrame)


def test_load_all_domains_has_required_columns(synthetic_legacy_dataset: pd.DataFrame) -> None:
    df = synthetic_legacy_dataset
    assert set(df.columns) >= {"text", "rating", "label", "domain", "source_file"}


def test_load_all_domains_four_domains(synthetic_legacy_dataset: pd.DataFrame) -> None:
    df = synthetic_legacy_dataset
    assert df["domain"].nunique() == 4


def test_load_all_domains_binary_labels(synthetic_legacy_dataset: pd.DataFrame) -> None:
    df = synthetic_legacy_dataset
    assert set(df["label"].unique()) == {0, 1}


def test_load_all_domains_no_empty_text(synthetic_legacy_dataset: pd.DataFrame) -> None:
    df = synthetic_legacy_dataset
    assert df["text"].str.strip().ne("").all()


def test_load_all_domains_count(synthetic_legacy_dataset: pd.DataFrame) -> None:
    assert len(synthetic_legacy_dataset) == 8


def test_load_unlabeled_domains_schema() -> None:
    df = load_unlabeled_domains()
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert set(df.columns) >= {"text", "rating", "label", "domain", "source_file"}
        assert set(df["label"].unique()) == {-1}
        assert df["source_file"].eq("unlabeled.review").all()
