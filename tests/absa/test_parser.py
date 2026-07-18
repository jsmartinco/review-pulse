from pathlib import Path

import pytest
from defusedxml.common import EntitiesForbidden

from src.absa.data.audit import audit_examples
from src.absa.data.parser import parse_aspect_examples


def test_parser_retains_offsets_and_reports_invalid_alignment(tmp_path: Path) -> None:
    xml = tmp_path / "sample.xml"
    xml.write_text(
        '<sentences><sentence id="s1"><text>Great food, slow service.</text><aspectTerms>'
        '<aspectTerm term="food" polarity="positive" from="6" to="10"/>'
        '<aspectTerm term="service" polarity="negative" from="17" to="24"/>'
        '<aspectTerm term="wrong" polarity="neutral" from="0" to="5"/>'
        '</aspectTerms></sentence></sentences>', encoding="utf-8"
    )
    examples = parse_aspect_examples(xml, "train")
    assert [example.offset_valid for example in examples] == [True, True, False]
    report = audit_examples(examples)
    assert report["polarity_counts"] == {"negative": 1, "neutral": 1, "positive": 1}
    assert report["offset_invalid"] == 1


def test_parser_rejects_xml_entities(tmp_path: Path) -> None:
    xml = tmp_path / "unsafe.xml"
    xml.write_text(
        '<!DOCTYPE sentences [<!ENTITY expansion "unsafe">]>'
        '<sentences><sentence id="s1"><text>&expansion;</text></sentence></sentences>',
        encoding="utf-8",
    )
    with pytest.raises(EntitiesForbidden):
        parse_aspect_examples(xml, "train")
