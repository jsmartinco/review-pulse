"""Parse SemEval-2014 Restaurants aspect-term annotations."""

import xml.etree.ElementTree as etree
from pathlib import Path

from .schema import AspectExample


def parse_aspect_examples(path: Path, source_split: str) -> list[AspectExample]:
    """Return one canonical record per annotated aspect term in *path*."""
    root = etree.parse(path).getroot()
    if root.tag != "sentences":
        raise ValueError(f"Expected <sentences> root in {path}")
    examples: list[AspectExample] = []
    for sentence in root.findall("sentence"):
        sentence_id = sentence.get("id")
        text = sentence.findtext("text")
        if not sentence_id or text is None:
            raise ValueError(f"Sentence missing id or text in {path}")
        for term in sentence.findall("./aspectTerms/aspectTerm"):
            aspect = term.get("term")
            polarity = term.get("polarity")
            try:
                start, end = int(term.get("from", "")), int(term.get("to", ""))
            except ValueError as exc:
                raise ValueError(f"Aspect has invalid offsets in sentence {sentence_id}") from exc
            if not aspect or not polarity:
                raise ValueError(f"Aspect missing term or polarity in sentence {sentence_id}")
            examples.append(AspectExample(
                sentence_id=sentence_id, review_raw=text, aspect=aspect,
                aspect_from=start, aspect_to=end, label=polarity,
                source_split=source_split, offset_valid=text[start:end] == aspect,
            ))
    return examples
