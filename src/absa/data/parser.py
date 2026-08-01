"""Parse SemEval-2014 Restaurants aspect-term annotations."""

from pathlib import Path

from defusedxml import ElementTree as etree

from .schema import AspectExample


ACQUISITION_HINT = (
    "SemEval-2014 Restaurants data is not redistributed with this repository. "
    "Obtain the official XML from https://alt.qcri.org/semeval2014/task4/ and "
    "place it with:\n"
    "    python scripts/prepare_semeval_restaurants.py "
    "--train <Restaurants_Train_v2.xml> --test <Restaurants_Test_Gold.xml>\n"
    "See docs/dle602-a3/semeval-restaurants.md for the full procedure."
)


def parse_aspect_examples(path: Path, source_split: str) -> list[AspectExample]:
    """Return one canonical record per annotated aspect term in *path*.

    Raises:
        FileNotFoundError: when the dataset is absent, carrying acquisition
            instructions rather than a bare path, since a reader following the
            documented commands reaches this before obtaining the licensed data.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing SemEval data file: {path}\n\n{ACQUISITION_HINT}")
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
