"""Gold-label subsets used to test aspect conditioning."""

from collections import defaultdict

from ..labels import LABELS
from ..data.schema import AspectExample


def mixed_polarity_multi_aspect(examples: list[AspectExample]) -> list[AspectExample]:
    """Return retained examples from sentences with >=2 different gold polarities."""
    by_sentence: dict[str, list[AspectExample]] = defaultdict(list)
    for example in examples:
        if example.label in LABELS:
            by_sentence[example.sentence_id].append(example)
    return [
        example
        for rows in by_sentence.values()
        if len(rows) >= 2 and len({row.label for row in rows}) >= 2
        for example in rows
    ]
