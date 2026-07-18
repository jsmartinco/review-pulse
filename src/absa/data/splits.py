"""Leakage-safe partitioning for aspect-level SemEval examples."""

from dataclasses import dataclass

from sklearn.model_selection import GroupShuffleSplit

from ..labels import LABELS
from .schema import AspectExample


@dataclass(frozen=True)
class ABSASplits:
    train: tuple[AspectExample, ...]
    development: tuple[AspectExample, ...]
    test: tuple[AspectExample, ...]

    @property
    def label_distributions(self) -> dict[str, dict[str, int]]:
        """Expose label counts so grouped split balance is visible to callers."""
        return {
            "train": label_distribution(self.train),
            "development": label_distribution(self.development),
            "test": label_distribution(self.test),
        }


def label_distribution(examples: tuple[AspectExample, ...]) -> dict[str, int]:
    """Return counts for every core label, including labels absent from a partition."""
    return {label: sum(example.label == label for example in examples) for label in LABELS}


def retained_examples(examples: list[AspectExample]) -> list[AspectExample]:
    """Keep the core three-class task; audit reports original conflict separately."""
    return [example for example in examples if example.label in LABELS]


def assert_disjoint_sentence_ids(*partitions: tuple[AspectExample, ...]) -> None:
    """Raise if a sentence is represented in more than one partition."""
    seen: set[str] = set()
    for partition in partitions:
        ids = {example.sentence_id for example in partition}
        overlap = seen & ids
        if overlap:
            raise ValueError(f"Sentence leakage across partitions: {sorted(overlap)}")
        seen.update(ids)


def split_official_data(
    train_examples: list[AspectExample],
    test_examples: list[AspectExample],
    *,
    seed: int = 42,
    development_fraction: float = 0.2,
) -> ABSASplits:
    """Create train/development from official train data and preserve official test."""
    if not 0 < development_fraction < 1:
        raise ValueError("development_fraction must be between 0 and 1")
    train_rows = retained_examples(train_examples)
    test_rows = retained_examples(test_examples)
    if not train_rows or not test_rows:
        raise ValueError("Train and official test must each contain retained examples")
    groups = [example.sentence_id for example in train_rows]
    splitter = GroupShuffleSplit(n_splits=1, test_size=development_fraction, random_state=seed)
    train_indices, development_indices = next(splitter.split(train_rows, groups=groups))
    result = ABSASplits(
        train=tuple(train_rows[index] for index in train_indices),
        development=tuple(train_rows[index] for index in development_indices),
        test=tuple(test_rows),
    )
    assert_disjoint_sentence_ids(result.train, result.development, result.test)
    return result
