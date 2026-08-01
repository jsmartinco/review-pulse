"""Demonstration samples for the ReviewPulse v3 interface.

Every sample is a verbatim sentence from the official SemEval-2014 Task 4
Restaurants test split, carrying the dataset's own gold polarity for each
annotated aspect term. Quoting six sentences for illustration is not
redistribution of the corpus: the raw XML stays out of version control and must
be obtained from the official source, as documented in
``docs/dle602-a3/semeval-restaurants.md``.

    Pontiki, M., Galanis, D., Pavlopoulos, J., Papageorgiou, H.,
    Androutsopoulos, I., & Manandhar, S. (2014). SemEval-2014 Task 4: Aspect
    based sentiment analysis. *Proceedings of SemEval 2014*, 27-35.

Each ``sentence_id`` is traceable to ``outputs/absa/evaluation/predictions.csv``,
whose SHA-256 digest is recorded in the frozen ``results.json``.

Scenarios describe a property of the sentence, never an expected model outcome.
A label such as "all models miss this" would silently become false the moment
artifacts are retrained, whereas "contrastive clause" stays true regardless. The
gold polarity is surfaced in the interface so a reader judges the models
directly instead of taking a claim on trust.
"""

import random
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class AspectSample:
    """One official test-split sentence with its gold aspect polarities."""

    review: str
    aspects: str
    scenario: str
    gold: Mapping[str, str]
    sentence_id: str


SAMPLES: tuple[AspectSample, ...] = (
    AspectSample(
        review="Great food but the service was dreadful!",
        aspects="food, service",
        scenario="Opposite polarities in one short sentence",
        gold=MappingProxyType({"food": "positive", "service": "negative"}),
        sentence_id="11351513#832512#0",
    ),
    AspectSample(
        review="The falafal was rather over cooked and dried but the chicken was fine.",
        aspects="falafal, chicken",
        scenario="Contrastive clause introduced by 'but'",
        gold=MappingProxyType({"falafal": "negative", "chicken": "positive"}),
        sentence_id="32935729#785247#5",
    ),
    AspectSample(
        review=(
            "i went in one day asking for a table for a group and was greeted "
            "by a very rude hostess."
        ),
        aspects="hostess, table",
        scenario="Complaint aimed at one aspect while another is only mentioned",
        gold=MappingProxyType({"hostess": "negative", "table": "neutral"}),
        sentence_id="33060905#1138585#0",
    ),
    AspectSample(
        review="Even when the chef is not in the house, the food and service are right on target.",
        aspects="chef, food, service",
        scenario="Aspect named without an opinion attached to it",
        gold=MappingProxyType({"chef": "neutral", "food": "positive", "service": "positive"}),
        sentence_id="11359727#487554#2",
    ),
    AspectSample(
        review="The waitress came by to pick up the soy sauce WHILE we were eating our lunch!!!!!",
        aspects="waitress, soy sauce, lunch",
        scenario="Negative tone carried without any negative word",
        gold=MappingProxyType(
            {"waitress": "negative", "soy sauce": "neutral", "lunch": "neutral"}
        ),
        sentence_id="32894669#1075584#2",
    ),
    AspectSample(
        review=(
            "While there's a decent menu, it shouldn't take ten minutes to get "
            "your drinks and 45 for a dessert pizza."
        ),
        aspects="menu, drinks, dessert pizza",
        scenario="Complaint about timing and price stated indirectly",
        gold=MappingProxyType(
            {"menu": "positive", "drinks": "neutral", "dessert pizza": "neutral"}
        ),
        sentence_id="33085939#758010#0",
    ),
)


def get_random_sample(current_review: str = "") -> AspectSample:
    """Return a sample different from the visible review where possible."""
    candidates = [sample for sample in SAMPLES if sample.review != current_review.strip()]
    return random.choice(candidates or list(SAMPLES))


def find_sample(review: str) -> AspectSample | None:
    """Return the sample matching *review* verbatim, or None once it is edited.

    Gold polarity belongs to the exact annotated sentence, so an edited review
    must lose the label rather than display one that no longer applies.
    """
    target = review.strip()
    for sample in SAMPLES:
        if sample.review == target:
            return sample
    return None
