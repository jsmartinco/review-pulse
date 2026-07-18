"""SemEval acquisition, parsing, auditing and grouped split helpers."""

from .parser import parse_aspect_examples
from .schema import AspectExample

__all__ = ["AspectExample", "parse_aspect_examples"]
