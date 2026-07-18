"""Contracts for the isolated ABSA configuration root."""

from pathlib import Path

from src.absa.config import ABSA_DATA_DIR, ABSA_METRICS_DIR, ABSA_OUTPUTS_DIR, ABSA_PLOTS_DIR, PROJECT_ROOT


def test_absa_paths_are_path_instances() -> None:
    for path in (PROJECT_ROOT, ABSA_DATA_DIR, ABSA_OUTPUTS_DIR, ABSA_METRICS_DIR, ABSA_PLOTS_DIR):
        assert isinstance(path, Path)


def test_absa_paths_are_isolated_from_legacy_outputs() -> None:
    assert ABSA_DATA_DIR == PROJECT_ROOT / "data" / "semeval2014"
    assert ABSA_OUTPUTS_DIR == PROJECT_ROOT / "outputs" / "absa"
    assert ABSA_METRICS_DIR.parent == ABSA_OUTPUTS_DIR
    assert ABSA_PLOTS_DIR.parent == ABSA_OUTPUTS_DIR
