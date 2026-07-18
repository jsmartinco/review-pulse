"""Smoke tests that keep the v3 namespace independent of legacy concerns."""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = str(Path(__file__).parents[2])


def test_absa_import_does_not_pull_legacy_packages() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.absa; "
            "forbidden = ('src.inference', 'src.training', 'src.evaluation', "
            "'src.models', 'src.tokenization', 'src.data'); "
            "bad = [module for module in sys.modules "
            "if any(module == prefix or module.startswith(prefix + '.') for prefix in forbidden)]; "
            "assert not bad, bad",
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr or result.stdout
