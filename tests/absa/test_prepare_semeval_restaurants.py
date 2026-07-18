"""Executable contract for local SemEval Restaurants preparation."""

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "prepare_semeval_restaurants.py"


def _write_xml(path: Path, sentence_id: str) -> None:
    path.write_text(
        f'<sentences><sentence id="{sentence_id}"><text>Great food.</text></sentence></sentences>',
        encoding="utf-8",
    )


def test_prepare_and_verify_local_xmls(tmp_path: Path) -> None:
    train = tmp_path / "official_train.xml"
    test = tmp_path / "official_test.xml"
    destination = tmp_path / "prepared"
    _write_xml(train, "1")
    _write_xml(test, "2")

    prepared = subprocess.run(
        [sys.executable, str(SCRIPT), "--train", str(train), "--test", str(test), "--destination", str(destination)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert prepared.returncode == 0, prepared.stderr

    manifest = json.loads((destination / "sha256.json").read_text(encoding="utf-8"))
    assert manifest["files"]["train"]["filename"] == "restaurants_train.xml"
    assert manifest["files"]["test"]["filename"] == "restaurants_test.xml"

    verified = subprocess.run(
        [sys.executable, str(SCRIPT), "--destination", str(destination), "--verify"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert verified.returncode == 0, verified.stderr
