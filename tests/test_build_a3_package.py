import json
import zipfile
from pathlib import Path

import pytest

from scripts.build_a3_package import (
    PACKAGE_ROOT,
    PackageEntry,
    build_package,
    validate_entries,
)


def test_validate_entries_rejects_restricted_xml(tmp_path: Path) -> None:
    source = tmp_path / "restaurants.xml"
    source.write_text("<sentences />", encoding="utf-8")

    with pytest.raises(ValueError, match="Prohibited package path"):
        validate_entries(
            [
                PackageEntry(
                    source,
                    PACKAGE_ROOT / "data" / "semeval2014" / source.name,
                )
            ]
        )


def test_validate_entries_rejects_duplicate_paths(tmp_path: Path) -> None:
    source = tmp_path / "README.md"
    source.write_text("example", encoding="utf-8")
    entry = PackageEntry(source, PACKAGE_ROOT / "README.md")

    with pytest.raises(ValueError, match="Duplicate package path"):
        validate_entries([entry, entry])


def test_source_package_is_deterministic_and_allowlisted(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_manifest = build_package(repo, first, "none", require_clean=False)
    second_manifest = build_package(repo, second, "none", require_clean=False)

    assert first.read_bytes() == second.read_bytes()
    assert first_manifest["archive"]["sha256"] == second_manifest["archive"]["sha256"]

    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
        manifest_name = str(PACKAGE_ROOT / "PACKAGE_MANIFEST.json")
        assert manifest_name in names
        assert str(PACKAGE_ROOT / "README.md") in names
        assert not any(name.endswith((".xml", ".review", ".env")) for name in names)
        assert not any("/outputs/absa/" in name for name in names)
        manifest = json.loads(archive.read(manifest_name))

    assert manifest["artifact_mode"] == "none"
    assert manifest["source_commit"] == first_manifest["source_commit"]
