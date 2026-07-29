import json
import zipfile
from pathlib import Path

import pytest

from scripts.build_a3_package import (
    PACKAGE_ROOT,
    PackageEntry,
    _is_lfs_pointer,
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
        assert str(PACKAGE_ROOT / "docs" / "architecture.md") in names
        assert str(PACKAGE_ROOT / "docs" / "assessment-files" / "presentation-outline.md") not in names
        assert str(PACKAGE_ROOT / "notebooks" / "EDA.ipynb") not in names
        assert not any(name.endswith((".xml", ".review", ".env")) for name in names)
        assert not any("/outputs/absa/" in name for name in names)
        manifest = json.loads(archive.read(manifest_name))

    assert manifest["artifact_mode"] == "none"
    assert manifest["source_commit"] == first_manifest["source_commit"]


def test_output_collision_preserves_source() -> None:
    repo = Path(__file__).resolve().parents[1]
    readme = repo / "README.md"
    original = readme.read_bytes()

    with pytest.raises(ValueError, match="must not overwrite"):
        build_package(repo, readme, "none", require_clean=False)

    assert readme.read_bytes() == original


def test_lfs_pointer_detection(tmp_path: Path) -> None:
    pointer = tmp_path / "model.pt"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0123456789abcdef\n"
        "size 123\n",
        encoding="utf-8",
    )
    materialised = tmp_path / "materialised.pt"
    materialised.write_bytes(b"model weights")

    assert _is_lfs_pointer(pointer)
    assert not _is_lfs_pointer(materialised)
