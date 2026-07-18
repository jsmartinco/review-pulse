#!/usr/bin/env python3
"""Prepare local SemEval-2014 Restaurants XML files without redistributing them.

The official Task 4 page provides the corpus under a fair-use/third-party
terms notice. This helper copies user-obtained XML files into the canonical
local layout, validates the XML root, and creates a SHA-256 manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as etree
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = PROJECT_ROOT / "data" / "semeval2014" / "restaurants"
MANIFEST_NAME = "sha256.json"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_semeval_xml(path: Path) -> None:
    """Reject unreadable XML or a file that is not a SemEval sentence corpus."""
    try:
        root = etree.parse(path).getroot()
    except etree.ParseError as exc:
        raise ValueError(f"Invalid XML: {path}") from exc
    if root.tag != "sentences":
        raise ValueError(f"Expected a <sentences> root in {path}, found <{root.tag}>")
    if root.find("sentence") is None:
        raise ValueError(f"Expected at least one <sentence> in {path}")


def prepare_file(source: Path, destination: Path, *, force: bool) -> dict[str, object]:
    """Validate and copy one source XML, returning manifest metadata."""
    if not source.is_file():
        raise ValueError(f"Source XML does not exist: {source}")
    validate_semeval_xml(source)
    if destination.exists() and not force:
        raise ValueError(f"Destination exists: {destination}. Pass --force to replace it.")
    shutil.copyfile(source, destination)
    return {
        "filename": destination.name,
        "sha256": sha256(destination),
        "bytes": destination.stat().st_size,
        "source_filename": source.name,
    }


def verify(destination: Path) -> None:
    """Verify prepared XMLs against their recorded SHA-256 manifest."""
    manifest_path = destination / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"Manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split in ("train", "test"):
        entry = manifest["files"][split]
        path = destination / entry["filename"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch for {split}: {path}")
    print(f"Verified {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, help="Official annotated Restaurants training XML.")
    parser.add_argument("--test", type=Path, help="Official annotated Restaurants test/gold XML.")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--force", action="store_true", help="Replace prepared XMLs if they exist.")
    parser.add_argument("--verify", action="store_true", help="Verify an existing SHA-256 manifest only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.verify:
            verify(args.destination)
            return 0
        if args.train is None or args.test is None:
            raise ValueError("--train and --test are required unless --verify is used.")

        args.destination.mkdir(parents=True, exist_ok=True)
        files = {
            "train": prepare_file(args.train, args.destination / "restaurants_train.xml", force=args.force),
            "test": prepare_file(args.test, args.destination / "restaurants_test.xml", force=args.force),
        }
        manifest = {
            "dataset": "SemEval-2014 Task 4 Restaurants",
            "source_page": "https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools",
            "files": files,
        }
        manifest_path = args.destination / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Prepared {args.destination}")
        print(f"Wrote {manifest_path}")
        return 0
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
