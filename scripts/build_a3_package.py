"""Build a deterministic, allowlisted DLE602 A3 submission archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


PACKAGE_ROOT = PurePosixPath("ReviewPulse-v3.0.0")
LIGHTWEIGHT_ARTIFACTS = (
    "outputs/absa/tfidf_baseline.joblib",
    "outputs/absa/tfidf_baseline_metrics.json",
    "outputs/absa/target_lstm.pt",
    "outputs/absa/target_lstm_metrics.json",
    "outputs/absa/target_gru.pt",
    "outputs/absa/target_gru_metrics.json",
    "outputs/absa/text_cnn.pt",
    "outputs/absa/text_cnn_metrics.json",
    "outputs/absa/text_cnn_config_search.json",
    "outputs/absa/atae_lstm.pt",
    "outputs/absa/atae_lstm_metrics.json",
)
DISTILBERT_ARTIFACTS = (
    "outputs/absa/distilbert/config.json",
    "outputs/absa/distilbert/model.safetensors",
    "outputs/absa/distilbert/tokenizer.json",
    "outputs/absa/distilbert/tokenizer_config.json",
    "outputs/absa/distilbert/training_run.json",
    "outputs/absa/distilbert_metrics.json",
)
PROHIBITED_SUFFIXES = (".review", ".xml", ".env", ".pyc", ".pyo")
PROHIBITED_PARTS = (
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
)


@dataclass(frozen=True)
class PackageEntry:
    source: Path
    archive_path: PurePosixPath


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_safe(relative_path: PurePosixPath) -> bool:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return False
    if any(part in PROHIBITED_PARTS for part in relative_path.parts):
        return False
    if relative_path.name == ".env":
        return False
    if relative_path.suffix.lower() in PROHIBITED_SUFFIXES:
        return False
    if relative_path.parts[:2] == ("data", "semeval2014") and relative_path.name != ".gitkeep":
        return False
    return True


def tracked_source_entries(repo: Path) -> list[PackageEntry]:
    entries: list[PackageEntry] = []
    result = subprocess.run(
        ("git", "ls-files", "-z"),
        cwd=repo,
        check=True,
        capture_output=True,
    )
    for raw_value in result.stdout.split(b"\0"):
        if not raw_value:
            continue
        value = raw_value.decode("utf-8")
        relative = PurePosixPath(value)
        if relative.parts[0] == "outputs" and relative.name != ".gitkeep":
            continue
        if not _is_safe(relative):
            continue
        source = repo / Path(relative)
        if source.is_symlink():
            raise ValueError(f"Symlinks are not allowed in the package: {relative}")
        if source.is_file():
            entries.append(PackageEntry(source, PACKAGE_ROOT / relative))
    return entries


def artifact_entries(repo: Path, mode: str) -> list[PackageEntry]:
    if mode == "none":
        return []
    paths = list(LIGHTWEIGHT_ARTIFACTS)
    if mode == "all":
        paths.extend(DISTILBERT_ARTIFACTS)
    entries: list[PackageEntry] = []
    missing: list[str] = []
    for value in paths:
        source = repo / value
        if not source.is_file():
            missing.append(value)
            continue
        if source.is_symlink():
            raise ValueError(f"Symlinks are not allowed in the package: {value}")
        entries.append(PackageEntry(source, PACKAGE_ROOT / PurePosixPath(value)))
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n- " + "\n- ".join(missing))
    return entries


def report_entry(report: Path | None) -> list[PackageEntry]:
    if report is None:
        return []
    resolved = report.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Report not found: {resolved}")
    if resolved.suffix.lower() not in {".pdf", ".md"}:
        raise ValueError("Report must be a PDF or Markdown file")
    return [PackageEntry(resolved, PACKAGE_ROOT / "report" / resolved.name)]


def validate_entries(entries: list[PackageEntry]) -> None:
    archive_paths: set[PurePosixPath] = set()
    for entry in entries:
        relative = entry.archive_path.relative_to(PACKAGE_ROOT)
        if not _is_safe(relative):
            raise ValueError(f"Prohibited package path: {relative}")
        if entry.archive_path in archive_paths:
            raise ValueError(f"Duplicate package path: {entry.archive_path}")
        archive_paths.add(entry.archive_path)


def build_package(
    repo: Path,
    output: Path,
    artifact_mode: str,
    report: Path | None = None,
    *,
    require_clean: bool = True,
) -> dict[str, object]:
    repo = repo.resolve()
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if require_clean and dirty:
        raise RuntimeError("Refusing to package a dirty working tree")
    commit = _git(repo, "rev-parse", "HEAD")
    commit_epoch = int(_git(repo, "show", "-s", "--format=%ct", commit))
    timestamp = datetime.fromtimestamp(commit_epoch, timezone.utc)
    zip_time = max(timestamp, datetime(1980, 1, 1, tzinfo=timezone.utc)).timetuple()[:6]

    entries = tracked_source_entries(repo)
    entries.extend(artifact_entries(repo, artifact_mode))
    entries.extend(report_entry(report))
    entries.sort(key=lambda item: str(item.archive_path))
    validate_entries(entries)

    manifest_entries = [
        {
            "path": str(entry.archive_path.relative_to(PACKAGE_ROOT)),
            "bytes": entry.source.stat().st_size,
            "sha256": _sha256(entry.source),
        }
        for entry in entries
    ]
    manifest: dict[str, object] = {
        "package": str(PACKAGE_ROOT),
        "source_commit": commit,
        "source_commit_time_utc": timestamp.isoformat(),
        "artifact_mode": artifact_mode,
        "entries": manifest_entries,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for entry in entries:
            info = zipfile.ZipInfo(str(entry.archive_path), zip_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, entry.source.read_bytes())
        info = zipfile.ZipInfo(str(PACKAGE_ROOT / "PACKAGE_MANIFEST.json"), zip_time)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, manifest_bytes)

    manifest["archive"] = {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/ReviewPulse-v3.0.0-DLE602-A3.zip"),
    )
    parser.add_argument(
        "--artifact-mode",
        choices=("none", "lightweight", "all"),
        default="none",
        help="Include no v3 artifacts, five lightweight artifacts, or all six models.",
    )
    parser.add_argument("--report", type=Path, help="Optional final A3 report PDF or Markdown source.")
    args = parser.parse_args()
    manifest = build_package(args.repo, args.output, args.artifact_mode, args.report)
    print(json.dumps(manifest["archive"], indent=2))


if __name__ == "__main__":
    main()
