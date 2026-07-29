# DLE602 A3 Submission Package

The package builder uses Git-tracked source as its allowlist, excludes tracked legacy model binaries, rejects restricted/raw-data paths and optionally adds verified v3 artifacts and the final report. Every entry is recorded with its byte size and SHA-256 digest in `PACKAGE_MANIFEST.json`.

## Build modes

Source and documentation only:

```bash
.venv/bin/python scripts/build_a3_package.py \
  --artifact-mode none \
  --report /path/to/DLE602_A3_Report.pdf
```

Source plus TF-IDF, LSTM, GRU, TextCNN and ATAE-LSTM:

```bash
.venv/bin/python scripts/build_a3_package.py \
  --artifact-mode lightweight \
  --report /path/to/DLE602_A3_Report.pdf
```

Source plus all six v3 artifacts, including DistilBERT:

```bash
.venv/bin/python scripts/build_a3_package.py \
  --artifact-mode all \
  --report /path/to/DLE602_A3_Report.pdf
```

The default output is `dist/ReviewPulse-v3.0.0-DLE602-A3.zip`. The builder refuses a dirty working tree. Entry timestamps are fixed to the source commit time and paths are sorted, so identical source, report and artifacts produce identical ZIP bytes.

## Artifact decision

Use `lightweight` when the submission limit cannot accommodate the approximately 256 MB DistilBERT directory. Use `all` only after confirming the LMS limit. If DistilBERT is omitted, document that its app path produces the controlled missing-artifact state unless a separately checksum-verified artifact is installed.

No mode includes SemEval XML, `.review` files, credentials, virtual environments, caches or generated row-level prediction exports. SemEval acquisition remains documented in `semeval-restaurants.md`.

## Verification

```bash
.venv/bin/python -m pytest tests/test_build_a3_package.py -q
unzip -l dist/ReviewPulse-v3.0.0-DLE602-A3.zip
shasum -a 256 dist/ReviewPulse-v3.0.0-DLE602-A3.zip
```

Extract the final candidate into a clean directory and follow `README.md` plus `docs/submission-checklist.md` before tagging `v3.0.0`.
