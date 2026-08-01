# ReviewPulse v3.0.0 release verification (#89)

Evidence recorded while preparing the release branch. Every measurement below was
produced on the branch `release/v3.0.0`; the final tag must be created only after
the A3 report and contribution evidence are complete, so the source commit and
archive digest in `docs/submission-checklist.md` stay blank until then.

Environment: macOS 26.5, Apple Silicon.

## 1. Git LFS

Six artifacts are tracked through LFS by `.gitattributes`:

| Artifact | Size |
|---|---:|
| `outputs/absa/distilbert/model.safetensors` | 268 MB |
| `outputs/absa/atae_lstm.pt` | 2.8 MB |
| `outputs/absa/target_lstm.pt` | 2.4 MB |
| `outputs/absa/target_gru.pt` | 2.1 MB |
| `outputs/absa/text_cnn.pt` | 1.9 MB |
| `outputs/absa/tfidf_baseline.joblib` | 812 KB |

All six materialise after `git lfs pull` in a fresh clone; none remained an
unresolved pointer. The package builder independently rejects pointer files, so a
clone without `git lfs pull` fails the build rather than shipping stubs.

## 2. Clean-room installation

A fresh clone of `release/v3.0.0` with a new virtual environment installed from
`requirements.txt -c constraints-a3.txt` without manual intervention. Resolved
versions match the recorded baseline:

| Package | Resolved |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.13.0 |
| scikit-learn | 1.8.0 |
| Transformers | 5.14.1 |
| Streamlit | 1.59.2 |
| pandas | 3.0.3 |

## 3. Test suite: the clean-room count differs, by design

| Environment | Passed | Skipped |
|---|---:|---:|
| Development machine | 357 | 8 |
| Clean room | 351 | 14 |

The six-test gap is **expected and must not be treated as a regression**. The
sample-provenance tests in `tests/absa/test_samples.py` check each demo sample
against `outputs/absa/evaluation/predictions.csv`, which is intentionally not
tracked: it carries the review text and gold polarity of 1,120 annotated
instances and publishing it would redistribute a substantial part of the licensed
corpus.

Those tests therefore **skip silently in any clean clone**, and a green suite
there is not evidence that the samples still match the dataset. The check must be
run where the frozen evaluation outputs exist. Confirmed on the development
machine for this release: all six executed and passed rather than skipping.

The eight development-machine skips are the legacy Amazon `.review` files, which
are also not redistributed.

## 4. Offline behaviour

`scripts/smoke_absa.py` passes in the clean room with **no SemEval data present**:
all four core models clean-load from the LFS artifacts and return one prediction
per aspect. The application is therefore usable by a reader who never obtains the
dataset.

Commands that genuinely require the corpus — `src.absa.data.audit` and
`src.absa.evaluation.runner` — previously exited with a bare `FileNotFoundError`
traceback naming an absolute path. Both are documented first-run commands, so a
reader following the README reached a raw crash before learning the data must be
acquired separately. `parse_aspect_examples` now raises a `FileNotFoundError`
carrying the official source URL, the `prepare_semeval_restaurants.py` invocation
and a pointer to `semeval-restaurants.md`.

## 5. Package size

Built with `scripts/build_a3_package.py`. Two consecutive builds of the same mode
produced an identical archive SHA-256, confirming the deterministic contract.

| Mode | Size | Entries | Contents |
|---|---:|---:|---|
| `none` | 2.5 MB | 176 | Source, tests and documentation only |
| `lightweight` | 52 MB | 191 | Adds legacy artifacts and the five small v3 models |
| `all` | 288 MB | 197 | Adds the 268 MB DistilBERT directory |

**288 MB is the decision point.** It exceeds the upload limit of many learning
management systems, and the A2 risk register already recorded the contingency:
ship lightweight artifacts and document reproducible DistilBERT retrieval. The
LMS limit must be confirmed before choosing `all`; if `lightweight` is submitted,
the report must state that the DistilBERT path shows the controlled
missing-artifact state until the checkpoint is installed separately.

## 6. Content scan

No tracked file matches SemEval XML, `.review` data, virtual environments,
byte-code caches, editor state, `.env` files, private keys or common credential
patterns. The generated `sha256.json` data manifest is untracked as intended.

The built archive contains no `.xml`, no `predictions.csv`, no `results.json`, no
`error_analysis.json`, no `__pycache__` and nothing under `data/semeval2014/`
beyond `.gitkeep`.

Structural filtering cannot detect a credential embedded inside an otherwise
approved source file, so the manual secret review in
`docs/submission-checklist.md` remains required before sign-off.

## 7. Still outstanding before the tag

- Final A3 report PDF, and its commit recorded in the checklist.
- Contribution evidence from all group members.
- Confirmed LMS upload limit, which selects the artifact mode.
- Final archive built from the post-merge commit, with its SHA-256 recorded.
- `v3.0.0` tag, created only after the items above.
