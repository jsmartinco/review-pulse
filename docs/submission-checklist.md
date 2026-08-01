# DLE602 A3 Submission Checklist — ReviewPulse v3.0.0

Use this checklist against one frozen source commit. Do not create the final tag or upload the ZIP until every required item is evidenced.

## Release identity

- [ ] Exact post-merge #89 source commit recorded: `________________`
- [ ] Academic report commit recorded: `________________`
- [ ] Submission ZIP SHA-256 recorded: `________________`
- [ ] ZIP size recorded: `________________`
- [ ] LMS upload limit confirmed: `________________`
- [ ] Artifact mode chosen against that limit. Measured on `release/v3.0.0`: `none` 2.5 MB, `lightweight` 52 MB, `all` 288 MB
- [ ] `v3.0.0` tag points to the verified source commit
- [ ] GitHub release notes and submitted package describe the same contents

The implementation baseline before #89 is merge commit `0f02be3` (PR #100). The final archive must be built only after #89 is merged and must identify that exact post-merge commit.

## Report and group record

- [ ] Final report is 1,350–1,650 words under its declared counting rule
- [ ] Canonical four-model results remain separate from exploratory GRU/TextCNN results
- [ ] Tables, figures and token-evidence examples trace to frozen outputs
- [ ] Attention and attribution are described as indicative, not causal
- [ ] Contribution record and dated hand-offs are confirmed by all members
- [ ] Academic Integrity Declaration and Statement of Acknowledgement are complete
- [ ] Final PDF is copied into the package

Group members:

- Luis Faria — A00187785
- Victor Dorantes — A00179705
- Juan Martinez — A00167145

## Source and licensing

- [ ] Package includes the required Python source, tests, README and DLE602 documentation
- [ ] No `.env`, credentials, tokens, private keys, editor state, caches or temporary files
- [ ] No `.git/`, `.venv/`, `__pycache__/`, `.pytest_cache/` or Hugging Face cache
- [ ] No restricted SemEval XML or derived row-level dataset is redistributed
- [ ] SemEval acquisition, placement and checksum instructions are included
- [ ] Third-party dependencies and cited model/data sources are documented
- [ ] Git status is clean before the package is built

## Environment and installation

Run in a new environment using the reviewed A3 constraints:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -c constraints-a3.txt
```

- [ ] Python and platform versions recorded
- [ ] Installation succeeds without undocumented manual changes
- [ ] Resolved critical dependency versions match `constraints-a3.txt`
- [ ] CPU-only import and application startup succeed. Verified on `release/v3.0.0`: every neural predictor loaded wholly on CPU in the clean room **even though MPS was available on that host**, because all four torch adapters pin `map_location="cpu"`

## Automated verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke_absa.py
.venv/bin/python scripts/export_absa_evidence.py
```

- [ ] Full suite passes; counts and expected skips are recorded
- [ ] Sample-provenance tests **executed rather than skipped**: run where `outputs/absa/evaluation/predictions.csv` exists and confirm the six `test_sample_matches_the_official_test_split` cases are not in the skip list. They skip silently in a clean clone, so a green suite there does not evidence this check (see `dle602-a3/release-verification.md`)
- [ ] Legacy ISY503 regression path remains functional
- [ ] All available v3 artifacts clean-load
- [ ] `food` and `service` smoke predictions return one result per aspect
- [ ] TF-IDF/LSTM/GRU/TextCNN evidence state is explicitly unsupported
- [ ] ATAE-LSTM attention aligns to visible review offsets
- [ ] DistilBERT attribution aligns to visible review offsets

## Data audit and evaluation evidence

With legitimately acquired Restaurants XML files:

```bash
.venv/bin/python -m src.absa.data.audit
.venv/bin/python -m src.absa.evaluation.runner --device cpu
```

- [ ] Audit reproduces the documented label and offset counts
- [ ] Grouped split overlap assertions pass
- [ ] Official retained test count is 1,120
- [ ] Mixed-polarity subset is 228 instances across 80 sentences
- [ ] Canonical evaluation output and prediction digest are preserved
- [ ] CPU evaluation completes or any documented hardware limitation is reproduced honestly

The supplemental six-model command is:

```bash
.venv/bin/python -m src.absa.evaluation.runner \
  --models tfidf target_lstm target_gru text_cnn atae_lstm distilbert \
  --device cpu
```

- [ ] Supplemental output remains separate from `outputs/absa/evaluation/`
- [ ] GRU and TextCNN remain labelled exploratory

## Streamlit acceptance

```bash
.venv/bin/streamlit run app.py
```

- [ ] Landing page clearly separates ISY503 v2.3.0 and DLE602 v3.0.0
- [ ] Intro page does not duplicate the sidebar logo
- [ ] Sidebar logo, menu order and favicon render correctly
- [ ] Sample generator fills a mixed-polarity review and aspects
- [ ] Manual comma-separated aspects preserve input order
- [ ] Each of the six v3 models can be selected when its artifact exists
- [ ] Supported token evidence renders safely with its limitation
- [ ] Missing artifacts and invalid input show controlled errors without silent fallback
- [ ] No stack trace or debug output appears in the user workflow

Capture at least:

- [ ] v3 input/result view
- [ ] ATAE-LSTM heatmap
- [ ] DistilBERT attribution view
- [ ] one controlled missing-artifact or validation message

## Artifact strategy

Record every included artifact:

| Artifact | Included? | Bytes | SHA-256 | Runtime/network dependency |
|---|:---:|---:|---|---|
| TF-IDF | [ ] | | | |
| Target LSTM | [ ] | | | |
| Target GRU | [ ] | | | |
| TextCNN | [ ] | | | |
| ATAE-LSTM | [ ] | | | |
| DistilBERT | [ ] | | | |

- [ ] Included artifacts load offline, or each external retrieval is explicit and checksum-verified
- [ ] Artifact-bearing modes include the four legacy v2 files required by the preserved ISY503 page
- [ ] The lightweight CPU strategy includes at least the verified small-model path
- [ ] DistilBERT packaging decision is consistent with the confirmed LMS limit
- [ ] No package claims offline support if a Hugging Face download is still required

## Package inspection

- [ ] Archive is built with `scripts/build_a3_package.py` using the selected artifact mode
- [ ] Archive is built from a documented allowlist, not the entire working directory
- [ ] Archive expands into one clearly named root folder
- [ ] README quick-start is visible at the package root
- [ ] No broken symlinks or absolute local paths
- [ ] Largest files and total uncompressed/compressed sizes are reviewed
- [ ] Secret scan returns no findings
- [ ] Cache/temporary-file scan returns no findings
- [ ] Restricted-data scan returns no findings
- [ ] ZIP is extracted into a clean directory and the documented verification path is rerun

## Final sign-off

| Gate | Owner | Status | Evidence |
|---|---|:---:|---|
| Report and references | Group | [ ] | |
| Contribution record | Group | [ ] | |
| Clean installation | | [ ] | |
| Tests and CPU smoke | | [ ] | |
| Artifact checksums/sizes | | [ ] | |
| Streamlit acceptance | | [ ] | |
| Package content/security scan | | [ ] | |
| ZIP extraction retest | | [ ] | |
| Final tag and GitHub release | | [ ] | |

Final sequence:

1. Freeze the accepted report and source commits.
2. Merge #89.
3. Build and inspect the deterministic ZIP from the exact post-merge commit.
4. Extract and retest the ZIP.
5. Record sizes and SHA-256 digests.
6. Obtain group sign-off.
7. Create and publish `v3.0.0` from the verified release commit.
