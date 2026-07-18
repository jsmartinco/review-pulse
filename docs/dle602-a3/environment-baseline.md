# ReviewPulse v3.0 environment and v2.3.0 regression baseline

Issue: #73  
Branch: `feat/reviewpulse-v3-absa`  
Baseline commit: `3880b2e` (`main` before v3 changes)  
Recorded: 18 July 2026

## Recreate the environment

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install scikit-learn==1.8.0
```

The resolved package set is committed as `constraints-a3.txt`. It is the reference environment for v3 experiments and clean-install checks.

## Environment record

- Python: `3.12.10` (Apple Silicon macOS)
- PyTorch: `2.13.0`
- scikit-learn: `1.8.0`
- Transformers: `5.14.1`
- Streamlit: `1.59.2`
- pandas: `3.0.3`
- pytest: `9.1.1`

`scikit-learn==1.8.0` is intentional: the committed TF-IDF artifact was serialised with that version. Installing the open-ended requirement selected 1.9.0 and emitted an artifact-version warning.

## Legacy regression baseline

Command:

```bash
.venv/bin/python -m pytest -q
```

Result after the v3 integration checks: **217 passed, 8 skipped**.

The skipped cases require the local Amazon `.review` files for Books, DVD, Electronics and Kitchen & Housewares. Those directories are intentionally gitignored and are absent in a clean clone. They are data-availability skips, not v3 regressions:

- `test_load_all_domains_has_required_columns`
- `test_load_all_domains_four_domains`
- `test_load_all_domains_binary_labels`
- `test_load_all_domains_no_empty_text`
- `test_load_all_domains_count`

All paths that do not require the uncommitted Amazon dataset pass. When those legacy data files are placed under `data/`, the corresponding checks run instead of skipping.

## Guardrails

- Do not change the legacy v2.3.0 code or artifacts to make this baseline pass.
- Keep all v3 data, models and generated outputs isolated from legacy paths.
- Re-run this command after dependency changes and before the v3 release.
