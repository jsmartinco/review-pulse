# ReviewPulse v3 smoke checks

The SemEval Restaurants XML files remain local because their licence/provenance is handled separately. Verified v3 inference artifacts are versioned through Git LFS; run `git lfs pull` after a manual clone when they are not fetched automatically. Then run from the repository root:

```bash
.venv/bin/python -m pytest tests/absa -q
.venv/bin/python scripts/smoke_absa.py
```

The first command verifies data parsing, grouped split, labels, model boundaries and inference contracts. The second clean-loads each available local v3 artifact and predicts `food` and `service` in one mixed-polarity review. It requires:

- `outputs/absa/tfidf_baseline.joblib`
- `outputs/absa/target_lstm.pt`
- `outputs/absa/atae_lstm.pt`
- `outputs/absa/distilbert/`

Missing artifacts are reported as controlled application errors; they are not silently replaced by legacy v2 artifacts.
