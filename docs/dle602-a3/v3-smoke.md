# ReviewPulse v3 smoke checks

The SemEval Restaurants XML files and v3 artifacts are intentionally local: the dataset licence/provenance and the artifact policy are recorded separately. After preparing those inputs, run from the repository root:

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
