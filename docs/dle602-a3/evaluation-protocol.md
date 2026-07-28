# ReviewPulse v3 canonical and supplemental evaluation protocol

This protocol produces the evidence required for DLE602 A3 RQ1 and RQ2. The submitted four-model experiment remains canonical. A separate six-model mode adds the exploratory GRU and TextCNN without rewriting the A2 design. Both modes evaluate the same retained official SemEval Restaurants examples; ISY503 metrics and checkpoints are not accepted.

## 1. Regenerate verified artifacts

From the repository root and recorded A3 environment:

```bash
.venv/bin/python -m src.absa.training.runner --device auto
```

The command trains TF-IDF, target-agnostic LSTM, ATAE-LSTM and DistilBERT with seed 42. Neural models use development macro-F1 checkpoint selection and restore the selected state before official-test evaluation. The run records training duration, configuration, history, best epoch and the overfitting/multi-seed diagnostic described in `training-protocol.md`.

For staged compute checks, a subset can be selected without changing the final command contract:

```bash
.venv/bin/python -m src.absa.training.runner --models tfidf target_lstm atae_lstm
.venv/bin/python -m src.absa.training.runner --models distilbert --device mps
```

Running a subset updates only those artifacts. The final evaluation must use four artifacts generated from the same commit, dataset checksums and seed policy.

After the CNN configuration gate, one explicit wrapper can regenerate all six artifacts from one frozen commit:

```bash
.venv/bin/python scripts/select_text_cnn_config.py
.venv/bin/python scripts/train_selected_text_cnn.py --all-six --device auto
```

The wrapper verifies that the selected CNN configuration, Git commit and Restaurants checksums agree before forwarding the exact winning widths/count and the six-model order to the common training runner.

## 2. Generate the comparison evidence

```bash
# Canonical A2 experiment; writes outputs/absa/evaluation/.
.venv/bin/python -m src.absa.evaluation.runner --device auto

# Exploratory supplement; writes outputs/absa/evaluation-six-model/.
.venv/bin/python -m src.absa.evaluation.runner \
  --models tfidf target_lstm target_gru text_cnn atae_lstm distilbert \
  --device auto
```

The default verification gate rejects pre-#91 LSTM checkpoints without configuration/training metadata and DistilBERT directories without `training_run.json`. `--allow-unverified-artifacts` exists only to diagnose legacy local files; its results must not be cited in the A3 report.

The evaluation command writes below `outputs/absa/evaluation/`:

| File | Purpose |
|---|---|
| `results.json` | Dataset/environment manifest, full and mixed-subset metrics, per-class results, matrices and efficiency evidence |
| `predictions.csv` | One shared official-test record per aspect with gold and every selected model prediction |
| `comparison.md` | Report-ready metrics and efficiency table; supplemental rows are explicitly labelled exploratory |
| `confusion_matrices.png` | Four or six caption-ready official-test confusion matrices |
| `error_analysis.json` | Counts and ordered candidates for conditioned wins, review-only wins, disagreements and common errors |

`predictions.csv` is the single source for full-test metrics, mixed-polarity metrics and error analysis. The runner records its SHA-256 digest in `results.json` so tables and examples can be traced to the same prediction set.

## Metric and efficiency definitions

- **Full test:** every retained positive, neutral or negative aspect instance in the official Restaurants test XML.
- **Mixed-polarity multi-aspect subset:** aspect instances from sentences containing at least two retained gold aspects with different polarities.
- **Cold-start prediction:** artifact loading plus the first single-example prediction.
- **Warm latency:** elapsed full-test batch prediction after one warm-up example, divided by the number of examples.
- **Artifact size:** recursive on-disk bytes for the loaded artifact.
- **Parameter count:** fitted logistic coefficients/intercepts for TF-IDF; all trainable and frozen parameters stored by each neural model.
- **Training time:** measured inside each model's fit/epoch loop and persisted when the artifact is generated.

Timing claims must state the recorded device and environment. Results from different devices or commits remain visible but must not be presented as a controlled efficiency comparison without qualification.

## Error review

The generated examples preserve official-test order rather than selecting only favourable outputs. Before report writing, the group must review representative cases from both directions:

- aspect-conditioned model correct while all selected review-only models are wrong on the mixed subset;
- review-only model correct while both aspect-conditioned models are wrong;
- all selected models wrong;
- models disagree despite receiving the same gold example.

Unexpected results are retained. Attention or attribution evidence belongs to issue #85 and must not be inferred from this quantitative runner.

The six-model output is supplemental evidence for #88. It does not replace the canonical four-model table or retroactively change the submitted research questions.
