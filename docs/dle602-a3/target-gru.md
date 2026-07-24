# Optional Target-Agnostic GRU Protocol

## Experimental role

The target-agnostic GRU is the fifth-model candidate mapped in issue #94. It is an exploratory review-only recurrent ablation, not a replacement for the submitted four-model A2 experiment.

Like the target-agnostic LSTM, the GRU receives only the review. `predict_aspects()` repeats the same review-level prediction for each supplied aspect, making its expected limitation on mixed-polarity multi-aspect sentences explicit. The model reports token evidence as unsupported.

The canonical four-model training command, evaluation outputs and Streamlit options remain unchanged until the supplemental integration in #96.

## Controlled configuration

The GRU matches the target-agnostic LSTM wherever practical:

| Control | LSTM | GRU |
|---|---:|---:|
| Input | Review only | Review only |
| Vocabulary | Training reviews | Training reviews |
| Maximum length | 80 | 80 |
| Embedding dimension | 100 | 100 |
| Hidden dimension per direction | 128 | 128 |
| Directions | 2 | 2 |
| Recurrent layers | 1 | 1 |
| Dropout | 0.5 | 0.5 |
| Output logits | 3 | 3 |
| Epoch budget | 8 | 8 |
| Batch size | 64 | 64 |
| Optimizer | Adam | Adam |
| Learning rate | 0.001 | 0.001 |
| Weight decay | 0.0001 | 0.0001 |
| Selection | Development macro-F1 | Development macro-F1 |
| Early-stopping patience | 2 | 2 |

The deliberate architectural difference is the recurrent cell. A GRU uses update/reset gating without a separate LSTM cell state, so it has fewer parameters at matched embedding and hidden dimensions. Parameter count, training time and artifact size are recorded rather than assumed to be better.

## Reproducibility and artifacts

The trainer uses the #91 controls:

- explicit Python, NumPy, PyTorch and DataLoader seed;
- sentence-grouped development split and untouched official test split;
- weight decay and dropout;
- development macro-F1 checkpoint selection;
- early stopping and restored best checkpoint;
- complete epoch history and overfitting diagnostic;
- source commit, dataset checksums and generated timestamp from the shared runner.

Train and clean-load the optional artifact independently:

```bash
.venv/bin/python -m src.absa.training.runner \
  --models target_gru \
  --device cpu

.venv/bin/python scripts/smoke_target_gru.py
```

Generated, gitignored artifacts:

```text
outputs/absa/target_gru.pt
outputs/absa/target_gru_metrics.json
```

`load_target_gru_evaluator()` exposes the same `LoadedEvaluator` contract used by the core models. Full-test and mixed-polarity metrics therefore use the existing label order, subset definition and metric functions; the supplemental multi-model files remain deferred to #96.

## Reporting guardrails

- Label GRU results exploratory.
- Preserve a negative or negligible result.
- Compare GRU primarily against the target-agnostic LSTM.
- Do not interpret repeated per-aspect outputs as aspect awareness.
- Do not display attention or attribution for this review-only model.
- Do not alter the canonical four-model evidence when adding the optional artifact.
