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

## Verified #94 candidate run

The clean-load candidate was trained on CPU with seed 42 from commit `bdea54d`, using the same Restaurants checksums, official 1,120-example test set and 228-example mixed-polarity subset as #84. Best development macro-F1 occurred at epoch 8. No material overfitting was detected and the recorded diagnostic does not recommend a multi-seed contingency.

| Measure | Target LSTM #84 | Target GRU #94 |
|---|---:|---:|
| Full-test accuracy | 0.6688 | **0.6750** |
| Full-test macro-F1 | 0.4326 | **0.4603** |
| Mixed accuracy | **0.4167** | 0.4079 |
| Mixed macro-F1 | **0.3264** | 0.3156 |
| CPU training time | 9.32 s | **7.66 s** |
| Parameters | 571,291 | **512,411** |
| Artifact size | 2.25 MB | **2.02 MB** |

At matched dimensions, the GRU used 10.31% fewer parameters, produced a 9.99% smaller artifact and trained 17.83% faster in the recorded runs. Its overall metrics were slightly stronger, but its mixed-polarity metrics were slightly weaker. This is a useful negative boundary rather than a contradiction: changing the recurrent cell does not solve the missing-aspect limitation shared by both review-only models.

The LSTM and GRU candidate artifacts were generated from different reviewed source commits because the GRU did not exist in the #84 commit. The final supplemental comparison in #96 must regenerate all six artifacts from one frozen commit before using cross-model timing as a final report claim.

## Reporting guardrails

- Label GRU results exploratory.
- Preserve a negative or negligible result.
- Compare GRU primarily against the target-agnostic LSTM.
- Do not interpret repeated per-aspect outputs as aspect awareness.
- Do not display attention or attribution for this review-only model.
- Do not alter the canonical four-model evidence when adding the optional artifact.
