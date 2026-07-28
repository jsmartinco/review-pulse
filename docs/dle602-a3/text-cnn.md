# Optional Review-Only TextCNN Protocol

## Experimental role

The TextCNN is the sixth-model candidate mapped in issue #95. It is an exploratory review-only, non-recurrent baseline, not a replacement for the submitted four-model A2 experiment and not a reproduction of Zhao, Gui and Zhang's Twitter architecture.

Like TF-IDF, the target-agnostic LSTM and the optional GRU, the CNN receives only the review. `predict_aspects()` repeats the same review-level prediction for each supplied aspect, making its expected limitation on mixed-polarity multi-aspect sentences explicit. Convolution activations are not exposed as an explanation, and the model reports aspect-specific token evidence as unsupported.

The canonical four-model training command, evaluation outputs and Streamlit options remain unchanged until the supplemental integration in #96.

## Architecture contract

The candidate uses:

| Control | Value |
|---|---:|
| Input | Review only |
| Vocabulary | Training reviews |
| Maximum length | 80 |
| Embedding dimension | 100 |
| Candidate filter widths | `(2,3,4)` or `(3,4,5)` |
| Candidate filters per width | 64 or 100 |
| Activation | ReLU |
| Pooling | Global max |
| Short-input handling | Right-pad to widest filter |
| Dropout | 0.5 |
| Output logits | 3 |
| Epoch budget | 8 final; 4 configuration gate |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Selection | Development macro-F1 |
| Early-stopping patience | 2 |

The model embeds the review, applies parallel one-dimensional convolutions, globally max-pools each feature map, concatenates the pooled features and applies dropout before the three-class head. Inputs shorter than the widest filter are right-padded in the model; the training encoder already pads normal batches to the recorded maximum length.

## Bounded development-only selection

The configuration gate compares exactly three candidates:

1. widths `(2,3,4)`, 64 filters per width;
2. widths `(3,4,5)`, 64 filters per width;
3. widths `(3,4,5)`, 100 filters per width.

All candidates use seed 42, the same sentence-grouped train/development split, four epochs maximum and development macro-F1 selection. The search path sets `official_test_evaluated` to `false`; the untouched official test is evaluated only after the configuration is locked. Ties retain the earlier, lower-compute candidate.

Run the bounded gate, train the selected candidate and verify a clean load:

```bash
.venv/bin/python scripts/select_text_cnn_config.py

.venv/bin/python -m src.absa.training.runner \
  --models text_cnn \
  --cnn-filter-widths 3 4 5 \
  --cnn-num-filters 100 \
  --device cpu

.venv/bin/python scripts/smoke_text_cnn.py
```

Generated, gitignored evidence:

```text
outputs/absa/text_cnn_config_search.json
outputs/absa/text_cnn.pt
outputs/absa/text_cnn_metrics.json
```

`load_text_cnn_evaluator()` exposes the common `LoadedEvaluator` contract. Full-test and mixed-polarity metrics therefore use the existing label order, subset definition and metric functions; six-model shared outputs remain deferred to #96.

The widths/count shown in the training command are the defaults. Replace them with the `selected` values from `text_cnn_config_search.json` when the gate selects another candidate.

## Verified #95 candidate run

The configuration gate and final candidate were run on CPU with seed 42 from commit `4e66f7a`. All three gate candidates used the same Restaurants checksums and grouped development split, and `official_test_evaluated` remained `false` until the winning configuration was locked.

| Widths × filters | Development macro-F1 | Gate time | Parameters |
|---|---:|---:|---:|
| `(2,3,4) × 64` | 0.4309 | 8.74 s | 393,371 |
| `(3,4,5) × 64` | 0.3776 | 10.47 s | 412,571 |
| **`(3,4,5) × 100`** | **0.4722** | 13.60 s | 456,203 |

The selected configuration was then trained for the full eight-epoch budget. Best development macro-F1 occurred at epoch 6. No material within-run overfitting was detected, so the recorded diagnostic does not recommend the multi-seed contingency.

| Measure | Target LSTM #84 | Target GRU #94 | TextCNN #95 |
|---|---:|---:|---:|
| Full-test accuracy | 0.6688 | 0.6750 | **0.6893** |
| Full-test macro-F1 | 0.4326 | **0.4603** | 0.4498 |
| Mixed accuracy | **0.4167** | 0.4079 | **0.4167** |
| Mixed macro-F1 | **0.3264** | 0.3156 | 0.3106 |
| CPU training time | 9.32 s | **7.66 s** | 26.73 s |
| Parameters | 571,291 | 512,411 | **456,203** |
| Artifact size | 2.25 MB | 2.02 MB | **1.81 MB** |

The CNN produced the strongest candidate accuracy and the smallest review-only neural artifact, but it did not improve macro-F1 and trained more slowly in these separate CPU runs. Its mixed-polarity neutral recall was zero. This result is retained because it demonstrates that a non-recurrent sentence encoder still cannot resolve aspect-specific contradictions merely by changing architecture.

The LSTM, GRU and CNN values above were generated from different reviewed source commits. Issue #96 must regenerate all six artifacts from one frozen commit before presenting cross-model timing or efficiency as a final report claim.

## Reproducibility and artifacts

The trainer uses the #91 controls:

- explicit Python, NumPy, PyTorch and DataLoader seed;
- sentence-grouped development split and untouched official test split;
- weight decay and dropout;
- development macro-F1 checkpoint selection;
- early stopping and restored best checkpoint;
- complete epoch history and overfitting diagnostic;
- source commit, dataset checksums and generated timestamp from the shared runner;
- persisted filter widths, filter count, pooling and padding behaviour.

## Reporting guardrails

- Label CNN results exploratory.
- Preserve a negative or negligible result.
- Compare it with the other review-only baselines, especially TF-IDF, LSTM and GRU.
- Do not interpret repeated per-aspect outputs as aspect awareness.
- Do not present convolution activations as attention, attribution or causal reasoning.
- Do not alter the canonical four-model evidence when adding the optional artifact.
