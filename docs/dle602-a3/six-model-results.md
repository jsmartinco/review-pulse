# ReviewPulse v3 supplemental six-model results

## Status and provenance

This is the verified exploratory result for issue #96. It supplements, but does not replace, the four-model experiment submitted in A2.

- Artifact source commit: `cef08fa90e8e66b6bd016268ba4f5e2af379ae80`
- Evaluation source commit: `941148cc7b9f32271da0cd71b3b9a496811667e0`
- Evaluation generated at: `2026-07-28T20:29:40.349135+00:00`
- Training seed: `42`
- Restaurants train SHA-256: `223601da1bded6caa4ef9cf91a7007578141ca6d8ed50d5a5c217565f89d2fc5`
- Restaurants test SHA-256: `f21509cfa37e16534cd5b2da043be487355b64ef48fe8d6aaacaeca6b49cc0fb`
- Official retained test examples: `1,120`
- Mixed-polarity examples: `228` from `80` sentences
- Shared prediction SHA-256: `9d439207a8fdcafed5328d513ee4921bfc8b0dc4ecefcb3e7a9622f66f40e196`
- Environment: Python 3.12.10, PyTorch 2.13.0, scikit-learn 1.8.0 and Transformers 5.14.1 on macOS arm64

All six artifact records carry the same source commit, dataset checksums, seed and fixed label order (`negative`, `neutral`, `positive`). The TextCNN gate selected widths `(3,4,5)` with 100 filters per width using development macro-F1 without evaluating the official test.

## Shared comparison

GRU and TextCNN are exploratory review-only extensions. The other four rows remain the canonical A2 model ladder.

| Model | Scope | Test acc. | Test macro-F1 | Mixed acc. | Mixed macro-F1 | Train s | Cold ms | Warm ms/example | Throughput ex/s | Parameters | Artifact MB | Device |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TF-IDF | A2 core | 0.7018 | 0.4605 | 0.4430 | 0.3319 | 0.14 | 80.66 | 0.009 | 112,612.81 | 50,919 | 0.77 | CPU |
| LSTM | A2 core | 0.6687 | 0.4326 | 0.4167 | 0.3264 | 8.07 | 13.52 | 0.097 | 10,327.37 | 571,291 | 2.25 | CPU |
| GRU | Exploratory | 0.6750 | 0.4603 | 0.4079 | 0.3156 | 7.02 | 12.63 | 0.067 | 14,946.12 | 512,411 | 2.02 | CPU |
| TextCNN | Exploratory | 0.6893 | 0.4498 | 0.4167 | 0.3106 | 25.13 | 14.46 | 0.727 | 1,376.02 | 456,203 | 1.81 | CPU |
| ATAE-LSTM | A2 core | 0.6438 | 0.4799 | 0.4737 | 0.4491 | 10.84 | 11.71 | 0.121 | 8,289.67 | 674,348 | 2.64 | CPU |
| DistilBERT | A2 core | 0.8250 | 0.7199 | 0.6667 | 0.6473 | 122.08 | 675.87 | 3.288 | 304.18 | 66,955,779 | 256.11 | MPS |

Timing is observational. The five smaller models were evaluated on CPU, whereas DistilBERT used Apple MPS, so their timings must not be presented as a controlled hardware comparison.

## Per-class evidence

The tables below report F1. The generated `comparison.md` and `results.json` retain precision, recall, F1 and support for every class.

### Full official test

| Model | Negative F1 (n=196) | Neutral F1 (n=196) | Positive F1 (n=728) |
|---|---:|---:|---:|
| TF-IDF | 0.3827 | 0.1794 | 0.8195 |
| LSTM | 0.3605 | 0.1322 | 0.8053 |
| GRU | 0.4298 | 0.1504 | 0.8007 |
| TextCNN | 0.4412 | 0.0952 | 0.8130 |
| ATAE-LSTM | 0.3759 | 0.2888 | 0.7749 |
| DistilBERT | 0.7763 | 0.4842 | 0.8991 |

### Mixed-polarity multi-aspect subset

| Model | Negative F1 (n=60) | Neutral F1 (n=83) | Positive F1 (n=85) |
|---|---:|---:|---:|
| TF-IDF | 0.3778 | 0.0230 | 0.5950 |
| LSTM | 0.3684 | 0.0460 | 0.5647 |
| GRU | 0.3750 | 0.0235 | 0.5483 |
| TextCNN | 0.3689 | 0.0000 | 0.5630 |
| ATAE-LSTM | 0.4553 | 0.3423 | 0.5495 |
| DistilBERT | 0.7680 | 0.4483 | 0.7256 |

## Interpretation for A3

DistilBERT is strongest on the full test and the mixed-polarity subset, but it also has by far the largest artifact and parameter count. ATAE-LSTM has lower full-test accuracy than every review-only model, yet its mixed macro-F1 and neutral F1 are materially stronger. This supports analysing aspect conditioning rather than assuming that aggregate accuracy alone identifies the most useful lightweight model.

The exploratory GRU and TextCNN do not overturn the submitted four-model conclusion. Neither exceeds TF-IDF on full-test accuracy, and neither closes the review-only gap on mixed-polarity neutral examples. TextCNN's zero neutral F1 on the mixed subset is retained as an unexpected negative result.

The shared prediction file produced 55 mixed-subset cases where both aspect-conditioned models were correct while every selected review-only model was wrong, and three cases in the opposite direction. There were 131 examples missed by all six models and 454 examples with cross-model disagreement. Candidate examples retain official-test order and still require qualitative group review before being quoted in A3.

Apple MPS operations can remain nondeterministic despite the fixed seed. Report claims must therefore identify this frozen artifact commit and prediction hash rather than assume bit-identical reruns.

## Reproduction

```bash
.venv/bin/python scripts/select_text_cnn_config.py
.venv/bin/python scripts/train_selected_text_cnn.py --all-six --device auto
.venv/bin/python -m src.absa.evaluation.runner \
  --models tfidf target_lstm target_gru text_cnn atae_lstm distilbert \
  --device auto
```

The final command writes the machine-readable manifest, shared predictions, complete comparison, confusion matrices and error candidates to `outputs/absa/evaluation-six-model/`.
