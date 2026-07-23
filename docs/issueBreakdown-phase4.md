# Issue Breakdown IV — ReviewPulse v3.0 ABSA

This is the implementation track for **ReviewPulse v3.0**, delivered for DLE602 Assessment 3. It follows the completed ISY503 v1.0/v2.x work: v3 changes the learning task from one binary label per review to a three-class prediction for each supplied aspect.

The GitHub tracker is [#72](https://github.com/lfariabr/review-pulse/issues/72). This document records the execution order and architectural guardrails; each linked issue is the detailed source of work and acceptance criteria.

## Scope boundary

### Core delivery

- SemEval-2014 Task 4 **Restaurants** only.
- Gold aspect terms for training and evaluation; manually supplied aspects in the app.
- Three labels in fixed order: `negative`, `neutral`, `positive`.
- Four v3 models: TF-IDF review-only baseline, target-agnostic LSTM, ATAE-LSTM, and DistilBERT sentence pair.
- Sentence-grouped splits, shared evaluation, mixed-polarity multi-aspect analysis, reproducible artifacts and an aspect-aware Streamlit flow.

### Deferred scope

Laptops, automatic aspect extraction, topic modelling, multi-seed variance, perturbation faithfulness testing, hosted artifacts and deeper DistilBERT tuning are stretch work. None may delay a working Restaurants pipeline.

## Architecture guardrails

1. **Preserve v2.3.0.** The legacy `predict_sentiment(text, model_name)` API, binary artifacts and current commands must not change as a side effect of v3 work.
2. **Isolate the new task.** New code lives under `src/absa/`; v3 models, vocabularies, metrics and plots live below `outputs/absa/`.
3. **Keep contracts narrow.** The v3 public surface is `predict_aspects(review, aspects, model_name)`. It returns one compatible result per requested aspect and does not widen legacy predictor interfaces.
4. **Protect data alignment.** SemEval raw text and character offsets are the source of truth. Do not pass offset-destroying legacy `clean_text()` output to ABSA parsing or evidence views.
5. **Prevent leakage.** Group by `sentence_id` before splitting aspect instances. Fit every vocabulary, tokenizer and TF-IDF feature on training data only.
6. **Separate two meanings of conflict.** Count and exclude the original SemEval `conflict` label. The analytical subset is called the **mixed-polarity multi-aspect subset**: at least two gold aspects with different retained polarities in one sentence.
7. **Keep comparison fair.** Every v3 model uses the same retained examples, label order and test set. Review-only models deliberately receive only the review text; their contradiction on mixed-polarity examples is part of RQ1.
8. **Limit explainability claims.** ATAE-LSTM attention and optional Transformer attribution/attention are indicative token-level evidence, not model reasoning or causal explanation. TF-IDF and target-agnostic LSTM need no heatmap.
9. **No broad refactor during delivery.** Reuse v2 patterns through adapters first. Extract generic shared code only after the v3 path works and tests protect it.
10. **Verify clean loading.** Every submitted v3 artifact must load from a clean process with its label mapping and metadata intact.

## Target v3 boundary

```text
src/absa/
  config.py, labels.py, checkpointing.py
  data/              # SemEval parser, audit, grouped splits, schema
  tokenization/      # sequence spans and transformer pair encoding
  models/            # baseline, target LSTM, ATAE-LSTM, DistilBERT
  training/          # one trainer per model family
  evaluation/        # metrics, subsets and common runner
  inference/         # loaders, predictors, registry and public API
  interpretability/  # attention / optional attribution alignment
tests/absa/
outputs/absa/
docs/dle602-a3/
```

## Ordered issue track

PR #90 merged issues #73-#83 and #86-#87 into `main`; #91 and #84 are also complete. The remaining core critical path, reconciled with the submitted A2 report, is `#85 -> #88 -> #89`.

| Order | Issue | Suggested short branch | Depends on | Exit evidence |
|---:|---|---|---|---|
| 1 | [#73](https://github.com/lfariabr/review-pulse/issues/73) environment + v2.3.0 baseline | `chore/v3-environment-baseline` | — | Constraints and legacy baseline recorded |
| 2 | [#74](https://github.com/lfariabr/review-pulse/issues/74) phase-4 breakdown + guardrails | `docs/v3-phase4-guardrails` | #73 | This document and non-regression policy |
| 3 | [#75](https://github.com/lfariabr/review-pulse/issues/75) isolated v3 scaffold | `feat/absa-scaffold` | #73, #74 | `src/absa`, `tests/absa`, labels and boundary tests |
| 4 | [#76](https://github.com/lfariabr/review-pulse/issues/76) SemEval Restaurants provenance | `docs/absa-semeval-restaurants` | #75 | Acquisition and checksum record |
| 5 | [#77](https://github.com/lfariabr/review-pulse/issues/77) parser + offset audit | `feat/absa-parser-audit` | #76 | Canonical records and conflict report |
| 6 | [#78](https://github.com/lfariabr/review-pulse/issues/78) grouped splits | `test/absa-grouped-splits` | #77 | Deterministic no-overlap assertions |
| 7 | [#79](https://github.com/lfariabr/review-pulse/issues/79) multiclass evaluation | `feat/absa-multiclass-evaluation` | #77, #78 | Metrics, plots and subset tests |
| 8 | [#80](https://github.com/lfariabr/review-pulse/issues/80) TF-IDF baseline | `feat/absa-tfidf-baseline` | #78, #79 | Artifact, adapter and common metrics |
| 9 | [#81](https://github.com/lfariabr/review-pulse/issues/81) target-agnostic LSTM | `feat/absa-target-lstm` | #78, #79 | Three-logit checkpoint and history |
| 10 | [#82](https://github.com/lfariabr/review-pulse/issues/82) ATAE-LSTM | `feat/absa-atae-lstm` | #78, #79, #81 | Aspect sensitivity and aligned attention |
| 11 | [#83](https://github.com/lfariabr/review-pulse/issues/83) DistilBERT pair model | `feat/absa-distilbert` | #78, #79 | Pair checkpoint and common metrics |
| 12 | [#91](https://github.com/lfariabr/review-pulse/issues/91) training reproducibility + regularisation | `fix/absa-training-reproducibility` | #81–#83 | Seeds, histories, early stopping and restored best checkpoints |
| 13 | [#84](https://github.com/lfariabr/review-pulse/issues/84) four-model comparison | `feat/absa-comparison` | #80–#83, #91 | Tables, efficiency evidence and errors |
| 14 | [#85](https://github.com/lfariabr/review-pulse/issues/85) token evidence | `feat/absa-token-evidence` | #82, #83 | Caveated aligned evidence |
| 15 | [#86](https://github.com/lfariabr/review-pulse/issues/86) legacy/v3 app | `feat/app-absa-workflow` | #80–#85 | Ordered multi-aspect user flow |
| 16 | [#87](https://github.com/lfariabr/review-pulse/issues/87) integration + clean load | `test/absa-integration` | #80–#86 | Legacy and v3 smoke coverage |
| 17 | [#88](https://github.com/lfariabr/review-pulse/issues/88) report + results package | `docs/dle602-a3-report` | #84–#87, #91 | Evidence-backed report draft |
| 18 | [#89](https://github.com/lfariabr/review-pulse/issues/89) v3.0.0 release | `release/v3.0.0` | #87, #88, #91 | Clean package, notes and tag |

## Branch and merge policy

PR #90 completed the long-lived integration phase and established the v3 implementation baseline on `main`. Each remaining issue now starts from current `main`, receives focused validation and merges through its own reviewed pull request. Do not reuse or revive the merged integration branch.

## Verification commands

```bash
# Legacy fast baseline; Amazon-data tests require local gitignored files.
.venv/bin/python -m pytest tests/ -q -m "not slow"

# Future v3-focused checks.
.venv/bin/python -m pytest tests/absa/ -q

# Whole suite before integration/release.
.venv/bin/python -m pytest tests/ -q
```

The exact Python environment and the initial legacy result are recorded in [`docs/dle602-a3/environment-baseline.md`](dle602-a3/environment-baseline.md) and [`constraints-a3.txt`](../constraints-a3.txt).
