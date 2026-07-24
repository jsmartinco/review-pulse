# Indicative Token Evidence Protocol

## Purpose

ReviewPulse v3 answers RQ3 with human-readable token evidence for the two aspect-conditioned models. These views help inspect which visible review tokens receive larger model-specific scores for a supplied aspect. They do **not** expose reasoning and are not causal explanations.

TF-IDF and the target-agnostic LSTM are review-only baselines. Their inference payloads and the Streamlit interface explicitly report token evidence as unsupported instead of displaying an empty or invented heatmap.

## Methods

### ATAE-LSTM

The ATAE-LSTM view uses the learned aspect-conditioned attention distribution returned by the implemented model. Each non-padding attention weight is aligned to the exact review token span used by the recurrent tokenizer. Capitalisation, punctuation and visible whitespace are preserved in the heatmap.

Changing the aspect changes the aspect embedding supplied to the attention layer and can therefore change both the prediction and token weights. Attention concentration remains indicative only; it does not establish which token caused the output.

### DistilBERT

The DistilBERT view uses **gradient × input attribution for the predicted class**:

1. The fast tokenizer encodes `(review, aspect)` and returns offset mappings.
2. The predicted-class logit is differentiated with respect to the input embeddings.
3. The L2 norm of each element-wise gradient × embedding vector becomes a non-negative wordpiece score.
4. Only first-sequence wordpieces are retained; aspect-sequence and special-token scores are excluded.
5. Wordpieces are aggregated onto exact visible review spans and normalised within that aspect view.

This is deterministic for a fixed checkpoint and input while the model is in evaluation mode. It is a local diagnostic that can be sensitive to gradients and checkpoint state; it is not a faithful causal explanation.

## Alignment and presentation contract

Every supported evidence payload contains:

- `status: supported`;
- the supplied `aspect`;
- a named `method`;
- exact `token`, `start` and `end` fields plus a numeric `score`;
- a shared non-causal caveat and a method-specific limitation.

Unsupported models return `status: unsupported`, an empty token list and an explicit reason. The Streamlit view shades exact review spans, preserves punctuation and whitespace, escapes HTML, states that darker shading means a larger within-view score and displays both caveats.

Automated tests cover exact ATAE offsets, punctuation, DistilBERT subword aggregation, ignored aspect-sequence scores, deterministic gradient attribution, safe heatmap rendering and supported/unsupported payloads.

## Representative mixed-polarity export

Generate the report-ready machine-readable bundle with the verified local artifacts:

```bash
.venv/bin/python scripts/export_absa_evidence.py
```

The gitignored output is:

```text
outputs/absa/evidence/rq3_food_service.json
```

The fixed example is:

> Great food but the service was dreadful!

with aspects `food` and `service`. A verified run on the #84 checkpoints produced distinct aspect views:

| Model / aspect | Prediction | Confidence | Highest-scored visible tokens |
|---|---|---:|---|
| ATAE-LSTM / food | positive | 86.4% | `Great` 0.241, `food` 0.131, `was` 0.127 |
| ATAE-LSTM / service | positive | 74.2% | `Great` 0.193, `!` 0.132, `dreadful` 0.130 |
| DistilBERT / food | negative | 82.7% | `dreadful` 0.288, `the` 0.163, `service` 0.151 |
| DistilBERT / service | negative | 91.3% | `dreadful` 0.488, `food` 0.095, `service` 0.095 |

The example is retained honestly: evidence changes with the aspect, but the labels do not necessarily resolve both polarities correctly. It supports visual inspection for RQ3 and must not be presented as proof of faithful explanation.
