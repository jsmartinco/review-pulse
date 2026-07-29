# ReviewPulse

ReviewPulse is a phased sentiment-analysis project:

- **v1.0 / v2.x — ISY503:** binary sentiment for an entire Amazon product review.
- **v3.0 — DLE602:** three-class aspect-based sentiment analysis (ABSA) for one or more manually supplied restaurant aspects.

The current DLE602 release compares six models on SemEval-2014 Restaurants. TF-IDF, LSTM, GRU and TextCNN receive only the review; ATAE-LSTM and DistilBERT receive the `(review, aspect)` pair. GRU and TextCNN are exploratory controls, while the submitted four-model experiment remains canonical.

The Streamlit application preserves both academic phases as separate workflows. Token evidence is available only for ATAE-LSTM attention and DistilBERT gradient × input attribution, and is presented as indicative rather than causal.

[Main study repo](https://github.com/lfariabr/masters-swe-ai)

## DLE602 v3 Results

The official Restaurants test contains 1,120 retained three-class aspect instances. The mixed-polarity subset contains 228 instances across 80 sentences.

| Model | Scope | Test accuracy | Test macro-F1 | Mixed accuracy | Mixed macro-F1 |
|---|---|---:|---:|---:|---:|
| TF-IDF | Core | 0.7018 | 0.4605 | 0.4430 | 0.3319 |
| LSTM | Core | 0.6687 | 0.4326 | 0.4167 | 0.3264 |
| GRU | Exploratory | 0.6750 | 0.4603 | 0.4079 | 0.3156 |
| TextCNN | Exploratory | 0.6893 | 0.4498 | 0.4167 | 0.3106 |
| ATAE-LSTM | Core | 0.6438 | 0.4799 | 0.4737 | 0.4491 |
| DistilBERT | Core | **0.8250** | **0.7199** | **0.6667** | **0.6473** |

The frozen six-model provenance, efficiency measurements and limitations are documented in [`docs/dle602-a3/six-model-results.md`](docs/dle602-a3/six-model-results.md).

## ISY503 v2 Results

Held-out test split: 1,159 reviews, stratified 70/15/15 split, seed=42.

| Model | Accuracy | F1 | Misclassified |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 82.7% | 81.9% | 201 |
| BiLSTM + GloVe | 81.0% | 80.3% | 220 |
| DistilBERT | 88.2% | 88.6% | 137 |

The baseline remains the simplest strong benchmark. DistilBERT is the strongest model in this build. BiLSTM demonstrates the neural sequence-model path required by the assessment.

## Project Structure

```text
review-pulse/
  app.py                    # Streamlit UI: layout, input, result display
  requirements.txt
  .streamlit/config.toml    # Streamlit watcher config

  src/
    config.py               # paths, model names, prediction threshold
    app/                    # Streamlit service helpers
      service.py            # cached app loaders + model availability helpers

    data/                   # parser, preprocessing, EDA feature helpers
      parser.py             # pseudo-XML parser -> DataFrame
      preprocess.py         # label audit, cleaning, outlier removal, splits
      features.py           # EDA helpers

    tokenization/           # shared sequence/vocab + DistilBERT tokenization
      vocab.py              # vocab save/load/build helpers
      sequence.py           # BiLSTM Dataset/DataLoader/tokenize helpers
      bert.py               # DistilBERT tokenizer + Dataset/DataLoader helpers

    models/                 # model definitions
      baseline.py           # TF-IDF + LogisticRegression pipeline factory
      bilstm.py             # BiLSTMSentiment nn.Module
      bert.py               # DistilBERTSentiment HF wrapper

    training/               # training orchestration and metric helpers
      baseline.py           # TF-IDF baseline train/evaluate/load
      bilstm.py             # BiLSTM training loop + checkpointing
      bert.py               # DistilBERT training stage orchestration

    checkpoint_bert.py      # DistilBERT checkpoint save/load helpers

    inference/              # single-text prediction package
      loaders.py            # artifact/model loading and caches
      predictors.py         # Predictor protocol + concrete predictors
      registry.py           # predictor registry
      api.py                # predict_sentiment() public API

    evaluation/             # batch evaluation package
      metrics.py            # metric computation
      plots.py              # confusion matrix PNG
      errors.py             # error-analysis CSV
      bilstm.py             # BiLSTM + baseline evaluation runner
      bert.py               # DistilBERT evaluation runner
      runner.py             # CLI orchestration

    utils/
      samples.py            # Streamlit demo review samples

  tests/                    # pytest suite
  notebooks/                # EDA notebook
  data/                     # local raw .review files
  outputs/                  # committed model artifacts + generated reports
  embeddings/               # optional GloVe files, gitignored
  docs/                     # architecture, issue breakdowns, release notes
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -c constraints-a3.txt
```

Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt -c constraints-a3.txt
```

Place `.review` data files under `data/`:

```text
data/
  books/positive.review
  books/negative.review
  dvd/positive.review
  dvd/negative.review
  electronics/positive.review
  electronics/negative.review
  kitchen_&_housewares/positive.review
  kitchen_&_housewares/negative.review
```

## Model Artifacts

The app expects trained artifacts in `outputs/`:

| Artifact | Purpose | Committed |
|---|---|:---:|
| `outputs/baseline.joblib` | TF-IDF + Logistic Regression pipeline | Yes |
| `outputs/vocab.json` | BiLSTM vocabulary | Yes |
| `outputs/bilstm.pt` | BiLSTM checkpoint | Yes |
| `outputs/distilbert.pt` | Compact DistilBERT checkpoint | Yes |

Generated evaluation reports such as PNG confusion matrices and CSV error analysis files are gitignored.

DistilBERT note: `outputs/distilbert.pt` is a compact checkpoint. It stores the classification head and fine-tuned encoder layers, but frozen base encoder weights are loaded from `distilbert-base-uncased` through Hugging Face. A fresh machine may need network access or a pre-populated Hugging Face cache.

See `docs/architecture.md` for the full artifact policy and DistilBERT model-card notes.

## GloVe Embeddings

GloVe pre-trained vectors are optional for BiLSTM training.

To enable GloVe:

1. Download `glove.6B.zip` from Stanford NLP.
2. Extract `glove.6B.100d.txt`.
3. Place it in `embeddings/glove.6B.100d.txt`.
4. Re-run BiLSTM training.

If the file is absent, BiLSTM training proceeds with randomly initialized embeddings and prints a warning. The `embeddings/` directory is gitignored.

## Train

Train the TF-IDF baseline:

```bash
python -m src.training.baseline
```

Train the BiLSTM:

```bash
python -m src.training.bilstm
```

Train DistilBERT:

```bash
python -m src.training.bert
```

`src.training.bert` uses Hugging Face `distilbert-base-uncased`, freezes the encoder for head training, then fine-tunes the last encoder layers. It writes the deployment artifact to `outputs/distilbert.pt`.

## Evaluate

```bash
python -m src.evaluation.runner
```

Evaluation loads the trained artifacts, runs the held-out test split, prints metrics, and writes generated reports to `outputs/`.

Evaluation helpers can also run without file side effects:

```python
from src.evaluation import run_evaluation

metrics = run_evaluation(save_outputs=False)
```

## Run The App

```bash
.venv/bin/streamlit run app.py
```

The landing page links to two deliberately separate workflows:

- **ReviewPulse v2.3.0 (ISY503):** binary sentiment for one complete Amazon review.
- **ReviewPulse v3.0.0 (DLE602):** three-class sentiment for one or more manually supplied restaurant aspects.

The v3 page includes mixed-polarity samples, supports the four-model comparison ladder and shows aligned ATAE-LSTM attention or DistilBERT gradient × input attribution only as indicative evidence, not model reasoning. TF-IDF and the target-agnostic LSTM explicitly report token evidence as unsupported. SemEval Restaurants data and v3 artifacts are local inputs; a missing model is reported as a controlled application error rather than falling back to a v2 model.

After preparing the local v3 artifacts, verify the ABSA path with:

```bash
.venv/bin/python -m pytest tests/absa -q
.venv/bin/python scripts/smoke_absa.py
.venv/bin/python scripts/export_absa_evidence.py
```

See `docs/dle602-a3/v3-smoke.md` for the required local artifacts, `docs/dle602-a3/semeval-restaurants.md` for data provenance, `docs/dle602-a3/training-protocol.md` for seed, early-stopping and best-checkpoint rules, and `docs/dle602-a3/token-evidence.md` for the RQ3 methods and limitations.

Regenerate verified v3 artifacts and produce the common four-model A3 comparison with:

```bash
.venv/bin/python -m src.absa.training.runner --device auto
.venv/bin/python -m src.absa.evaluation.runner --device auto
```

The evaluation rejects pre-#91 artifacts by default and writes metrics, shared predictions, efficiency evidence, error candidates and confusion matrices below `outputs/absa/evaluation/`. See `docs/dle602-a3/evaluation-protocol.md` for the measurement definitions and staged commands.

The optional GRU and TextCNN candidates remain outside those canonical four-model outputs. Train and smoke them independently with:

```bash
.venv/bin/python -m src.absa.training.runner --models target_gru --device cpu
.venv/bin/python scripts/smoke_target_gru.py

.venv/bin/python scripts/select_text_cnn_config.py
.venv/bin/python scripts/train_selected_text_cnn.py --device cpu
.venv/bin/python scripts/smoke_text_cnn.py
```

The matched GRU controls and results are documented in `docs/dle602-a3/target-gru.md`; the bounded development-only CNN gate and architecture contract are documented in `docs/dle602-a3/text-cnn.md`. Issue #96 keeps their joint six-model evidence explicitly separate from the canonical four-model outputs.

Regenerate and evaluate the explicit six-model supplement from one frozen commit:

```bash
.venv/bin/python scripts/select_text_cnn_config.py
.venv/bin/python scripts/train_selected_text_cnn.py --all-six --device auto
.venv/bin/python -m src.absa.evaluation.runner \
  --models tfidf target_lstm target_gru text_cnn atae_lstm distilbert \
  --device auto
```

This writes separate evidence below `outputs/absa/evaluation-six-model/`; the default four-model commands and `outputs/absa/evaluation/` remain unchanged. The verified same-commit result and A3 interpretation are recorded in `docs/dle602-a3/six-model-results.md`.

## Inference API

```python
from src.inference import predict_sentiment

result = predict_sentiment(
    "This blender is great.",
    model_name="distilbert",
)
```

Response shape:

```python
{
    "label": "Positive review",
    "confidence": 0.923,
    "model": "distilbert",
}
```

Available model names:

- `"baseline"`
- `"bilstm"`
- `"distilbert"`

Future models can be registered through `register_predictor()` in `src.inference`. The implementation is split across `src/inference/loaders.py`, `src/inference/predictors.py`, `src/inference/registry.py`, and `src/inference/api.py`.

## Run Tests

Fast suite:

```bash
pytest tests/ -q -m "not slow"
```

Full suite:

```bash
pytest tests/
```

Current status:

- Full suite on the #89 package candidate: 269 passed, 8 skipped.
- Skips apply only when the optional, gitignored legacy Amazon dataset is absent.

## Documentation Map

- `docs/architecture.md` - current architecture, data flow, artifact policy, DistilBERT model card
- `docs/issueBreakdown-phase1.md` - original assessment delivery breakdown
- `docs/issueBreakdown-phase2.md` - completed #30-#39 refactor track
- `docs/issueBreakdown-phase3.md` - proposed modular package refactor plan
- `docs/assessment-files/` - presentation outline, individual report template, demo test cases
- `docs/releaseNotes/v1.0.0.md` - baseline + BiLSTM release
- `docs/releaseNotes/v2.0.0.md` - DistilBERT release
- `docs/releaseNotes/v2.1.0.md` - refactor track release
- `docs/releaseNotes/v2.2.0.md` - modular package release
- `docs/releaseNotes/v2.3.0.md` - compatibility wrapper removal release
- `docs/releaseNotes/v3.0.0.md` - consolidated v3 delivery status and release gates
- `docs/dle602-a3/` - v3 environment, SemEval provenance and smoke instructions
- `docs/dle602-a3/submission-package.md` - deterministic A3 ZIP modes and verification

## Issue Creator (batch issue helper)

Use the local `issue_creator` helper to create many issues with one command.

Template file:

- `docs/templates/issue_creator.template.json`
- `docs/issueBreakdown-phaseX.md` (same format as phase3: `### Issue #NN - title`)

Commands:

```bash
# Dry-run (default)
./scripts/issue_creator.sh docs/templates/issue_creator.template.json

# Dry-run from markdown breakdown
./scripts/issue_creator.sh docs/issueBreakdown-phase3.md

# Create for real
./scripts/issue_creator.sh docs/templates/issue_creator.template.json --create

# Create for real from markdown breakdown
./scripts/issue_creator.sh docs/issueBreakdown-phase3.md --create
```

You can also call Python directly:

```bash
python3 scripts/issue_creator.py --template docs/templates/issue_creator.template.json --create
```

If you pass a `.json` path that does not exist but a sibling `.md` exists, the script automatically falls back to the markdown file.
