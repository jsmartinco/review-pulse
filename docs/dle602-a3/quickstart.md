# DLE602 marker quick start

Three self-contained paths. Pick one; none of them depends on the others.

| Path | Start from | Models available | Needs SemEval data |
|---|---|---|---|
| [A](#path-a--run-the-submitted-lightweight-zip) | The submitted ZIP | Five v3 models, plus the complete v2.3.0 workflow | No |
| [B](#path-b--run-all-six-v3-models-from-github) | A GitHub clone | All six v3 models | No |
| [C](#path-c--reproduce-training-and-evaluation) | Path B, plus the licensed corpus | All six, retrained and re-evaluated | Yes |

Every path installs with the reviewed constraint set, `pip install -r requirements.txt -c constraints-a3.txt`, which pins the versions the reported results were produced with.

**Neither inference nor the application requires the SemEval corpus.** The trained artifacts are shipped, so the models predict without it. SemEval is needed only to retrain or re-evaluate, which is Path C. The legacy Amazon `.review` files are likewise not required by any path; they are optional inputs for the ISY503 v1/v2 data pipeline only.

---

## Path A - Run the submitted lightweight ZIP

```bash
unzip ReviewPulse-v3.0.0-DLE602-A3.zip
cd ReviewPulse-v3.0.0

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt -c constraints-a3.txt

python -m pytest -q
streamlit run app.py
```

The application opens on an introduction page. Use the sidebar to reach **ReviewPulse v3.0.0** for aspect-level sentiment, or **ReviewPulse v2.3.0** for the ISY503 review-level workflow.

### What the lightweight package contains

Five v3 artifacts are included, covering every model except the transformer:

| Model | Artifact |
|---|---|
| TF-IDF review-only | `outputs/absa/tfidf_baseline.joblib` |
| LSTM review-only | `outputs/absa/target_lstm.pt` |
| GRU review-only, exploratory | `outputs/absa/target_gru.pt` |
| Text CNN review-only, exploratory | `outputs/absa/text_cnn.pt` |
| ATAE-LSTM aspect-conditioned | `outputs/absa/atae_lstm.pt` |

The complete ISY503 v2.3.0 artifact set is also included, so that page works fully, **including its own DistilBERT model**. That legacy checkpoint (`outputs/distilbert.pt`) is a binary review-level model and is unrelated to the v3 aspect-level DistilBERT.

### The v3 DistilBERT checkpoint is deliberately absent

`outputs/absa/distilbert/` is around 256 MB, which pushes the archive to roughly 288 MB against 52 MB without it. It is excluded to stay inside the submission size limit, as the Assessment 2 risk register anticipated.

Selecting **DistilBERT sentence-pair** on the v3 page therefore reports a model-unavailable error instead of a prediction. The application does not fall back to another model, by design: a missing artifact is always visible rather than silently substituted. To obtain that model, use [Path B](#path-b--run-all-six-v3-models-from-github).

Its measured results are still reported: see the six-model table in [`six-model-results.md`](six-model-results.md) and the frozen evidence it links.

### Verifying the package

`python -m pytest -q` is the verification command for this path. Expect roughly **349 passed and 16 skipped**. Every skip is an intentional absence, not a failure:

| Skips | Reason |
|---:|---|
| 6 | Sample-provenance checks need `outputs/absa/evaluation/predictions.csv`, which is not redistributed |
| 8 | Legacy Amazon `.review` files are not redistributed |
| 2 | The package-builder tests need Git metadata, and an extracted ZIP is not a repository |

Individual light models can also be smoke-tested:

```bash
python scripts/smoke_target_gru.py
python scripts/smoke_text_cnn.py
```

> `scripts/smoke_absa.py` is **not** usable here. It exercises all four canonical models including the v3 DistilBERT, so it fails on this package. Use it on [Path B](#path-b--run-all-six-v3-models-from-github).

This path is not a Git repository, so there is nothing to fetch with Git LFS.

---

## Path B - Run all six v3 models from GitHub

This path includes the full v3 DistilBERT artifact and therefore supports all six v3 models.

```bash
git lfs install
git clone https://github.com/lfariabr/review-pulse.git
cd review-pulse
git lfs pull
```

Confirm the artifacts materialised rather than remaining pointers. Each line should be marked `*`:

```bash
git lfs ls-files -s
```

Expect six entries: the DistilBERT `model.safetensors` at roughly 268 MB, plus the ATAE-LSTM, LSTM, GRU, Text CNN and TF-IDF artifacts. A line marked `-` means the content was not fetched; rerun `git lfs pull`.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt -c constraints-a3.txt

python -m pytest -q
python scripts/smoke_absa.py
streamlit run app.py
```

`scripts/smoke_absa.py` clean-loads all four canonical models and prints one prediction per aspect. Expect roughly **351 passed and 14 skipped** from the suite here: the six provenance skips and the eight legacy-data skips described above, without the two package-builder skips, since this is a real checkout.

Inference runs on CPU regardless of the machine, because every artifact adapter pins `map_location="cpu"`. An accelerator is not required and is not used for prediction. See [`release-verification.md`](release-verification.md) for the recorded evidence.

The clone contains **no** SemEval data. Running the application or the smoke test does not need it.

---

## Path C - Reproduce training and evaluation

Start from a working Path B checkout. This path additionally requires the licensed corpus.

### 1. Obtain and place the corpus

SemEval-2014 Task 4 is not redistributed with this repository. Download the corrected Restaurants training XML and the gold test XML from the [official task page](https://alt.qcri.org/semeval2014/task4/), then:

```bash
python scripts/prepare_semeval_restaurants.py \
  --train /path/to/Restaurants_Train_v2.xml \
  --test  /path/to/Restaurants_Test_Gold.xml
```

The helper validates the XML, copies both files to stable local names and writes a SHA-256 manifest. Verify it at any later point:

```bash
python scripts/prepare_semeval_restaurants.py --verify
```

Provenance, licensing and the redistribution decision are documented in [`semeval-restaurants.md`](semeval-restaurants.md). Cite Pontiki et al. (2014) for any use of the data.

Commands that need the corpus fail with acquisition instructions if it is absent, so a missing dataset is self-explaining rather than a bare traceback.

### 2. Audit the data

```bash
python -m src.absa.data.audit
```

Reports split sizes, label counts, excluded `conflict` rows and offset validity.

### 3. Evaluate

The canonical four-model experiment, which produces the results reported in the assessment:

```bash
python -m src.absa.evaluation.runner --device auto
```

The optional supplemental six-model comparison, adding the exploratory GRU and Text CNN:

```bash
python -m src.absa.evaluation.runner \
  --models tfidf target_lstm target_gru text_cnn atae_lstm distilbert \
  --device auto
```

Both write below `outputs/absa/evaluation/`. Metric definitions, the mixed-polarity subset and the verification gate on unverified artifacts are specified in [`evaluation-protocol.md`](evaluation-protocol.md).

### 4. Retrain, if required

Evaluation uses the shipped artifacts, so retraining is optional and is not needed to reproduce the reported tables.

> **The training runner overwrites the versioned artifacts in place.** Running it replaces the frozen checkpoints and their metrics files, which are the evidence the assessment reports. In a Git checkout, restore them with `git checkout -- outputs/` and confirm with `git status`. Consider retraining on a branch or a copy.

```bash
# Inexpensive: the five small models, CPU is sufficient
python -m src.absa.training.runner \
  --models tfidf target_lstm target_gru text_cnn atae_lstm \
  --device cpu

# Expensive: the transformer
python -m src.absa.training.runner --models distilbert --device auto
```

**Cost.** The five small models train in seconds to tens of seconds each on CPU. DistilBERT is the only expensive step, taking roughly two minutes on Apple MPS in the recorded run and substantially longer on CPU.

**Device selection.** Both runners accept `--device` with `auto`, `cpu`, `mps` or `cuda`. `auto` prefers CUDA, then MPS, then CPU. Timing measured across different devices is not a controlled comparison and must be qualified as such.

Seeds, early stopping, checkpoint selection and the persisted run record are specified in [`training-protocol.md`](training-protocol.md). Retraining overwrites the shipped artifacts, so the reported figures will only be reproduced exactly under the recorded environment and seed.

---

## Further reading

| Document | Covers |
|---|---|
| [`v3-smoke.md`](v3-smoke.md) | Required local artifacts and the clean-load smoke contract |
| [`semeval-restaurants.md`](semeval-restaurants.md) | Corpus provenance, licensing and preparation |
| [`training-protocol.md`](training-protocol.md) | Seeds, regularisation, early stopping, saved run records |
| [`evaluation-protocol.md`](evaluation-protocol.md) | Metrics, mixed-polarity subset, generated evidence files |
| [`six-model-results.md`](six-model-results.md) | Frozen six-model comparison and efficiency measurements |
| [`submission-package.md`](submission-package.md) | Package modes, exclusions and archive verification |
| [`release-verification.md`](release-verification.md) | Clean-room install, LFS, CPU inference and package sizes |
