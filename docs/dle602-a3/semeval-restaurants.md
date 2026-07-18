# SemEval-2014 Restaurants acquisition

ReviewPulse v3 uses the **Restaurants** portion of SemEval-2014 Task 4 for its core ABSA experiments. Laptops is a cross-domain stretch goal and is not required by this setup.

## Provenance and task contract

The [official SemEval-2014 Task 4 page](https://alt.qcri.org/semeval2014/task4/) defines aspect-based sentiment analysis as identifying aspects and their sentiment. Its Restaurants dataset contains over 3,000 English review sentences with human-authored annotations; each aspect term can be `positive`, `negative`, `neutral`, or `conflict`. The XML includes a sentence ID, raw text, aspect term, polarity and (for training data) character offsets.

The accompanying [Data and Tools page](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools) is the authoritative acquisition source. It links the corrected training data (v2.0) and the evaluation test releases, and states that the task data were collected manually within fair use and the data providers' terms and conditions. The task paper is Pontiki et al. (2014), [*SemEval-2014 Task 4: Aspect Based Sentiment Analysis*](https://aclanthology.org/S14-2004/).

## Redistribution treatment

The task website does **not** state an open redistribution licence for the XML corpus; instead it gives the fair-use/third-party-terms notice above. Consequently:

- Raw SemEval XML files and generated SHA manifests are ignored by Git.
- This repository contains acquisition and verification tooling, not a copied corpus or a third-party mirror.
- Obtain the files from the official page and comply with its terms before preparing them locally.
- Cite Pontiki et al. (2014) in the report and any publication that uses the data.

## Obtain and prepare the two files

1. On the official Data and Tools page, obtain the corrected Restaurants training XML (Train Data v2.0) and the annotated/gold Restaurants test XML appropriate for the final evaluation release.
2. Keep the downloaded files outside the repository or in a temporary directory. They may retain their original filenames.
3. Run:

```bash
.venv/bin/python scripts/prepare_semeval_restaurants.py \
  --train /path/to/official/Restaurants_Train_v2.xml \
  --test /path/to/official/Restaurants_Test_Gold.xml
```

The helper rejects malformed XML, copies the inputs to stable local names and writes:

```text
data/semeval2014/restaurants/
  restaurants_train.xml
  restaurants_test.xml
  sha256.json
```

The source filenames may vary between official releases; the command arguments, rather than their names, define train and test roles. Do not replace the gold test file after model selection begins.

## Checksum verification

After preparation, verify that the local raw files still match the manifest:

```bash
.venv/bin/python scripts/prepare_semeval_restaurants.py --verify
```

The generated `sha256.json` records SHA-256, byte count and original filename for both raw inputs. It is intentionally local because it describes the exact files the user acquired under the source terms. Record its two digests in the A3 experiment log before training.

## v3 label policy

The parser in #77 will count all four original values. The core experiment retains only `negative`, `neutral` and `positive`, and reports excluded `conflict` rows separately. This is not the same as the **mixed-polarity multi-aspect subset**, which will be computed from sentences containing at least two retained aspects with different polarities.
