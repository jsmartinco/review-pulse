"""Load verified ABSA artifacts behind one batch-prediction contract."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..data.schema import AspectExample
from ..labels import ID_TO_LABEL, LABELS
from ..model_order import CORE_MODEL_ORDER, SIX_MODEL_ORDER
from ..models.atae_lstm import ATAELSTM
from ..models.distilbert import ABSADistilBERT
from ..models.target_gru import TargetAgnosticGRU
from ..models.target_lstm import TargetAgnosticLSTM
from ..models.text_cnn import TextCNN
from ..tokenization.bert_dataset import AspectPairDataset
from ..tokenization.sequence import encode


MODEL_NAMES = {
    "tfidf": "TF-IDF review-only",
    "target_lstm": "LSTM review-only",
    "target_gru": "GRU review-only (exploratory)",
    "text_cnn": "Text CNN review-only (exploratory)",
    "atae_lstm": "ATAE-LSTM",
    "distilbert": "DistilBERT sentence-pair",
}
RUN_RECORD_FILES = {
    "tfidf": "tfidf_baseline_metrics.json",
    "target_lstm": "target_lstm_metrics.json",
    "target_gru": "target_gru_metrics.json",
    "text_cnn": "text_cnn_metrics.json",
    "atae_lstm": "atae_lstm_metrics.json",
    "distilbert": "distilbert_metrics.json",
}


class UnverifiedArtifactError(ValueError):
    """Raised when an artifact predates the reproducible #91 protocol."""


@dataclass(frozen=True)
class LoadedEvaluator:
    """A loaded model plus evidence required for fair efficiency measurement."""

    key: str
    display_name: str
    device: str
    artifact_bytes: int
    parameter_count: int | None
    training_seconds: float | None
    training_config: dict[str, object]
    provenance: dict[str, object]
    load_seconds: float
    predict_batch: Callable[[Sequence[AspectExample]], list[str]]


def preferred_device() -> torch.device:
    """Prefer CUDA, then Apple Metal, while retaining CPU compatibility."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def artifact_size(path: Path) -> int:
    """Return the recursive size of one file or artifact directory."""
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    raise FileNotFoundError(path)


def _training_seconds(metadata: dict[str, object], source: Path, require_verified: bool) -> float | None:
    missing = [key for key in ("config", "training_seconds") if key not in metadata]
    if missing and require_verified:
        raise UnverifiedArtifactError(
            f"{source} is missing #91 metadata: {', '.join(missing)}. Regenerate the artifact."
        )
    value = metadata.get("training_seconds")
    return float(value) if value is not None else None


def _read_run_records(
    artifact_dir: Path,
    require_verified: bool,
    model_keys: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for key in model_keys:
        filename = RUN_RECORD_FILES[key]
        path = artifact_dir / filename
        record = json.loads(path.read_text())
        _training_seconds(record, path, require_verified)
        if require_verified and "provenance" not in record:
            raise UnverifiedArtifactError(
                f"{path} has no shared training provenance. Regenerate it with the training runner."
            )
        records[key] = record

    if require_verified:
        identity_keys = ("git_commit", "train_sha256", "test_sha256")
        reference_key = model_keys[0]
        reference = _provenance(records[reference_key])
        missing = [key for key in identity_keys if key not in reference]
        if missing:
            raise UnverifiedArtifactError(
                f"{reference_key} provenance is missing comparison identity: {', '.join(missing)}"
            )
        identity = {key: reference[key] for key in identity_keys}
        seeds = set()
        for model_key, record in records.items():
            provenance = _provenance(record)
            candidate = {key: provenance.get(key) for key in identity_keys}
            if candidate != identity:
                raise UnverifiedArtifactError(
                    f"{model_key} was not generated from the same commit and dataset checksums"
                )
            seeds.add(_config(record).get("seed"))
        if len(seeds) != 1 or None in seeds:
            raise UnverifiedArtifactError(
                "The selected artifacts do not share one explicit training seed: "
                f"{sorted(seeds, key=str)}"
            )
    return records


def _normalise_model_keys(model_keys: Sequence[str]) -> tuple[str, ...]:
    keys = tuple(model_keys)
    invalid = sorted(set(keys) - set(SIX_MODEL_ORDER))
    if invalid:
        raise ValueError(f"Unknown evaluation models: {invalid}")
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Evaluation models must be non-empty and unique")
    expected = tuple(key for key in SIX_MODEL_ORDER if key in set(keys))
    if keys != expected:
        raise ValueError(f"Evaluation models must follow this order: {expected}")
    return keys


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _assert_labels(
    labels: object,
    source: Path,
    require_verified: bool,
) -> None:
    actual = list(labels) if labels is not None else []
    if require_verified and actual != list(LABELS):
        raise UnverifiedArtifactError(
            f"{source} label order must be {list(LABELS)}"
        )


def _config(record: dict[str, object]) -> dict[str, object]:
    value = record.get("config")
    return dict(value) if isinstance(value, dict) else {}


def _provenance(record: dict[str, object]) -> dict[str, object]:
    value = record.get("provenance")
    return dict(value) if isinstance(value, dict) else {}


def _assert_embedded_metadata(
    embedded: dict[str, object],
    record: dict[str, object],
    source: Path,
    require_verified: bool,
) -> None:
    if not require_verified:
        return
    for key in ("config", "training_seconds"):
        if embedded.get(key) != record.get(key):
            raise UnverifiedArtifactError(
                f"{source} {key} does not match its metrics run record"
            )


def _batched(rows: Sequence[AspectExample], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _load_tfidf(
    artifact_dir: Path,
    record: dict[str, object],
    require_verified: bool,
) -> LoadedEvaluator:
    model_path = artifact_dir / "tfidf_baseline.joblib"
    started = perf_counter()
    training_seconds = _training_seconds(
        record,
        artifact_dir / "tfidf_baseline_metrics.json",
        require_verified,
    )
    model = joblib.load(model_path)
    _assert_labels(model.classes_, model_path, require_verified)
    classifier = model.named_steps["classifier"]
    parameter_count = int(classifier.coef_.size + classifier.intercept_.size)
    load_seconds = perf_counter() - started

    def predict(rows: Sequence[AspectExample]) -> list[str]:
        return model.predict([row.review_raw for row in rows]).tolist()

    return LoadedEvaluator(
        key="tfidf",
        display_name=MODEL_NAMES["tfidf"],
        device="cpu",
        artifact_bytes=artifact_size(model_path),
        parameter_count=parameter_count,
        training_seconds=training_seconds,
        training_config=_config(record),
        provenance=_provenance(record),
        load_seconds=load_seconds,
        predict_batch=predict,
    )


def _load_target_lstm(
    artifact_dir: Path,
    record: dict[str, object],
    require_verified: bool,
    batch_size: int,
) -> LoadedEvaluator:
    path = artifact_dir / "target_lstm.pt"
    started = perf_counter()
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    _assert_labels(artifact.get("labels"), path, require_verified)
    training_seconds = _training_seconds(record, path, require_verified)
    _assert_embedded_metadata(artifact, record, path, require_verified)
    vocab = artifact["vocab"]
    model = TargetAgnosticLSTM(len(vocab))
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    load_seconds = perf_counter() - started

    def predict(rows: Sequence[AspectExample]) -> list[str]:
        predicted: list[str] = []
        with torch.no_grad():
            for batch in _batched(rows, batch_size):
                logits = model(encode([row.review_raw for row in batch], vocab))
                predicted.extend(ID_TO_LABEL[int(index)] for index in logits.argmax(1))
        return predicted

    return LoadedEvaluator(
        key="target_lstm",
        display_name=MODEL_NAMES["target_lstm"],
        device="cpu",
        artifact_bytes=artifact_size(path),
        parameter_count=_parameter_count(model),
        training_seconds=training_seconds,
        training_config=_config(record),
        provenance=_provenance(record),
        load_seconds=load_seconds,
        predict_batch=predict,
    )


def _load_target_gru(
    artifact_dir: Path,
    record: dict[str, object],
    require_verified: bool,
    batch_size: int,
) -> LoadedEvaluator:
    path = artifact_dir / "target_gru.pt"
    started = perf_counter()
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    _assert_labels(artifact.get("labels"), path, require_verified)
    training_seconds = _training_seconds(record, path, require_verified)
    _assert_embedded_metadata(artifact, record, path, require_verified)
    vocab = artifact["vocab"]
    config = _config(record)
    max_length = int(config.get("max_length", 80))
    model = TargetAgnosticGRU(
        len(vocab),
        embedding_dim=int(config.get("embedding_dim", 100)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.5)),
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    load_seconds = perf_counter() - started

    def predict(rows: Sequence[AspectExample]) -> list[str]:
        predicted: list[str] = []
        with torch.no_grad():
            for batch in _batched(rows, batch_size):
                logits = model(
                    encode(
                        [row.review_raw for row in batch],
                        vocab,
                        max_length,
                    )
                )
                predicted.extend(
                    ID_TO_LABEL[int(index)] for index in logits.argmax(1)
                )
        return predicted

    return LoadedEvaluator(
        key="target_gru",
        display_name=MODEL_NAMES["target_gru"],
        device="cpu",
        artifact_bytes=artifact_size(path),
        parameter_count=_parameter_count(model),
        training_seconds=training_seconds,
        training_config=config,
        provenance=_provenance(record),
        load_seconds=load_seconds,
        predict_batch=predict,
    )


def load_target_gru_evaluator(
    artifact_dir: Path,
    *,
    require_verified: bool = True,
    batch_size: int = 64,
) -> LoadedEvaluator:
    """Load the optional GRU without widening the canonical four-model runner."""
    if batch_size < 1:
        raise ValueError("Evaluation batch size must be at least 1")
    metrics_path = artifact_dir / "target_gru_metrics.json"
    record = json.loads(metrics_path.read_text())
    _training_seconds(record, metrics_path, require_verified)
    if require_verified and "provenance" not in record:
        raise UnverifiedArtifactError(
            f"{metrics_path} has no shared training provenance. "
            "Regenerate it with the training runner."
        )
    return _load_target_gru(
        artifact_dir,
        record,
        require_verified,
        batch_size,
    )


def _load_text_cnn(
    artifact_dir: Path,
    record: dict[str, object],
    require_verified: bool,
    batch_size: int,
) -> LoadedEvaluator:
    path = artifact_dir / "text_cnn.pt"
    started = perf_counter()
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    _assert_labels(artifact.get("labels"), path, require_verified)
    training_seconds = _training_seconds(record, path, require_verified)
    _assert_embedded_metadata(artifact, record, path, require_verified)
    vocab = artifact["vocab"]
    config = _config(record)
    max_length = int(config.get("max_length", 80))
    model = TextCNN(
        len(vocab),
        embedding_dim=int(config.get("embedding_dim", 100)),
        num_filters=int(config.get("num_filters", 100)),
        filter_widths=tuple(config.get("filter_widths", (3, 4, 5))),
        dropout=float(config.get("dropout", 0.5)),
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    load_seconds = perf_counter() - started

    def predict(rows: Sequence[AspectExample]) -> list[str]:
        predicted: list[str] = []
        with torch.no_grad():
            for batch in _batched(rows, batch_size):
                logits = model(
                    encode(
                        [row.review_raw for row in batch],
                        vocab,
                        max_length,
                    )
                )
                predicted.extend(
                    ID_TO_LABEL[int(index)] for index in logits.argmax(1)
                )
        return predicted

    return LoadedEvaluator(
        key="text_cnn",
        display_name=MODEL_NAMES["text_cnn"],
        device="cpu",
        artifact_bytes=artifact_size(path),
        parameter_count=_parameter_count(model),
        training_seconds=training_seconds,
        training_config=config,
        provenance=_provenance(record),
        load_seconds=load_seconds,
        predict_batch=predict,
    )


def load_text_cnn_evaluator(
    artifact_dir: Path,
    *,
    require_verified: bool = True,
    batch_size: int = 64,
) -> LoadedEvaluator:
    """Load the optional TextCNN without widening the canonical runner."""
    if batch_size < 1:
        raise ValueError("Evaluation batch size must be at least 1")
    metrics_path = artifact_dir / "text_cnn_metrics.json"
    record = json.loads(metrics_path.read_text())
    _training_seconds(record, metrics_path, require_verified)
    if require_verified and "provenance" not in record:
        raise UnverifiedArtifactError(
            f"{metrics_path} has no shared training provenance. "
            "Regenerate it with the training runner."
        )
    return _load_text_cnn(
        artifact_dir,
        record,
        require_verified,
        batch_size,
    )


def _load_atae_lstm(
    artifact_dir: Path,
    record: dict[str, object],
    require_verified: bool,
    batch_size: int,
) -> LoadedEvaluator:
    path = artifact_dir / "atae_lstm.pt"
    started = perf_counter()
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    _assert_labels(artifact.get("labels"), path, require_verified)
    training_seconds = _training_seconds(record, path, require_verified)
    _assert_embedded_metadata(artifact, record, path, require_verified)
    vocab = artifact["vocab"]
    model = ATAELSTM(len(vocab))
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    load_seconds = perf_counter() - started

    def predict(rows: Sequence[AspectExample]) -> list[str]:
        predicted: list[str] = []
        with torch.no_grad():
            for batch in _batched(rows, batch_size):
                logits = model(
                    encode([row.review_raw for row in batch], vocab),
                    encode([row.aspect for row in batch], vocab, 12),
                )
                predicted.extend(ID_TO_LABEL[int(index)] for index in logits.argmax(1))
        return predicted

    return LoadedEvaluator(
        key="atae_lstm",
        display_name=MODEL_NAMES["atae_lstm"],
        device="cpu",
        artifact_bytes=artifact_size(path),
        parameter_count=_parameter_count(model),
        training_seconds=training_seconds,
        training_config=_config(record),
        provenance=_provenance(record),
        load_seconds=load_seconds,
        predict_batch=predict,
    )


def _load_distilbert(
    artifact_dir: Path,
    record: dict[str, object],
    require_verified: bool,
    batch_size: int,
    device: torch.device,
    require_explicit_labels: bool,
) -> LoadedEvaluator:
    path = artifact_dir / "distilbert"
    run_path = path / "training_run.json"
    started = perf_counter()
    if run_path.exists():
        metadata = json.loads(run_path.read_text())
    elif require_verified:
        raise UnverifiedArtifactError(
            f"{run_path} is missing. Regenerate DistilBERT with the #91 training protocol."
        )
    else:
        metadata = {}
    training_seconds = _training_seconds(record, run_path, require_verified)
    _assert_embedded_metadata(metadata, record, run_path, require_verified)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = ABSADistilBERT.from_pretrained(path, local_files_only=True).to(device)
    if require_explicit_labels:
        labels = [
            model.config.id2label[index]
            for index in range(model.config.num_labels)
        ]
        _assert_labels(labels, path / "config.json", require_verified)
    model.eval()
    load_seconds = perf_counter() - started

    def predict(rows: Sequence[AspectExample]) -> list[str]:
        loader = DataLoader(AspectPairDataset(tokenizer, rows), batch_size=batch_size)
        predicted: list[str] = []
        with torch.no_grad():
            for batch in loader:
                batch.pop("labels")
                features = {name: tensor.to(device) for name, tensor in batch.items()}
                predicted.extend(
                    ID_TO_LABEL[int(index)]
                    for index in model(**features).logits.argmax(1).cpu()
                )
        return predicted

    return LoadedEvaluator(
        key="distilbert",
        display_name=MODEL_NAMES["distilbert"],
        device=str(device),
        artifact_bytes=artifact_size(path),
        parameter_count=_parameter_count(model),
        training_seconds=training_seconds,
        training_config=_config(record),
        provenance=_provenance(record),
        load_seconds=load_seconds,
        predict_batch=predict,
    )


def load_artifact_evaluators(
    artifact_dir: Path,
    *,
    model_keys: Sequence[str] = CORE_MODEL_ORDER,
    require_verified: bool = True,
    recurrent_batch_size: int = 64,
    transformer_batch_size: int = 16,
    device: torch.device | None = None,
) -> list[LoadedEvaluator]:
    """Load a verified ordered subset while preserving the canonical default."""
    if recurrent_batch_size < 1 or transformer_batch_size < 1:
        raise ValueError("Evaluation batch sizes must be at least 1")
    keys = _normalise_model_keys(model_keys)
    records = _read_run_records(artifact_dir, require_verified, keys)
    loaded: list[LoadedEvaluator] = []
    for key in keys:
        if key == "tfidf":
            evaluator = _load_tfidf(
                artifact_dir,
                records[key],
                require_verified,
            )
        elif key == "target_lstm":
            evaluator = _load_target_lstm(
                artifact_dir,
                records[key],
                require_verified,
                recurrent_batch_size,
            )
        elif key == "target_gru":
            evaluator = _load_target_gru(
                artifact_dir,
                records[key],
                require_verified,
                recurrent_batch_size,
            )
        elif key == "text_cnn":
            evaluator = _load_text_cnn(
                artifact_dir,
                records[key],
                require_verified,
                recurrent_batch_size,
            )
        elif key == "atae_lstm":
            evaluator = _load_atae_lstm(
                artifact_dir,
                records[key],
                require_verified,
                recurrent_batch_size,
            )
        else:
            evaluator = _load_distilbert(
                artifact_dir,
                records[key],
                require_verified,
                transformer_batch_size,
                device or preferred_device(),
                require_explicit_labels=keys == SIX_MODEL_ORDER,
            )
        loaded.append(evaluator)
    return loaded
