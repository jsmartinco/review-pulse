"""Artifact-backed predictor adapters for the ReviewPulse v3 contract."""

from pathlib import Path

import joblib
import torch
from transformers import AutoTokenizer

from ..config import ABSA_OUTPUTS_DIR
from ..interpretability.attention import EVIDENCE_CAVEAT, align_attention
from ..labels import ID_TO_LABEL
from ..models.atae_lstm import ATAELSTM
from ..models.distilbert import ABSADistilBERT
from ..models.target_lstm import TargetAgnosticLSTM
from ..tokenization.sequence import encode
from ..tokenization.transformer import encode_review_aspect_pairs


MODEL_OPTIONS = {
    "absa_tfidf": "TF-IDF review-only",
    "absa_target_lstm": "LSTM review-only",
    "absa_atae_lstm": "ATAE-LSTM aspect-conditioned",
    "absa_distilbert": "DistilBERT sentence-pair",
}


def _payload(aspect: str, logits: torch.Tensor, model_name: str, token_evidence=None) -> dict:
    probabilities = torch.softmax(logits, dim=-1)[0]
    index = int(probabilities.argmax())
    return {
        "aspect": aspect,
        "label": ID_TO_LABEL[index],
        "confidence": float(probabilities[index]),
        "model": model_name,
        "token_evidence": token_evidence,
    }


class TfidfAspectPredictor:
    def __init__(self, path: Path = ABSA_OUTPUTS_DIR / "tfidf_baseline.joblib") -> None:
        self.model = joblib.load(path)

    def predict(self, review: str, aspect: str, model_name: str) -> dict:
        probabilities = self.model.predict_proba([review])[0]
        labels = list(self.model.classes_)
        index = probabilities.argmax()
        return {"aspect": aspect, "label": str(labels[index]), "confidence": float(probabilities[index]), "model": model_name, "token_evidence": None}


class TargetLstmAspectPredictor:
    def __init__(self, path: Path = ABSA_OUTPUTS_DIR / "target_lstm.pt") -> None:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        self.vocab = artifact["vocab"]
        self.model = TargetAgnosticLSTM(len(self.vocab))
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()

    def predict(self, review: str, aspect: str, model_name: str) -> dict:
        with torch.no_grad():
            return _payload(aspect, self.model(encode([review], self.vocab)), model_name)


class AtaeLstmAspectPredictor:
    def __init__(self, path: Path = ABSA_OUTPUTS_DIR / "atae_lstm.pt") -> None:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        self.vocab = artifact["vocab"]
        self.model = ATAELSTM(len(self.vocab))
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()

    def predict(self, review: str, aspect: str, model_name: str) -> dict:
        with torch.no_grad():
            logits, weights = self.model(encode([review], self.vocab), encode([aspect], self.vocab, 12), return_attention=True)
        return _payload(aspect, logits, model_name, {"tokens": align_attention(review, weights[0]), "caveat": EVIDENCE_CAVEAT})


class DistilBertAspectPredictor:
    def __init__(self, path: Path = ABSA_OUTPUTS_DIR / "distilbert") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = ABSADistilBERT.from_pretrained(path, local_files_only=True)
        self.model.eval()

    def predict(self, review: str, aspect: str, model_name: str) -> dict:
        encoded = encode_review_aspect_pairs(self.tokenizer, [review], [aspect])
        with torch.no_grad():
            return _payload(aspect, self.model(**encoded).logits, model_name)


def get_predictor(model_name: str):
    """Load an explicitly supported predictor, or fail with a controlled message."""
    predictors = {
        "absa_tfidf": TfidfAspectPredictor,
        "absa_target_lstm": TargetLstmAspectPredictor,
        "absa_atae_lstm": AtaeLstmAspectPredictor,
        "absa_distilbert": DistilBertAspectPredictor,
    }
    try:
        return predictors[model_name]()
    except KeyError as error:
        raise ValueError(f"Unknown v3 model: {model_name}") from error
