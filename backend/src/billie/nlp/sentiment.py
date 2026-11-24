"""Sentiment classification with a linear probe on sentence embeddings.

A multinomial logistic-regression head (3 classes) trained on top of the same
multilingual embedding model used for retrieval. Because the embedding space is
shared across languages, a head trained on English/Spanish/Hindi/... tweets also
transfers to other languages. Training: ``python -m evals.train_sentiment``.

Weights live in ``billie/nlp/models/sentiment-<embedding model>.npz`` and are
tied to the embedding model they were trained with.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

LABELS = ("negative", "neutral", "positive")
MODELS_DIR = Path(__file__).parent / "models"


def model_path(embedding_model: str) -> Path:
    safe = embedding_model.replace(":", "_").replace("/", "_")
    return MODELS_DIR / f"sentiment-{safe}.npz"


@dataclass
class SentimentHead:
    weights: np.ndarray  # (classes, dim)
    bias: np.ndarray  # (classes,)
    embedding_model: str

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        return softmax(np.atleast_2d(embeddings) @ self.weights.T + self.bias)

    def predict(self, embedding: np.ndarray) -> tuple[str, float]:
        probs = self.predict_proba(embedding)[0]
        index = int(np.argmax(probs))
        return LABELS[index], float(probs[index])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weights=self.weights, bias=self.bias, embedding_model=np.array(self.embedding_model))

    @classmethod
    def load(cls, path: Path) -> SentimentHead:
        with np.load(path) as data:
            return cls(data["weights"], data["bias"], str(data["embedding_model"]))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train_softmax(
    x: np.ndarray,
    y: np.ndarray,
    *,
    classes: int = 3,
    l2: float = 1e-3,
    lr: float = 0.5,
    epochs: int = 400,
    class_weighted: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-batch gradient descent on L2-regularised cross-entropy (inputs are unit vectors)."""
    n, dim = x.shape
    weights = np.zeros((classes, dim), dtype=np.float64)
    bias = np.zeros(classes, dtype=np.float64)
    onehot = np.eye(classes)[y]
    if class_weighted:
        counts = np.bincount(y, minlength=classes).astype(np.float64)
        sample_w = (n / (classes * np.maximum(counts, 1)))[y]
    else:
        sample_w = np.ones(n)
    for _ in range(epochs):
        probs = softmax(x @ weights.T + bias)
        grad_logits = (probs - onehot) * sample_w[:, None] / n
        weights -= lr * (grad_logits.T @ x + l2 * weights)
        bias -= lr * grad_logits.sum(axis=0)
    return weights.astype(np.float32), bias.astype(np.float32)
