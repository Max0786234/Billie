"""Text embeddings for dense retrieval, served by a local Ollama model.

Kept separate from the chat provider on purpose: a cloud chat model can be used
while search stays local, free and private.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from billie.llm.base import LLMError
from billie.llm.ollama import OllamaProvider

# Task prompts recommended for EmbeddingGemma; harmless for other models.
_PROMPTS = {
    "embeddinggemma": ("task: search result | query: ", "title: {title} | text: "),
}


class Embedder(Protocol):
    model: str

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_documents(self, texts: Sequence[str], titles: Sequence[str] | None = None) -> np.ndarray: ...


class OllamaEmbedder:
    def __init__(self, model: str = "embeddinggemma", base_url: str = "http://127.0.0.1:11434", batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size
        self._provider = OllamaProvider(model, base_url, timeout=120.0)
        family = model.split(":")[0]
        self._query_prefix, self._doc_template = _PROMPTS.get(family, ("", ""))
        self._query_cache: dict[str, np.ndarray] = {}

    def available(self) -> bool:
        try:
            return any(name.split(":")[0] == self.model.split(":")[0] for name in self._provider.list_models())
        except LLMError:
            return False

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        # The same question is embedded by sentiment analysis and by retrieval in one turn;
        # a small LRU cache computes it once. Identical text -> identical vector, so results don't change.
        missing = [t for t in dict.fromkeys(texts) if t not in self._query_cache]
        if missing:
            for text, vector in zip(missing, self._embed([self._query_prefix + t for t in missing]), strict=True):
                self._query_cache[text] = vector
                if len(self._query_cache) > 512:
                    self._query_cache.pop(next(iter(self._query_cache)))
        return np.stack([self._query_cache[t] for t in texts]) if texts else np.zeros((0, 0), dtype=np.float32)

    def embed_documents(
        self,
        texts: Sequence[str],
        titles: Sequence[str] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        titles = titles or ["none"] * len(texts)
        prepared = [
            self._doc_template.format(title=title or "none") + text for title, text in zip(titles, texts, strict=True)
        ]
        return self._embed(prepared, progress)

    def _embed(self, texts: list[str], progress: Callable[[int, int], None] | None = None) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._provider.embed(texts[start : start + self.batch_size], self.model))
            if progress:
                progress(min(start + self.batch_size, len(texts)), len(texts))
        return normalize(np.asarray(vectors, dtype=np.float32))


def normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.reshape(0, 0).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)
