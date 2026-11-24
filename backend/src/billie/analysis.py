"""Message analysis: language and sentiment, behind one small interface.

* Language: script rules + romanised-Hindi detection + py3langid (``billie.nlp.language``).
* Sentiment: a softmax head trained on multilingual embeddings (``billie.nlp.sentiment``),
  falling back to a word list when the embedding model or the trained head is missing.

The chat pipeline only depends on :class:`MessageAnalyzer`, so implementations are swappable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, cast

import numpy as np

from billie.llm.base import LLMError
from billie.nlp.language import detect_language
from billie.nlp.lexicon_sentiment import lexicon_sentiment
from billie.nlp.sentiment import SentimentHead, model_path
from billie.retrieval.embeddings import Embedder

Sentiment = Literal["positive", "neutral", "negative"]


@dataclass(frozen=True)
class Analysis:
    language: str  # ISO 639-1 code, e.g. "en", "hi"
    language_name: str  # name used in the "Reply in ..." instruction
    language_confidence: float
    sentiment: Sentiment
    sentiment_score: float  # confidence of the predicted label, 0..1
    method: dict[str, str] = field(default_factory=dict)  # which technique produced each field


class MessageAnalyzer(Protocol):
    def analyze(self, text: str, embedding: np.ndarray | None = None) -> Analysis: ...


class ModelAnalyzer:
    """Each field degrades independently: no embedder or no trained head -> lexicon sentiment."""

    def __init__(self, embedder: Embedder | None = None, head: SentimentHead | None = None) -> None:
        self.embedder = embedder
        self.head = head

    @classmethod
    def from_embedder(cls, embedder: Embedder | None) -> ModelAnalyzer:
        head = None
        if embedder is not None:
            path = model_path(embedder.model)
            if path.exists():
                head = SentimentHead.load(path)
        return cls(embedder, head)

    def analyze(self, text: str, embedding: np.ndarray | None = None) -> Analysis:
        language = detect_language(text)
        label, score, method = self._sentiment(text, embedding)
        return Analysis(
            language=language.code,
            language_name=language.reply_language,
            language_confidence=language.confidence,
            sentiment=label,
            sentiment_score=score,
            method={"language": language.method, "sentiment": method},
        )

    def _sentiment(self, text: str, embedding: np.ndarray | None) -> tuple[Sentiment, float, str]:
        if self.head is not None and self.embedder is not None:
            try:
                vector = embedding if embedding is not None else self.embedder.embed_queries([text])[0]
                label, prob = self.head.predict(vector)
                # Domain shift: trained on tweets, the probe reads neutral medical questions ("What is
                # glaucoma?") as weakly negative (0.38-0.45), which triggered needless "don't worry"
                # openers (stress test). Require confidence or an explicit emotion word for negative.
                if label == "negative" and prob < 0.6 and lexicon_sentiment(text).label != "negative":
                    return "neutral", prob, "embedding-probe+lexicon-check"
                return cast(Sentiment, label), prob, "embedding-probe"
            except LLMError:
                pass
        result = lexicon_sentiment(text)
        return cast(Sentiment, result.label), result.score, "lexicon"
