from __future__ import annotations

import numpy as np

from billie.analysis import ModelAnalyzer
from billie.llm.base import LLMError
from billie.nlp.concepts import extract_concepts
from billie.nlp.lexicon_sentiment import lexicon_sentiment
from billie.nlp.medical_entities import extract_entities
from billie.nlp.sentiment import SentimentHead, train_softmax


def test_medical_entities_prefer_longest_term_and_categorise():
    entities = {(e.text, e.category) for e in extract_entities("I have chest pain and a fever after chemotherapy")}
    assert ("Chest pain", "symptom") in entities
    assert ("Pain", "symptom") not in entities  # covered by the longer term
    assert ("Fever", "symptom") in entities and ("Chemotherapy", "treatment") in entities


def test_concepts_rank_technical_phrases_above_single_words():
    text = (
        "We propose a graph neural network for molecule property prediction. The graph neural network "
        "outperforms transformer model baselines on molecular datasets."
    )
    concepts = [c.text for c in extract_concepts(text)]
    assert concepts[0] == "graph neural network"
    assert "graph" not in concepts and "we a graph" not in concepts


def test_lexicon_sentiment_handles_negation():
    assert lexicon_sentiment("This is great, thanks!").label == "positive"
    assert lexicon_sentiment("This is not good at all").label == "negative"
    assert lexicon_sentiment("I am worried about my results").label == "negative"
    assert lexicon_sentiment("What is glaucoma?").label == "neutral"


def test_softmax_head_learns_a_separable_problem_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    centers = np.eye(3, 8)
    x = np.vstack([c + 0.05 * rng.standard_normal((40, 8)) for c in centers]).astype(np.float32)
    y = np.repeat(np.arange(3), 40)
    weights, bias = train_softmax(x, y, epochs=300)
    head = SentimentHead(weights, bias, "fake")
    assert (head.predict_proba(x).argmax(axis=1) == y).mean() > 0.95
    head.save(tmp_path / "h.npz")
    loaded = SentimentHead.load(tmp_path / "h.npz")
    assert loaded.embedding_model == "fake" and loaded.predict(centers[2])[0] == "positive"


class _Embedder:
    model = "fake"

    def __init__(self, fail=False):
        self.fail = fail

    def embed_queries(self, texts):
        if self.fail:
            raise LLMError("down")
        return np.tile(np.eye(3, 8)[0], (len(texts), 1)).astype(np.float32)

    def embed_documents(self, texts, titles=None):
        return self.embed_queries(texts)


def test_model_analyzer_uses_probe_and_falls_back_to_lexicon():
    head = SentimentHead(np.eye(3, 8, dtype=np.float32) * 5, np.zeros(3, dtype=np.float32), "fake")
    probe = ModelAnalyzer(_Embedder(), head).analyze("anything")
    assert (probe.sentiment, probe.method["sentiment"]) == ("negative", "embedding-probe")
    fallback = ModelAnalyzer(_Embedder(fail=True), head).analyze("This is wonderful, thank you!")
    assert (fallback.sentiment, fallback.method["sentiment"]) == ("positive", "lexicon")
    hindi = ModelAnalyzer(None).analyze("मुझे बुखार है")
    assert hindi.language == "hi" and hindi.language_name == "Hindi"
