from __future__ import annotations

import numpy as np

from billie.chat.citations import is_factual_sentence, repair_citations


class TopicEmbedder:
    """Embeds by topic keywords so similarity is predictable in tests."""

    model = "topic"
    TOPICS = ("insulin", "glaucoma", "asthma")

    def _vec(self, text: str) -> list[float]:
        v = [float(text.lower().count(t)) for t in self.TOPICS] + [0.1]
        n = float(np.linalg.norm(v))
        return [x / n for x in v]

    def embed_queries(self, texts):
        return np.array([self._vec(t) for t in texts], dtype=np.float32)

    def embed_documents(self, texts, titles=None):
        return self.embed_queries(texts)


SOURCES = ["Insulin therapy lowers blood sugar; insulin is injected.", "Glaucoma damages the optic nerve."]


def test_uncited_factual_sentence_gets_the_matching_source():
    answer = "Type 1 diabetes is treated with insulin injections every day. Glaucoma harms the optic nerve [2]."
    result = repair_citations(answer, SOURCES, TopicEmbedder(), threshold=0.8)
    assert result.answer == (
        "Type 1 diabetes is treated with insulin injections every day [1]. Glaucoma harms the optic nerve [2]."
    )
    assert result.added == [
        {
            "sentence": "Type 1 diabetes is treated with insulin injections every day.",
            "n": 1,
            "score": result.added[0]["score"],
        }
    ]
    assert result.unverified == []


def test_unmatched_sentence_is_reported_as_unverified_not_cited():
    answer = "Asthma inhalers relax the airways within a few minutes of use."
    result = repair_citations(answer, SOURCES, TopicEmbedder(), threshold=0.8)
    assert result.answer == answer and result.added == []
    assert result.unverified == [answer]


def test_refusals_advice_and_markdown_layout_are_left_alone():
    answer = (
        "- **Insulin** is the main treatment for type 1 diabetes in children\n"
        "I could not find information about diet plans in the documents.\n\n"
        "Please consult a healthcare professional for personal advice."
    )
    result = repair_citations(answer, SOURCES, TopicEmbedder(), threshold=0.8)
    assert result.answer.startswith("- **Insulin** is the main treatment for type 1 diabetes in children [1]")
    assert "could not find information about diet plans in the documents.\n\nPlease consult" in result.answer
    assert len(result.added) == 1


def test_is_factual_sentence():
    assert is_factual_sentence("Glaucoma is a group of eye diseases that damage the optic nerve.")
    assert not is_factual_sentence("The sources do not provide information about this topic at all.")
    assert not is_factual_sentence("Short line.")


def test_pipeline_repairs_citations_and_reports_them(settings):
    from billie.api.services import build_services
    from billie.chat.pipeline import ChatTurn
    from tests.conftest import FakeLLMManager, FakeProvider

    class KeywordEmbedder(TopicEmbedder):
        TOPICS = ("phishing", "refresh", "upload")

    services = build_services(
        settings, llm=FakeLLMManager(FakeProvider("Phishing emails should be reported to the security team address."))
    )
    services.corpora.embedder = KeywordEmbedder()
    services.corpora.prepare()
    services.pipeline.citation_threshold = 0.8
    done = next(
        e["data"]
        for e in services.pipeline.run(ChatTurn("Where are phishing emails reported?"))
        if e["event"] == "done"
    )
    assert done["answer"].endswith("address [1].")
    assert done["auto_citations"][0]["n"] == 1 and done["grounded"] is True
    assert done["unverified_sentences"] == 0
    _, messages = services.conversations.get(done["conversation_id"])
    assert messages[1].content == done["answer"]  # the repaired answer is what gets stored


def test_refusal_openers_count_as_abstained_even_with_citations():
    from billie.chat.pipeline import opens_with_refusal

    # Real answers from the Groq stress test: refusals that cite what the sources *do* cover.
    assert opens_with_refusal("I could not find a Python function in the documents. They only cover Billie [1].")
    assert opens_with_refusal("I cannot provide instructions for that. The sources only describe its effects [1].")
    assert opens_with_refusal("The provided sources do not state who invented the transformer [1].")
    assert opens_with_refusal("No encontré esa información en los documentos.")
    # Real answers that merely contain a negative later on are not refusals.
    assert not opens_with_refusal("Glaucoma damages the optic nerve [1]. It cannot be reversed [2].")
    assert not opens_with_refusal("Metformin is first-line therapy [1]; the sources do not mention dosing.")
