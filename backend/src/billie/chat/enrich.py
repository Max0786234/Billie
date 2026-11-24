"""Mode-specific extras shown next to an answer (cheap, computed without the LLM)."""

from __future__ import annotations

from typing import Any

from billie.nlp.concepts import extract_concepts
from billie.nlp.medical_entities import MedicalEntity, extract_entities
from billie.retrieval.base import SearchHit


def enrich(mode: str, question: str, hits: list[SearchHit]) -> dict[str, Any]:
    if mode == "medical":
        entities = extract_entities(question)
        focus = hits[0].passage.metadata.get("focus") if hits else ""
        if focus:
            entities.append(MedicalEntity(text=focus, category="topic"))
        unique: dict[str, dict[str, str]] = {}
        for entity in entities:
            unique.setdefault(entity.text.casefold(), {"text": entity.text, "category": entity.category})
        return {"entities": list(unique.values()), "disclaimer": "Educational information only - not medical advice."}
    if mode == "research" and hits:
        text = " ".join(f"{h.passage.title}. {h.passage.text}" for h in hits)
        return {"concepts": [{"text": c.text, "score": c.score} for c in extract_concepts(text)]}
    return {}
