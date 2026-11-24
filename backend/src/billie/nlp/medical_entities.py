"""Lightweight medical entity tagging for the Medical-mode UI chips.

A small curated vocabulary (symptoms, treatments, disease words) matched on word
boundaries, plus the MedQuAD "focus" of the top retrieved answer. It is a UI aid,
not clinical NER; the LLM never relies on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SYMPTOMS = frozenset(
    {
        "ache", "bleeding", "blurred vision", "bruising", "chest pain", "chills", "constipation", "cough",
        "diarrhea", "dizziness", "fatigue", "fever", "headache", "itching", "nausea", "numbness", "pain",
        "rash", "shortness of breath", "sore throat", "sweating", "swelling", "thirst", "tiredness",
        "vomiting", "weakness", "weight loss", "wheezing",
    }
)  # fmt: skip
TREATMENTS = frozenset(
    {
        "antibiotic", "antibiotics", "chemotherapy", "dialysis", "immunotherapy", "inhaler", "insulin",
        "medication", "medicine", "metformin", "physical therapy", "radiation therapy", "stem cell transplant",
        "surgery", "targeted therapy", "therapy", "transplant", "vaccine",
    }
)  # fmt: skip
DISEASES = frozenset(
    {
        "anemia", "arthritis", "asthma", "cancer", "covid", "dementia", "depression", "diabetes", "disease",
        "disorder", "glaucoma", "hepatitis", "hypertension", "infection", "influenza", "leukemia", "malaria",
        "migraine", "obesity", "pneumonia", "stroke", "syndrome", "tuberculosis",
    }
)  # fmt: skip

_LEXICON = (
    [(term, "symptom") for term in SYMPTOMS]
    + [(t, "treatment") for t in TREATMENTS]
    + [(t, "disease") for t in DISEASES]
)
# Longest terms first so "chest pain" wins over "pain".
_PATTERNS = [
    (re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE), term, category)
    for term, category in sorted(_LEXICON, key=lambda x: -len(x[0]))
]


@dataclass(frozen=True)
class MedicalEntity:
    text: str
    category: str


def extract_entities(text: str) -> list[MedicalEntity]:
    found: list[MedicalEntity] = []
    covered: list[tuple[int, int]] = []
    for pattern, term, category in _PATTERNS:
        for match in pattern.finditer(text):
            span = match.span()
            if any(start <= span[0] < end for start, end in covered):
                continue  # already inside a longer term
            covered.append(span)
            found.append(MedicalEntity(term.capitalize(), category))
            break
    return found
