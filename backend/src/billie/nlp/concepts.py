"""Key-concept extraction for Research mode (shown as chips under an answer).

RAKE-style (Rose et al., 2010): text is split into candidate phrases at stopwords
and punctuation; phrases are scored by frequency, with a bonus for multi-word
phrases ending in a typical CS head noun ("graph neural network", "retrieval
model"). Deterministic and instant; the LLM writes the actual answer.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_HEADS = frozenset(
    "model models network networks learning attention retrieval transformer transformers embedding "
    "embeddings optimization classification generation inference reasoning algorithm algorithms dataset "
    "datasets graph graphs vision detection segmentation translation agent agents benchmark encoder "
    "decoder search prediction representation representations framework".split()
)
# Function words plus verbs and filler that are common in abstracts but never concepts.
_STOP = frozenset(
    "a an and are as at be been being but by can could do does for from had has have how i if in into is "
    "it its of on or our over so such than that the their them then there these they this those to under "
    "up via was we were what when where which while who will with within without would also both each "
    "more most other some very about across after against among based between different during existing "
    "further however including new novel number paper several significant state various approach "
    "approaches method methods problem propose proposed proposes present presents provide provides show "
    "shows shown achieve achieves achieving demonstrate demonstrates experiments results result "
    "performance improve improves improved outperform outperforms outperforming use uses using used "
    "recent first second study work works large small high low well better best two three one many often "
    "existing make makes made allow allows enable enables".split()
)
_SPLIT = re.compile(r"[^A-Za-z0-9\-\s]+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")


@dataclass(frozen=True)
class Concept:
    text: str
    score: float


def candidate_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for fragment in _SPLIT.split(text):
        run: list[str] = []
        for word in _WORD.findall(fragment):
            lower = word.lower()
            if lower in _STOP or len(lower) < 3:
                if run:
                    phrases.append(" ".join(run))
                run = []
            else:
                run.append(lower)
                if len(run) == 4:
                    phrases.append(" ".join(run))
                    run = []
        if run:
            phrases.append(" ".join(run))
    return phrases


def extract_concepts(text: str, top_k: int = 8) -> list[Concept]:
    counts = Counter(candidate_phrases(text))
    scores: dict[str, float] = {}
    for phrase, freq in counts.items():
        words = phrase.split()
        weight = 1.0 if len(words) == 1 else 2.0
        if len(words) >= 2 and words[-1] in _HEADS:
            weight = 3.0
        scores[phrase] = freq * weight
    # A single word that only ever appears inside a selected phrase adds nothing.
    multi = [p for p in scores if " " in p]
    for word in [p for p in scores if " " not in p]:
        if any(word in p.split() for p in multi):
            scores[word] *= 0.25
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    top = ranked[0][1]
    return [Concept(phrase, round(score / top, 3)) for phrase, score in ranked]
