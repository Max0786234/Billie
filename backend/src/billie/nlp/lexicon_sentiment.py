"""Word-list sentiment: the offline fallback when no embedding model is available.

Counts positive/negative cue words with simple negation handling ("not good").
The trained embedding probe (``billie.nlp.sentiment``) replaces it whenever possible;
``evals/train_sentiment.py`` reports both on the same test set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

POSITIVE = frozenset(
    "awesome excellent good great happy helpful love loved perfect pleased satisfied thanks thank "
    "wonderful amazing appreciate brilliant fantastic glad nice".split()
)
NEGATIVE = frozenset(
    "angry annoyed awful bad broken confused disappointed frustrated frustrating hate issue mad problem "
    "terrible upset worried worse worst wrong useless scared anxious afraid pain painful stuck".split()
)
NEGATIONS = frozenset(
    ["not", "never", "no", "cannot", "can't", "dont", "don't", "isnt", "isn't", "wasnt", "wasn't", "didnt", "didn't"]
)
_TOKEN = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class LexiconResult:
    label: str  # positive / neutral / negative
    score: float


def lexicon_sentiment(text: str) -> LexiconResult:
    tokens = _TOKEN.findall(text.lower())
    score = 0.0
    for i, token in enumerate(tokens):
        polarity = 1 if token in POSITIVE else -1 if token in NEGATIVE else 0
        if polarity and any(t in NEGATIONS for t in tokens[max(0, i - 3) : i]):
            polarity = -polarity
        score += polarity
    score += 0.25 * min(2, text.count("!")) * (1 if score > 0 else -1 if score < 0 else 0)
    if score >= 1:
        return LexiconResult("positive", min(1.0, 0.55 + 0.15 * score))
    if score <= -1:
        return LexiconResult("negative", min(1.0, 0.55 + 0.15 * -score))
    return LexiconResult("neutral", 0.5)
