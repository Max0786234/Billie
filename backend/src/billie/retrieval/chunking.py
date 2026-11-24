"""Sentence-aware chunking.

Splits on paragraph and sentence boundaries and packs sentences into chunks of
at most ``max_chars``, repeating the last sentence(s) as overlap. Chunks never
cut words in half, which keeps citations readable.
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")  # includes the Devanagari danda


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if paragraph:
            sentences.extend(s for s in _SENTENCE_END.split(paragraph) if s)
    return sentences


def chunk_text(text: str, max_chars: int = 900, overlap_chars: int = 150) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        # A single sentence longer than the limit is hard-wrapped on whitespace.
        if len(sentence) > max_chars:
            if current:
                chunks.append(" ".join(current))
                current, size = [], 0
            chunks.extend(_wrap(sentence, max_chars))
            continue
        if size + len(sentence) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            # Carry trailing sentences forward as overlap.
            carry: list[str] = []
            carried = 0
            for prev in reversed(current):
                if carried + len(prev) > overlap_chars:
                    break
                carry.insert(0, prev)
                carried += len(prev) + 1
            current, size = carry, carried
        current.append(sentence)
        size += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def _wrap(text: str, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    if line:
        lines.append(line)
    return lines
