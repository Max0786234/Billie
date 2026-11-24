"""Strip ``<think>...</think>`` reasoning blocks from a token stream.

Some local reasoning models (Qwen3, DeepSeek-R1 distills) emit their chain of
thought inline. Tags can be split across stream chunks, so the filter buffers
just enough text to recognise a tag boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

OPEN, CLOSE = "<think>", "</think>"


def strip_think(chunks: Iterable[str]) -> Iterator[str]:
    buffer = ""
    inside = False
    for chunk in chunks:
        buffer += chunk
        while True:
            tag = CLOSE if inside else OPEN
            idx = buffer.find(tag)
            if idx != -1:
                if not inside and idx:
                    yield buffer[:idx]
                buffer = buffer[idx + len(tag) :]
                inside = not inside
                continue
            # No full tag: emit everything except a possible partial tag suffix.
            keep = _partial_suffix(buffer, tag)
            if not inside and len(buffer) > keep:
                yield buffer[: len(buffer) - keep]
            buffer = buffer[len(buffer) - keep :] if keep else ""
            break
    if buffer and not inside:
        yield buffer


def _partial_suffix(text: str, tag: str) -> int:
    """Length of the longest suffix of ``text`` that is a prefix of ``tag``."""
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-size:]):
            return size
    return 0
