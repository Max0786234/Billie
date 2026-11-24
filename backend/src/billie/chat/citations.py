"""Citation repair: link uncited factual sentences to the source they came from.

Measured problem (``evals/answer_quality.py``): the 4B local model leaves ~45% of its
factual sentences uncited, and stricter prompt wording made it *worse* (a measured negative,
see DEVLOG P2.8). So citations are repaired after generation, without relying on the model:

* split the answer into sentences, keeping Markdown structure intact;
* for every factual-looking sentence without a citation, compare its embedding with the
  embeddings of the sources shown to the model;
* if the best match clears a calibrated threshold, insert that citation, recorded as
  *automatic* so the UI and logs stay transparent;
* otherwise count the sentence as *unverified*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from billie.chat.prompts import CITATION
from billie.retrieval.embeddings import Embedder

_SENTENCE_END = re.compile(r"(?<=[.!?])[ \t]+|\s*\n\s*")  # sentence ends and line breaks (list items)
# Sentences that are not factual claims: refusals, meta statements about the sources, advice.
_NOT_A_CLAIM = re.compile(
    r"could not find|couldn't find|not (mentioned|covered|included|available|provided|specified|addressed)|"
    r"(do|does|did)(n't| not) (mention|include|cover|address|provide|state|specify|confirm|discuss|say)|"
    r"\bno (source|document|information)\b|the (sources|documents) (only|do not|don't)|"
    r"consult|healthcare professional|doctor|rephrase|please|let me know|sorry|i'm not able|emergency",
    re.IGNORECASE,
)


def is_factual_sentence(text: str) -> bool:
    text = text.strip(" -*•\t")
    return len(text) > 40 and not _NOT_A_CLAIM.search(text)


@dataclass
class RepairResult:
    answer: str
    added: list[dict[str, object]] = field(default_factory=list)  # {"sentence", "n", "score"}
    unverified: list[str] = field(default_factory=list)


def _segments(answer: str) -> list[tuple[str, str]]:
    """(sentence, trailing whitespace) pieces whose concatenation is exactly ``answer``."""
    pieces: list[tuple[str, str]] = []
    position = 0
    for match in _SENTENCE_END.finditer(answer):
        pieces.append((answer[position : match.start()], match.group(0)))
        position = match.end()
    pieces.append((answer[position:], ""))
    return pieces


def _insert_citation(sentence: str, n: int) -> str:
    """'Text.' -> 'Text [2].' (citation before the final punctuation, as the model writes them)."""
    stripped = sentence.rstrip()
    if stripped and stripped[-1] in ".!?":
        return f"{stripped[:-1]} [{n}]{stripped[-1]}"
    return f"{stripped} [{n}]"


def repair_citations(
    answer: str,
    sources: list[str],
    embedder: Embedder,
    threshold: float,
    source_vectors: np.ndarray | None = None,
) -> RepairResult:
    if not sources or not answer.strip():
        return RepairResult(answer)
    pieces = _segments(answer)
    candidates = [
        i for i, (sentence, _) in enumerate(pieces) if not CITATION.search(sentence) and is_factual_sentence(sentence)
    ]
    if not candidates:
        return RepairResult(answer)
    sentence_vecs = embedder.embed_queries([pieces[i][0] for i in candidates])
    # Usually precomputed while the model was writing (see ChatPipeline); same vectors either way.
    source_vecs = source_vectors if source_vectors is not None else embedder.embed_documents(sources)
    scores = np.asarray(sentence_vecs) @ np.asarray(source_vecs).T  # cosine (unit vectors)
    result = RepairResult(answer)
    for row, i in enumerate(candidates):
        best = int(np.argmax(scores[row]))
        score = float(scores[row, best])
        sentence, gap = pieces[i]
        if score >= threshold:
            pieces[i] = (_insert_citation(sentence, best + 1), gap)
            result.added.append({"sentence": sentence.strip(), "n": best + 1, "score": round(score, 3)})
        else:
            result.unverified.append(sentence.strip())
    result.answer = "".join(sentence + gap for sentence, gap in pieces)
    return result
