"""Does spelling out medical abbreviations improve Medical-mode search? (DEVLOG P2.21)

Result: no (19/20 -> 19/20 on abbreviation questions; LiveQA nDCG 0.408 -> 0.275 on the 8 real
questions that trigger it), so the step was NOT added to the pipeline. The helpers live here.

    python -m evals.abbreviations_eval        # local model for the expansion call

1. 20 abbreviation questions with a known MedQuAD topic ("what is the VDRL test" -> syphilis):
   success = a top-4 passage title or opening names the topic, with and without expansion.
2. Side effects on real consumer questions: the TREC LiveQA questions that trigger the
   abbreviation detector are scored (nDCG@10, human judgments) with and without expansion.
"""

from __future__ import annotations

import json
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from billie.llm.base import GenerationOptions, Message, complete
from billie.paths import DATA_DIR, EVAL_DATASETS_DIR, EVAL_RESULTS_DIR
from billie.retrieval import HybridRetriever, PassageIndex
from billie.retrieval.embeddings import OllamaEmbedder
from billie.runtime import load_settings
from evals.common import eval_llm
from evals.retrieval_liveqa import load_qrels, load_questions, ranked_doc_ids, score

RESULTS = EVAL_RESULTS_DIR / "retrieval"

EXPAND_ABBREVIATIONS_PROMPT = (
    "Rewrite the user's medical question with every abbreviation or acronym spelled out in full, keeping the "
    "abbreviation in brackets, e.g. 'What is COPD?' -> 'What is chronic obstructive pulmonary disease (COPD)?'. "
    "If a test is named, add the condition it tests for, e.g. 'VDRL' -> 'Venereal Disease Research Laboratory "
    "(VDRL) test for syphilis'. Keep everything else unchanged. Reply with only the rewritten question."
)

_ABBREVIATION = re.compile(r"\b[A-Z][A-Z0-9]{1,5}s?\b")
_NOT_ABBREVIATIONS = frozenset("I OK US USA UK AM PM TV ID FAQ PDF AI".split())


def has_abbreviation(question: str) -> bool:
    """An all-caps token such as VDRL, COPD or HbA1c-style codes (not a shouted sentence)."""
    tokens = _ABBREVIATION.findall(question)
    if not tokens or sum(1 for w in question.split() if w.isupper()) > max(3, len(question.split()) // 2):
        return False  # shouting, not abbreviations
    return any(t.rstrip("s") not in _NOT_ABBREVIATIONS for t in tokens)


def expand_abbreviations(provider: Any, question: str) -> str | None:
    """The question with abbreviations spelled out, or None if nothing changed."""
    text = complete(
        provider,
        [Message("system", EXPAND_ABBREVIATIONS_PROMPT), Message("user", question)],
        GenerationOptions(temperature=0.0, max_tokens=80),
    )
    text = " ".join(text.split())[:300]
    return text if text and text.lower() != question.strip().lower() else None


def hit(retriever: HybridRetriever, query: str, gold: list[str]) -> tuple[bool, list[str]]:
    hits = retriever.search(query, top_k=4).hits
    titles = [h.passage.title for h in hits]
    found = any(g.lower() in f"{h.passage.title} {h.passage.text[:200]}".lower() for h in hits for g in gold)
    return found, titles


def main() -> int:
    base = load_settings()
    provider = eval_llm(Path(tempfile.mkdtemp()), False, base.paths.llm_settings).provider()
    index = PassageIndex(DATA_DIR / "index", "medquad")
    embedder = OllamaEmbedder("embeddinggemma")
    retriever = HybridRetriever(index, embedder)

    cases = json.loads((EVAL_DATASETS_DIR / "abbreviations.json").read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        expanded = expand_abbreviations(provider, case["q"])
        before, titles_before = hit(retriever, case["q"], case["gold"])
        after, titles_after = hit(retriever, f"{case['q']} {expanded}" if expanded else case["q"], case["gold"])
        rows.append(
            {
                **case,
                "expanded": expanded,
                "before": before,
                "after": after,
                "titles_before": titles_before,
                "titles_after": titles_after,
            }
        )
        print(
            f"{'+' if after and not before else '-' if before and not after else '='} {case['q']:32} "
            f"{before!s:5} -> {after!s:5}  | {expanded}",
            flush=True,
        )

    with sqlite3.connect(index.db_path) as conn:
        indexed = {r[0] for r in conn.execute("SELECT DISTINCT doc_id FROM passages")}
    questions, qrels = load_questions(), load_qrels(indexed)
    eval_ret = HybridRetriever(index, embedder, mode="hybrid", min_dense_score=0.0)
    liveqa = []
    for qid in sorted(qrels, key=int):
        text = questions[qid]["original"]
        if not has_abbreviation(text):
            continue
        expanded = expand_abbreviations(provider, text)
        base_docs, _ = ranked_doc_ids(eval_ret, text)
        exp_docs, _ = ranked_doc_ids(eval_ret, f"{text} {expanded}" if expanded else text)
        liveqa.append(
            {
                "qid": qid,
                "question": text[:160],
                "expanded": expanded,
                "ndcg_before": round(score(base_docs, qrels[qid])["ndcg@10"], 3),
                "ndcg_after": round(score(exp_docs, qrels[qid])["ndcg@10"], 3),
            }
        )

    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    summary = {
        "abbreviation_questions": len(rows),
        "top4_success_before": sum(r["before"] for r in rows),
        "top4_success_after": sum(r["after"] for r in rows),
        "liveqa_questions_evaluated": len(qrels),
        "liveqa_questions_triggering": len(liveqa),
        "liveqa_ndcg_before": mean([q["ndcg_before"] for q in liveqa]),
        "liveqa_ndcg_after": mean([q["ndcg_after"] for q in liveqa]),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "abbreviations.json").write_text(
        json.dumps({"summary": summary, "cases": rows, "liveqa": liveqa}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
