"""Medical retrieval benchmark on real consumer questions with human relevance judgments.

Queries    TREC 2017 LiveQA Medical test questions (consumer questions sent to the NLM),
           both the original message and the NIST assessors' paraphrase.
Judgments  The MedQuAD authors' qrels: 2,479 MedQuAD answers graded 1-Incorrect,
           2-Related, 3-Incomplete, 4-Excellent (Ben Abacha & Demner-Fushman, 2019).

Honest-evaluation notes (also written into the report):
* Answers from ADAM, MedlinePlus Drugs and Herbs were removed from MedQuAD for copyright
  reasons, so they are not in Billie's index. We keep only questions that have at least one
  answer graded 3-4 that *is* in the index.
* The judgment pool comes from other systems' runs, so unjudged answers count as
  non-relevant (standard TREC practice). Absolute scores are therefore conservative;
  comparisons between Billie's retrievers on the same pool are the meaningful part.
* Retrieval returns passages; metrics are computed over QA documents (first occurrence).

    python -m evals.retrieval_liveqa            # needs data/index/medquad (+ embeddings for dense/hybrid)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import statistics
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable

from billie.paths import DATA_DIR, EVAL_DATASETS_DIR, MEDQUAD_DIR, REPO_ROOT
from billie.retrieval.embeddings import OllamaEmbedder
from billie.retrieval.hybrid import HybridRetriever
from billie.retrieval.index import PassageIndex

QUESTIONS = EVAL_DATASETS_DIR / "liveqa" / "test_questions.xml"
QUESTIONS_URL = (
    "https://raw.githubusercontent.com/abachaa/LiveQA_MedicalTask_TREC2017/master/TestDataset/"
    "TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml"
)
QRELS = (
    MEDQUAD_DIR
    / "liveqa"
    / "QA-TestSet-LiveQA-Med-Qrels-2479-Answers"
    / ("All-qrels_LiveQAMed2017-TestQuestions_2479_Judged-Answers.txt")
)
RESULTS = REPO_ROOT / "docs" / "evaluation" / "retrieval"

SUBSET_FOLDER = {
    "CancerGov": "1", "GARD": "2", "GHR": "3", "MPlusHealthTopics": "4", "NIDDK": "5", "NINDS": "6",
    "NIHSeniorHealth": "7", "NHLBI": "8", "CDC": "9", "ADAM": "10", "MPlusDrugs": "11", "MPlusHerbsSuppls": "12",
}  # fmt: skip
GAIN = {1: 0, 2: 1, 3: 2, 4: 3}  # graded gain for nDCG; relevant for recall/MRR = grade >= 3
_ANSWER_ID = re.compile(r"^(?P<subset>[A-Za-z]+)_(?P<stem>.+)_Sec(?P<sec>\d+)\.txt$")
K = 10


def load_questions() -> dict[str, dict[str, str]]:
    if not QUESTIONS.exists():  # public TREC data; downloaded on first run rather than committed
        import httpx

        QUESTIONS.parent.mkdir(parents=True, exist_ok=True)
        QUESTIONS.write_bytes(httpx.get(QUESTIONS_URL, timeout=60, follow_redirects=True).raise_for_status().content)
    root = ET.parse(QUESTIONS).getroot()
    questions = {}
    for node in root.iter("NLM-QUESTION"):
        qid = node.get("qid", "").removeprefix("TQ")
        original = node.find("Original-Question")
        subject = (original.findtext("SUBJECT") or "").strip() if original is not None else ""
        message = (original.findtext("MESSAGE") or "").strip() if original is not None else ""
        questions[qid] = {
            "original": " ".join(f"{subject}. {message}".split()).strip(". "),
            "paraphrase": " ".join((node.findtext("NIST-PARAPHRASE") or "").split()),
        }
    return questions


def load_qrels(indexed_docs: set[str]) -> dict[str, dict[str, int]]:
    """qid -> {doc_id: grade}, restricted to answers present in Billie's index."""
    qrels: dict[str, dict[str, int]] = {}
    for line in QRELS.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        qid, judgment, answer_id = parts
        match = _ANSWER_ID.match(answer_id)
        if not match or match["subset"] not in SUBSET_FOLDER:
            continue
        doc_id = f"{SUBSET_FOLDER[match['subset']]}-{match['stem']}-{int(match['sec'])}"
        if doc_id in indexed_docs:
            qrels.setdefault(qid, {})[doc_id] = int(judgment.split("-")[0])
    return {qid: docs for qid, docs in qrels.items() if any(g >= 3 for g in docs.values())}


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def score(ranked_docs: list[str], judged: dict[str, int]) -> dict[str, float]:
    top = ranked_docs[:K]
    gains = [GAIN.get(judged.get(d, 1), 0) for d in top]
    ideal = sorted((GAIN[g] for g in judged.values()), reverse=True)[:K]
    relevant = {d for d, g in judged.items() if g >= 3}
    first = next((i for i, d in enumerate(top) if d in relevant), None)
    return {
        "ndcg@10": dcg(gains) / dcg(ideal) if dcg(ideal) else 0.0,
        "recall@10": len(relevant & set(top)) / len(relevant),
        "mrr@10": 1.0 / (first + 1) if first is not None else 0.0,
        "success@5": 1.0 if first is not None and first < 5 else 0.0,
    }


def ranked_doc_ids(retriever: HybridRetriever, query: str) -> tuple[list[str], float]:
    started = time.perf_counter()
    result = retriever.search(query, top_k=40)
    elapsed = (time.perf_counter() - started) * 1000
    seen: list[str] = []
    for hit in result.hits:
        if hit.passage.doc_id not in seen:
            seen.append(hit.passage.doc_id)
    return seen, elapsed


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-model", default="embeddinggemma")
    args = parser.parse_args()

    index = PassageIndex(DATA_DIR / "index", "medquad")
    if not index.exists():
        print("MedQuAD index missing. Run: python -m billie.ingest medquad")
        return 1
    with sqlite3.connect(index.db_path) as conn:
        indexed_docs = {row[0] for row in conn.execute("SELECT DISTINCT doc_id FROM passages")}
    questions = load_questions()
    qrels = load_qrels(indexed_docs)
    evaluated = sorted(qrels, key=int)
    print(f"{len(questions)} test questions; {len(evaluated)} have a grade-3/4 answer in the index")

    embedder = OllamaEmbedder(args.embed_model)
    systems: dict[str, Callable[[], HybridRetriever]] = {"bm25": lambda: HybridRetriever(index, mode="bm25")}
    if index.has_vectors() and embedder.available():
        systems["dense"] = lambda: HybridRetriever(index, embedder, mode="dense", min_dense_score=0.0)
        systems["hybrid (rrf)"] = lambda: HybridRetriever(index, embedder, mode="hybrid", min_dense_score=0.0)
    else:
        print("No embeddings available - evaluating BM25 only.")

    report: dict = {
        "benchmark": "TREC 2017 LiveQA-Medical test questions x MedQuAD qrels (Ben Abacha & Demner-Fushman 2019)",
        "questions_total": len(questions),
        "questions_evaluated": len(evaluated),
        "relevance": "grade>=3 (Incomplete/Excellent) relevant; nDCG gains Related=1, Incomplete=2, Excellent=3",
        "embedding_model": embedder.model if "dense" in systems else None,
        "passages_embedded": index.info().vectors,
        "results": {},
    }
    per_query: dict = {}
    for query_kind in ("original", "paraphrase"):
        for name, make in systems.items():
            retriever = make()
            metrics: dict[str, list[float]] = {"ndcg@10": [], "recall@10": [], "mrr@10": [], "success@5": []}
            latencies: list[float] = []
            for qid in evaluated:
                docs, ms = ranked_doc_ids(retriever, questions[qid][query_kind])
                latencies.append(ms)
                s = score(docs, qrels[qid])
                for key, value in s.items():
                    metrics[key].append(value)
                per_query.setdefault(f"{query_kind}/{name}", {})[qid] = {**s, "top3": docs[:3]}
            row = {key: round(statistics.mean(values), 4) for key, values in metrics.items()}
            row["latency_p50_ms"] = round(percentile(latencies, 50), 1)
            row["latency_p95_ms"] = round(percentile(latencies, 95), 1)
            report["results"][f"{query_kind}/{name}"] = row
            print(f"{query_kind:10} {name:13} " + "  ".join(f"{k}={v}" for k, v in row.items()))

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "liveqa_medquad.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESULTS / "liveqa_medquad_per_query.json").write_text(json.dumps(per_query, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS / 'liveqa_medquad.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
