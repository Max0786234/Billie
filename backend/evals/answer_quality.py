"""End-to-end answer-quality evaluation on real consumer health questions.

Retrieval quality is measured by ``retrieval_liveqa``. This script measures the *answers*:
it runs the full live pipeline (local LLM included) on the same 39 TREC 2017 LiveQA
questions and reports:

* **Outcome mix** - grounded / abstained / ungrounded / retrieval-only fallback.
* **Citation support** - every answer sentence carrying a [n] citation is checked against
  the passage(s) it cites by an LLM judge (SUPPORTED / PARTIAL / UNSUPPORTED).
  Caveat: the judge is the same local model that wrote the answer (self-judging bias),
  so a sample of verdicts is spot-checked by hand and reported in the DEVLOG.
* **Agreement with expert answers** - embedding cosine between Billie's answer and the NLM
  assessors' reference answers (max over references), compared with the simplest baseline:
  returning the top retrieved passage verbatim. This shows whether generation adds value.
* **Relevant source retrieved** - whether a human-judged relevant answer (grade 3-4) was
  among the sources given to the model.
* **Latency** per answer.

Conversations go to a temporary database (the owner's chat history is untouched) and the
answer cache is disabled, so every answer is generated fresh.

    python -m evals.answer_quality                 # ~45 min on a laptop CPU
    python -m evals.answer_quality --limit 5       # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np

from billie.api.services import build_services
from billie.chat.pipeline import ChatTurn
from billie.chat.prompts import CITATION, strip_citations
from billie.llm.base import GenerationOptions, LLMError, Message, complete
from billie.paths import REPO_ROOT
from billie.runtime import load_settings
from evals.common import eval_llm
from evals.retrieval_liveqa import QUESTIONS, load_qrels, load_questions

RESULTS = REPO_ROOT / "docs" / "evaluation" / "answers"
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
JUDGE_SYSTEM = (
    "You check whether a claim is supported by the evidence text. Answer with exactly one word:\n"
    "SUPPORTED - the evidence states or clearly implies the whole claim.\n"
    "PARTIAL - the evidence supports part of the claim but not all of it.\n"
    "UNSUPPORTED - the evidence does not support the claim."
)
VERDICT_SCORE = {"SUPPORTED": 1.0, "PARTIAL": 0.5, "UNSUPPORTED": 0.0}


def load_references() -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    for node in ET.parse(QUESTIONS).getroot().iter("NLM-QUESTION"):
        qid = node.get("qid", "").removeprefix("TQ")
        answers = [" ".join((a.text or "").split()) for a in node.iter("ANSWER") if (a.text or "").strip()]
        if answers:
            refs[qid] = answers
    return refs


def cited_claims(answer: str) -> list[tuple[str, list[int]]]:
    """(claim text without markers, cited source numbers) for each sentence that cites something."""
    claims = []
    for sentence in _SENTENCE.split(answer):
        numbers = sorted({int(n) for n in CITATION.findall(sentence)})
        text = strip_citations(sentence).strip(" -*•")
        if numbers and len(text) > 15:
            claims.append((text, numbers))
    return claims


_NOT_A_CLAIM = re.compile(
    r"could not find|couldn't find|not (mentioned|covered|included|available|provided|specified|addressed)|"
    r"(do|does|did)(n't| not) (mention|include|cover|address|provide|state|specify|confirm|discuss|say)|"
    r"no (source|document|information)|the (sources|documents) (only|do not|don't)|"
    r"consult|healthcare professional|doctor|rephrase|please|let me know|sorry|i'm not able|emergency",
    re.IGNORECASE,
)


def uncited_claims(answer: str) -> list[str]:
    """Factual-looking sentences without any citation (refusals and advice lines excluded)."""
    out = []
    for sentence in _SENTENCE.split(answer):
        text = sentence.strip(" -*•")
        if len(text) > 40 and not CITATION.search(text) and not _NOT_A_CLAIM.search(text):
            out.append(text)
    return out


def judge(provider, claim: str, evidence: str) -> str:
    reply = complete(
        provider,
        [Message("system", JUDGE_SYSTEM), Message("user", f"Evidence:\n{evidence[:2500]}\n\nClaim: {claim}")],
        GenerationOptions(temperature=0.0, max_tokens=4),
    ).upper()
    for verdict in ("UNSUPPORTED", "SUPPORTED", "PARTIAL"):  # UNSUPPORTED first: it contains "SUPPORTED"
        if verdict in reply:
            return verdict
    return "UNSUPPORTED"


def _norm(text: str) -> str:
    """Compare sentences ignoring citations, case, spacing and edge punctuation."""
    return " ".join(strip_citations(text).lower().split()).strip(" .-*•")


def _rate(verdicts: list[str]) -> float | None:
    return round(statistics.mean(VERDICT_SCORE[v] for v in verdicts), 3) if verdicts else None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # embeddings are L2-normalised


def rescore(path: Path) -> dict[str, float]:
    """Recompute the uncited-claim metrics from saved answers (no regeneration)."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    counts = [len(uncited_claims(r["answer"])) for r in rows]
    return {
        "answers_with_uncited_claims": round(sum(c > 0 for c in counts) / len(counts), 3),
        "uncited_claims_total": sum(counts),
        "cited_claims_total": sum(len(cited_claims(r["answer"])) for r in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--queries", choices=["original", "paraphrase"], default="original")
    parser.add_argument("--tag", default="", help="suffix for result files, e.g. 'baseline'")
    parser.add_argument("--use-app-model", action="store_true", help="use the app's configured model (may be cloud)")
    args = parser.parse_args()

    base = load_settings()
    tmp = Path(tempfile.mkdtemp(prefix="billie-eval-"))
    settings = replace(base, paths=replace(base.paths, sessions_db=tmp / "sessions.sqlite"))
    services = build_services(settings, llm=eval_llm(tmp, args.use_app_model, base.paths.llm_settings))
    services.pipeline.cache = None  # every answer generated fresh
    embedder = services.corpora.embedder
    provider = services.llm.provider()
    if embedder is None:
        print("This evaluation needs the embedding model (BILLIE_EMBED_MODEL).")
        return 1

    questions = load_questions()
    refs = load_references()
    import sqlite3

    with sqlite3.connect(services.corpora.index("medquad").db_path) as conn:
        indexed = {row[0] for row in conn.execute("SELECT DISTINCT doc_id FROM passages")}
    qrels = load_qrels(indexed)
    qids = [q for q in sorted(qrels, key=int) if q in refs][: args.limit]
    print(f"{len(qids)} questions, model {provider.model}, judge {provider.model} (self-judging)")

    rows = []
    for i, qid in enumerate(qids, start=1):
        question = questions[qid][args.queries]
        started = time.perf_counter()
        events = list(services.pipeline.run(ChatTurn(question, "medical")))
        elapsed = time.perf_counter() - started
        meta = next(e["data"] for e in events if e["event"] == "meta")
        done = next(e["data"] for e in events if e["event"] == "done")
        sources = meta["sources"]
        answer = done["answer"]

        outcome = (
            "fallback" if not done["used_llm"]
            else "abstained" if done["abstained"]
            else "grounded" if done["grounded"]
            else "ungrounded" if done["grounded"] is False
            else "no_sources"
        )  # fmt: skip
        relevant = {d for d, g in qrels[qid].items() if g >= 3}
        relevant_retrieved = any(s["id"].split(":", 1)[1].split("#")[0] in relevant for s in sources)

        auto_sentences = {_norm(a["sentence"]) for a in done.get("auto_citations") or []}
        verdicts = []
        for claim, numbers in cited_claims(answer):
            evidence = "\n\n".join(sources[n - 1]["text"] for n in numbers if 1 <= n <= len(sources))
            if evidence:
                try:
                    verdicts.append(
                        {
                            "claim": claim,
                            "cites": numbers,
                            "verdict": judge(provider, claim, evidence),
                            "auto": _norm(claim) in auto_sentences,  # citation added by citation repair
                        }
                    )
                except LLMError as exc:
                    print(f"  judge failed: {exc}")

        ref_vecs = embedder.embed_queries(refs[qid])
        answer_vec = embedder.embed_queries([strip_citations(answer)])[0]
        top_vec = embedder.embed_queries([sources[0]["text"]])[0] if sources else None
        row = {
            "qid": qid,
            "question": question,
            "outcome": outcome,
            "relevant_source_retrieved": relevant_retrieved,
            "answer": answer,
            "claims": verdicts,
            "uncited_claims": uncited_claims(answer),
            "auto_citations": done.get("auto_citations") or [],
            "unverified_sentences": done.get("unverified_sentences") or 0,
            "support": statistics.mean(VERDICT_SCORE[v["verdict"]] for v in verdicts) if verdicts else None,
            "ref_similarity_answer": max(cosine(answer_vec, r) for r in ref_vecs),
            "ref_similarity_top_passage": max(cosine(top_vec, r) for r in ref_vecs) if top_vec is not None else None,
            "seconds": round(elapsed, 1),
        }
        rows.append(row)
        print(
            f"[{i}/{len(qids)}] Q{qid} {outcome:10} support={row['support'] if row['support'] is not None else '-'} "
            f"sim={row['ref_similarity_answer']:.3f} vs top={row['ref_similarity_top_passage'] or 0:.3f} {elapsed:.0f}s",
            flush=True,
        )

    def share(name: str) -> float:
        return round(sum(r["outcome"] == name for r in rows) / len(rows), 3)

    all_verdicts = [v["verdict"] for r in rows for v in r["claims"]]
    paired = [
        (r["ref_similarity_answer"], r["ref_similarity_top_passage"]) for r in rows if r["ref_similarity_top_passage"]
    ]
    summary = {
        "benchmark": "TREC 2017 LiveQA Medical questions with MedQuAD qrels and NLM reference answers",
        "queries": args.queries,
        "questions": len(rows),
        "model": provider.model,
        "judge": f"{provider.model} (same model: self-judging bias; spot-checked manually)",
        "outcomes": {k: share(k) for k in ("grounded", "abstained", "ungrounded", "fallback", "no_sources")},
        "relevant_source_retrieved": round(statistics.mean(r["relevant_source_retrieved"] for r in rows), 3),
        "answers_with_uncited_claims": round(statistics.mean(bool(r["uncited_claims"]) for r in rows), 3),
        "uncited_claims_total": sum(len(r["uncited_claims"]) for r in rows),
        "auto_citations_added": sum(len(r.get("auto_citations") or []) for r in rows),
        "auto_citation_support_rate": _rate([v["verdict"] for r in rows for v in r["claims"] if v.get("auto")]),
        "model_citation_support_rate": _rate([v["verdict"] for r in rows for v in r["claims"] if not v.get("auto")]),
        "unverified_sentences_total": sum(r.get("unverified_sentences") or 0 for r in rows),
        "claims_judged": len(all_verdicts),
        "claim_verdicts": {v: all_verdicts.count(v) for v in VERDICT_SCORE},
        "citation_support_rate": round(statistics.mean(VERDICT_SCORE[v] for v in all_verdicts), 3)
        if all_verdicts
        else None,
        "ref_similarity": {
            "billie_answer_mean": round(statistics.mean(a for a, _ in paired), 3),
            "top_passage_baseline_mean": round(statistics.mean(b for _, b in paired), 3),
            "answer_better_share": round(sum(a > b for a, b in paired) / len(paired), 3),
        },
        "latency_seconds": {
            "p50": round(statistics.median(r["seconds"] for r in rows), 1),
            "max": round(max(r["seconds"] for r in rows), 1),
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = f"answer_quality_{args.queries}" + (f"_{args.tag}" if args.tag else "")
    summary["tag"] = args.tag
    (RESULTS / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS / f"{name}_per_question.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
