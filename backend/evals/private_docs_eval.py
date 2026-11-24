"""How well does Billie answer questions about a visitor's own documents? (DEVLOG P2.20)

    python -m evals.private_docs_eval                    # local model (no cloud quota)
    python -m evals.private_docs_eval --use-app-model --only F04,T03 --tag groq   # spot-check on the app's model

Uploads two PDFs as one demo visitor (a generated 22-page city report with known facts, and
the real 20-page project report), then asks 30 questions: plain facts, table values,
questions combining two sections, lists/summaries, and questions the documents do not answer.

Two scores per question:
* **retrieval** (model-independent): was a passage containing the answer among the sources?
* **answer**: does the reply contain the expected facts (or, for unanswerable questions,
  decline without inventing a number)? Every answer is also written out for reading by hand.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from billie.api.services import build_services
from billie.chat.pipeline import ChatTurn
from billie.paths import EVAL_DATASETS_DIR, EVAL_RESULTS_DIR
from billie.runtime import load_settings
from evals.common import eval_llm

DATA = EVAL_DATASETS_DIR / "private_docs"
RESULTS = EVAL_RESULTS_DIR / "private_docs"
DOCS = {"riverton": DATA / "riverton_annual_report_2025.pdf", "billie_report": DATA / "billie_project_report.pdf"}
OWNER = "v_" + "e" * 32
_NUMBER = re.compile(r"\d(?:[\d,.]*\d)?")  # ends on a digit, so "2023." is read as 2023


def _has(text: str, term: str) -> bool:
    return term.casefold() in text.casefold()


def expected_terms(case: dict) -> list[str]:
    """Strings that identify the gold passage (for the retrieval score)."""
    if "any" in case:
        return case["any"][0]
    return case.get("terms", [])


def score(case: dict, answer: str, sources: list[dict], done: dict) -> dict:
    texts = " ".join(s["text"] for s in sources)
    if case["kind"] == "unanswerable":
        declined = bool(done.get("abstained") or done.get("not_found"))
        # A decline must not smuggle in a made-up figure for the thing asked about.
        # Years are not figures ("check the 2024 report" is fine); numbers from the question are echoes.
        invented = [
            n
            for n in _NUMBER.findall(re.sub(r"\[\d+\]", "", answer))  # citation markers are not figures
            if n not in texts and n not in case["q"] and not re.fullmatch(r"(19|20)\d\d", n)
        ]
        return {
            "retrieval": None,
            "correct": declined and not invented,
            "detail": f"declined={declined} invented={invented}",
        }
    if "any" in case:
        correct = any(all(_has(answer, t) for t in group) for group in case["any"])
        evidence = case.get("evidence")  # computed answers: the figures they are computed from
        groups = [evidence] if evidence else case["any"]
        retrieved = any(all(_has(texts, t) for t in group) for group in groups)
        return {"retrieval": retrieved, "correct": correct, "detail": ""}
    found = [t for t in case["terms"] if _has(answer, t)]
    in_sources = [t for t in case["terms"] if _has(texts, t)]
    return {
        "retrieval": len(in_sources) >= case["min_of"],
        "correct": len(found) >= case["min_of"],
        "detail": f"answer has {len(found)}/{len(case['terms'])}, sources have {len(in_sources)}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-app-model", action="store_true", help="use the app's configured model (may be cloud)")
    parser.add_argument("--only", default="", help="comma-separated question ids")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds between questions (cloud rate limits)")
    parser.add_argument("--tag", default="local")
    args = parser.parse_args()
    cases = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only.split(","))]

    base = load_settings()
    tmp = Path(tempfile.mkdtemp(prefix="billie-private-eval-"))
    settings = replace(
        base,
        demo=True,
        paths=replace(base.paths, data_dir=tmp, sessions_db=tmp / "sessions.sqlite", index_dir=tmp / "index"),
    )
    services = build_services(settings, llm=eval_llm(tmp, args.use_app_model, base.paths.llm_settings))
    services.corpora.sync_knowledge()
    assert services.private is not None
    for path in DOCS.values():
        started = time.perf_counter()
        docs = services.private.add(OWNER, path.name, path.read_bytes())
        added = next(d for d in docs if d.name == path.name)
        print(f"uploaded {path.name}: {added.passages} passages in {time.perf_counter() - started:.1f}s", flush=True)

    rows = []
    for i, case in enumerate(cases, start=1):
        started = time.perf_counter()
        events = list(services.pipeline.run(ChatTurn(case["q"], "general", owner=OWNER)))
        done = next(e["data"] for e in events if e["event"] == "done")
        meta = next(e["data"] for e in events if e["event"] == "meta")
        sources = meta["sources"]
        result = score(case, done["answer"], sources, done)
        row = {
            **case,
            **result,
            "answer": done["answer"],
            "grounded": done.get("grounded"),
            "abstained": done.get("abstained"),
            "sources": [{"source": s["source"], "title": s["title"]} for s in sources],
            "cited_private": any(sources[n - 1]["metadata"].get("private") for n in done.get("citations", [])),
            "model": done.get("model"),
            "seconds": round(time.perf_counter() - started, 1),
        }
        rows.append(row)
        mark = "OK " if row["correct"] else "BAD"
        print(
            f"[{i}/{len(cases)}] {mark} {case['id']} {case['kind']:12} retrieval={row['retrieval']} {row['seconds']}s {row['detail']}",
            flush=True,
        )
        if args.pause and i < len(cases):
            time.sleep(args.pause)

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    answerable = [r for r in rows if r["retrieval"] is not None]
    summary = {
        "questions": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "retrieval_hit": f"{sum(bool(r['retrieval']) for r in answerable)}/{len(answerable)}",
        "by_kind": {k: f"{sum(r['correct'] for r in v)}/{len(v)}" for k, v in sorted(by_kind.items())},
        "mean_seconds": round(sum(r["seconds"] for r in rows) / max(len(rows), 1), 1),
        "model": rows[0]["model"] if rows else None,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"results_{args.tag}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / f"summary_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
