"""Broad behaviour + latency test: ~100 questions across every mode and question type.

Cases live in ``evals/datasets/stress_test.json``. Each states the expected behaviour:
  answer   a grounded answer that mentions at least one expected key fact (if any are listed)
  abstain  the documents don't cover it: Billie must decline (no citations) and avoid ``forbid`` facts
  alert    the emergency banner must be triggered
  any      no crash; language/forbid checks still apply
plus an optional reply language (``hi``, ``bn``, ``es``, ``fr``, or ``hinglish`` = Hindi in Latin script).

Automatic checks are deliberately simple; every answer is also written to the report so a
human can read it. Timing is broken down per pipeline stage to explain slow answers.

    python -m evals.stress_test            # ~60 min on a laptop CPU
    python -m evals.stress_test --only M01,M02
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from billie.api.services import build_services
from billie.chat.pipeline import ChatTurn
from billie.nlp.language import detect_language
from billie.paths import EVAL_DATASETS_DIR, REPO_ROOT
from billie.runtime import load_settings
from evals.common import eval_llm

CASES = EVAL_DATASETS_DIR / "stress_test.json"
IMAGES = EVAL_DATASETS_DIR / "stress_images"
RESULTS = REPO_ROOT / "docs" / "evaluation" / "stress_test"
CACHE_REPEATS = ["M01", "G01", "R01", "M02"]
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def language_ok(answer: str, expected: str | None) -> bool | None:
    if not expected:
        return None
    text = re.sub(r"\[\d+\]", "", answer)
    if expected == "hinglish":
        return not _DEVANAGARI.search(text)  # must stay in Latin script
    return detect_language(text).code == expected


def check(case: dict, done: dict, meta: dict) -> tuple[bool, list[str]]:
    answer = done["answer"]
    lower = answer.lower()
    problems: list[str] = []
    expect = case["expect"]
    if done.get("invalid_citations"):
        problems.append(f"cites non-existent sources {done['invalid_citations']}")
    if expect == "answer":
        if done.get("grounded") is not True:
            problems.append(f"not grounded (abstained={done.get('abstained')}, used_llm={done.get('used_llm')})")
        if case.get("kw") and not any(k in lower for k in case["kw"]):
            problems.append(f"none of the expected facts {case['kw']}")
    elif expect == "abstain":
        if done.get("citations") and not done.get("abstained"):
            problems.append("answered with citations although the documents don't cover it")
    elif expect == "alert" and not meta.get("alert"):
        problems.append("emergency banner NOT shown")
    for bad in case.get("forbid", []):
        if bad in lower:
            problems.append(f"contains forbidden content '{bad}'")
    lang = language_ok(answer, case.get("lang"))
    if lang is False:
        problems.append(f"reply not in expected language '{case['lang']}'")
    if case.get("sentiment") and meta["analysis"]["sentiment"] != case["sentiment"]:
        problems.append(f"sentiment {meta['analysis']['sentiment']} != {case['sentiment']}")
    return not problems, problems


def run_case(services, case: dict, cid: str | None = None) -> dict:
    image_ids = []
    if case.get("image"):
        image_ids = [services.images.save((IMAGES / case["image"]).read_bytes()).id]
    for earlier in case.get("turns", []):
        events = list(services.pipeline.run(ChatTurn(earlier, case["mode"], cid)))
        cid = next(e["data"] for e in events if e["event"] == "done")["conversation_id"]
    started = time.perf_counter()
    events = list(services.pipeline.run(ChatTurn(case["q"], case["mode"], cid, image_ids)))
    wall = time.perf_counter() - started
    meta = next(e["data"] for e in events if e["event"] == "meta")
    done = next(e["data"] for e in events if e["event"] == "done")
    ok, problems = check(case, done, meta)
    return {
        "id": case["id"],
        "mode": case["mode"],
        "kind": case["kind"],
        "question": case["q"],
        "expect": case["expect"],
        "pass": ok,
        "problems": problems,
        "answer": done["answer"],
        "outcome": "grounded"
        if done.get("grounded")
        else "abstained"
        if done.get("abstained")
        else ("ungrounded" if done.get("grounded") is False else "no_sources" if not meta["sources"] else "other"),
        "language": meta["analysis"]["language"],
        "sentiment": meta["analysis"]["sentiment"],
        "alert": bool(meta.get("alert")),
        "sources": [s["title"][:80] for s in meta["sources"]],
        "translated_query": meta["retrieval"].get("translated_query"),
        "retrieval_mode": meta["retrieval"]["mode"],
        "citations": done.get("citations"),
        "auto_citations": len(done.get("auto_citations") or []),
        "unverified_sentences": done.get("unverified_sentences"),
        "cached": done.get("cached", False),
        "answer_chars": len(done["answer"]),
        "timings_ms": done.get("timings_ms", {}),
        "wall_s": round(wall, 1),
    }


def summarise(rows: list[dict]) -> dict:
    fresh = [r for r in rows if not r["cached"]]

    def stats(values: list[float]) -> dict:
        if not values:
            return {}
        ordered = sorted(values)
        return {
            "n": len(values),
            "mean": round(statistics.mean(values), 1),
            "p50": round(statistics.median(values), 1),
            "p90": round(ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)], 1),  # nearest-rank
            "max": round(max(values), 1),
        }

    by_mode: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, list[float]] = defaultdict(list)
    for r in fresh:
        by_mode[r["mode"]].append(r["wall_s"])
        group = (
            "image" if r["kind"].startswith("image")
            else "non-english" if r["language"] != "en"
            else "emergency" if r["kind"].startswith("emergency")
            else "out-of-scope/abstain" if r["outcome"] in ("abstained", "no_sources")
            else "follow-up" if r["kind"].startswith("follow-up")
            else "standard"
        )  # fmt: skip
        by_kind[group].append(r["wall_s"])
    stage_means = {}
    for stage in (
        "vision_ms",
        "translation_ms",
        "retrieval_ms",
        "first_token_ms",
        "generation_ms",
        "citation_repair_ms",
    ):
        values = [r["timings_ms"][stage] / 1000 for r in fresh if stage in r["timings_ms"]]
        if values:
            stage_means[stage.removesuffix("_ms")] = {"n": len(values), "mean_s": round(statistics.mean(values), 2)}
    chars = [r["answer_chars"] for r in fresh]
    walls = [r["wall_s"] for r in fresh]
    corr = 0.0
    if len(fresh) > 2 and statistics.pstdev(chars) and statistics.pstdev(walls):
        mc, mw = statistics.mean(chars), statistics.mean(walls)
        cov = sum((c - mc) * (w - mw) for c, w in zip(chars, walls, strict=True)) / len(fresh)
        corr = cov / (statistics.pstdev(chars) * statistics.pstdev(walls))
    by_mode_pass: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_mode_pass[r["mode"]].append(r["pass"])
    return {
        "cases": len(rows),
        "passed": sum(r["pass"] for r in rows),
        "pass_rate": round(sum(r["pass"] for r in rows) / len(rows), 3),
        "pass_rate_by_mode": {m: round(sum(v) / len(v), 3) for m, v in by_mode_pass.items()},
        "outcomes": {o: sum(r["outcome"] == o for r in rows) for o in {r["outcome"] for r in rows}},
        "latency_fresh_s": stats(walls),
        "latency_by_mode_s": {m: stats(v) for m, v in by_mode.items()},
        "latency_by_group_s": {g: stats(v) for g, v in by_kind.items()},
        "latency_cached_s": stats([r["wall_s"] for r in rows if r["cached"]]),
        "stage_means": stage_means,
        "correlation_answer_length_vs_time": round(corr, 2),
        "mean_answer_chars": round(statistics.mean(chars), 0) if chars else 0,
    }


def write_report(rows: list[dict], summary: dict) -> None:
    lines = ["# Stress test report", "", "Generated by `python -m evals.stress_test`.", "", "## Summary", "",
             "```json", json.dumps(summary, indent=2), "```", "", "## Failures", ""]  # fmt: skip
    for r in rows:
        if not r["pass"]:
            lines += [f"### {r['id']} ({r['mode']}, {r['kind']})", f"**Q:** {r['question']}",
                      f"**Problems:** {'; '.join(r['problems'])}", f"**A:** {r['answer'][:700]}", ""]  # fmt: skip
    lines += [
        "## Slowest 10 (fresh answers)",
        "",
        "| id | mode | kind | total s | stages (s) | answer chars |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted((r for r in rows if not r["cached"]), key=lambda r: -r["wall_s"])[:10]:
        stages = ", ".join(
            f"{k.removesuffix('_ms')} {v / 1000:.1f}" for k, v in r["timings_ms"].items() if k != "total_ms"
        )
        lines.append(f"| {r['id']} | {r['mode']} | {r['kind']} | {r['wall_s']} | {stages} | {r['answer_chars']} |")
    lines += ["", "## All answers", ""]
    for r in rows:
        mark = "PASS" if r["pass"] else "FAIL"
        lines += [
            f"**{r['id']} [{mark}] ({r['wall_s']} s, {r['outcome']})** {r['question']}",
            "",
            f"> {r['answer'][:500]}",
            "",
        ]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma-separated case ids")
    parser.add_argument("--use-app-model", action="store_true", help="use the app's configured model (may be cloud)")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds between cases (stay under cloud rate limits)")
    parser.add_argument("--tag", default="", help="suffix for result files")
    args = parser.parse_args()
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    base = load_settings()
    tmp = Path(tempfile.mkdtemp(prefix="billie-stress-"))
    settings = replace(
        base,
        paths=replace(base.paths, sessions_db=tmp / "sessions.sqlite", image_uploads_dir=tmp / "images"),
    )
    services = build_services(settings, llm=eval_llm(tmp, args.use_app_model, base.paths.llm_settings))
    cache = services.pipeline.cache
    services.pipeline.cache = None  # fresh answers for the main run

    rows = []
    for i, case in enumerate(cases, start=1):
        row = run_case(services, case)
        rows.append(row)
        flag = "PASS" if row["pass"] else "FAIL " + "; ".join(row["problems"])
        print(f"[{i}/{len(cases)}] {case['id']} {row['wall_s']:>5}s {row['outcome']:10} {flag}", flush=True)
        if args.pause and i < len(cases):
            time.sleep(args.pause)

    if not args.only and cache is not None:  # cache: ask a few questions twice
        services.pipeline.cache = cache
        by_id = {c["id"]: c for c in cases}
        for cid in CACHE_REPEATS:
            run_case(services, by_id[cid])  # fills the cache
            row = run_case(services, {**by_id[cid], "id": f"{cid}-repeat"})
            rows.append(row)
            print(f"[cache] {cid} repeat {row['wall_s']}s cached={row['cached']}", flush=True)

    summary = summarise(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.tag:
        (RESULTS / f"results_{args.tag}.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (RESULTS / f"summary_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary["stage_means"], indent=2))
        return 0
    (RESULTS / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(rows, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
