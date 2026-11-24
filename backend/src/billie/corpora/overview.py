"""What each mode can answer: coverage computed from the indexes, plus vetted examples.

Users who don't know what is inside a retrieval system ask what they would ask a general
chatbot, get "not in the documents" several times and give up (owner feedback, P2.18).
This module makes the coverage visible:

* :func:`overview` - per-mode statistics computed from the live indexes (passages, topics,
  sources, question types, paper categories, years), example questions and what is *not*
  covered. Cached per index build, because it scans the arXiv index once.
* :func:`suggest` - as-you-type suggestions taken from real titles in the index (MedQuAD
  titles are questions, arXiv titles are papers), so every suggestion is answerable.
* :func:`clean_title` / :func:`question_for` - turn an index title into a question to ask.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from typing import Any

from billie.corpora.service import CORPUS_FOR_MODE, CorpusService
from billie.retrieval.index import PassageIndex

# Examples are questions that passed the 100-case stress test on the default setup
# (docs/evaluation/stress_test/results_groq_full.json), so a first click always works.
EXAMPLES: dict[str, list[str]] = {
    "general": [
        "How does hybrid search work in Billie?",
        "Does my data leave my computer?",
        "How do I add my own documents?",
        "What languages does Billie support?",
        "¿Cómo puedo subir mis propios documentos?",
    ],
    "medical": [
        "What are the early signs of Alzheimer's disease?",
        "Is Huntington disease inherited?",
        "What tests diagnose kidney disease?",
        "How can I prevent osteoporosis?",
        "sugar ki bimari ke lakshan kya hai",
        "अस्थमा का इलाज कैसे होता है?",
    ],
    "research": [
        "Recent approaches to reduce hallucination in large language models",
        "Papers on graph neural networks for drug discovery",
        "Compare BM25 and dense retrieval for question answering",
        "Recent work on LLM agents that use tools",
        "Speech recognition for low-resource languages",
        "मशीन लर्निंग में ओवरफिटिंग क्या है?",
    ],
}

GOOD_FOR: dict[str, list[str]] = {
    "general": [
        "How Billie works (search, citations, languages, privacy)",
        "Setup and settings",
        "Questions about documents you upload",
    ],
    "medical": [
        "What a disease or condition is",
        "Symptoms, causes and risk factors",
        "Treatments, tests and outlook",
        "Genetic conditions and inheritance",
    ],
    "research": [
        "Finding papers on a topic",
        "Comparing methods and approaches",
        "Explaining a technique, with papers that use it",
    ],
}

NOT_COVERED: dict[str, list[str]] = {
    "general": ["General knowledge or news", "Medical questions (use Medical)", "Research papers (use Research)"],
    "medical": [
        "Drug doses and medication-specific advice",
        "Personal diagnosis",
        "News, prices or non-medical topics",
    ],
    "research": ["Coding help or homework solutions", "Non-computer-science fields", "Papers older than the index"],
}

MEDQUAD_SITES = {
    "GHR": "Genetics Home Reference",
    "GARD": "Genetic and Rare Diseases Center",
    "NIDDK": "NIDDK (diabetes, digestive, kidney)",
    "NINDS": "NINDS (neurological disorders)",
    "MPlusHealthTopics": "MedlinePlus",
    "NIHSeniorHealth": "NIH SeniorHealth",
    "CancerGov": "National Cancer Institute",
    "NHLBI": "NHLBI (heart, lung, blood)",
    "CDC": "CDC",
}

ARXIV_CATEGORIES = {
    "cs.LG": "Machine learning",
    "cs.CV": "Computer vision",
    "cs.CL": "Natural language processing",
    "cs.AI": "Artificial intelligence",
    "cs.RO": "Robotics",
    "cs.CR": "Security and cryptography",
    "cs.IT": "Information theory",
    "cs.DS": "Data structures and algorithms",
    "cs.SE": "Software engineering",
    "cs.NI": "Networking",
    "cs.DC": "Distributed computing",
    "cs.HC": "Human-computer interaction",
    "cs.IR": "Information retrieval",
    "cs.SY": "Systems and control",
    "eess.SY": "Systems and control",
    "cs.NE": "Neural and evolutionary computing",
    "cs.SI": "Social networks",
    "cs.DB": "Databases",
    "cs.PL": "Programming languages",
    "cs.GT": "Game theory",
    "cs.CY": "Computers and society",
    "cs.MA": "Multi-agent systems",
    "cs.SD": "Sound and audio",
    "eess.AS": "Audio and speech",
    "eess.IV": "Image and video processing",
    "eess.SP": "Signal processing",
    "cs.LO": "Logic in computer science",
    "cs.CC": "Computational complexity",
    "cs.CG": "Computational geometry",
    "cs.DM": "Discrete mathematics",
    "cs.FL": "Formal languages",
    "cs.GR": "Graphics",
    "cs.AR": "Hardware architecture",
    "cs.NA": "Numerical analysis",
    "math.NA": "Numerical analysis",
    "stat.ML": "Statistical machine learning",
    "math.OC": "Optimization and control",
}

_cache: dict[str, tuple[str, dict[str, Any]]] = {}
_lock = threading.Lock()
_TOKEN = re.compile(r"\w+", re.UNICODE)


def overview(corpora: CorpusService, *, compute: bool = True) -> dict[str, Any]:
    """Coverage for every mode.

    Statistics are cached in memory and in ``<index>.overview.json`` per index build: the
    first scan of the full 980k-paper arXiv index takes ~18 s, so the API computes it in the
    background at startup and calls this with ``compute=False`` (basic counts until ready).
    """
    modes: dict[str, Any] = {}
    for mode, corpus in CORPUS_FOR_MODE.items():
        index = corpora.index(corpus)
        info = index.info()
        stats = _cached_stats(corpus, index, f"{info.meta.get('built_at')}|{info.passages}|{info.vectors}", compute)
        modes[mode] = {
            "corpus": corpus,
            "passages": info.passages,
            "vectors": info.vectors,
            **(stats or {}),
            "good_for": GOOD_FOR[mode],
            "not_covered": NOT_COVERED[mode],
            "examples": EXAMPLES[mode],
        }
    return {"modes": modes}


def _cached_stats(corpus: str, index: PassageIndex, stamp: str, compute: bool) -> dict[str, Any] | None:
    with _lock:
        cached = _cache.get(corpus)
        if cached and cached[0] == stamp:
            return cached[1]
        path = index.directory / f"{corpus}.overview.json"
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("stamp") == stamp:
                _cache[corpus] = (stamp, saved["stats"])
                return saved["stats"]
        except (OSError, ValueError, KeyError):
            pass
        if not compute or not index.exists():
            return None
        stats = _stats(corpus, index)
        _cache[corpus] = (stamp, stats)
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"stamp": stamp, "stats": stats}), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass  # read-only index folder: the in-memory cache still works
        return stats


def _stats(corpus: str, index: PassageIndex) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{index.db_path}?mode=ro", uri=True)
    try:
        if corpus == "medquad":
            questions = conn.execute(
                "SELECT COUNT(*) FROM passages WHERE COALESCE(json_extract(metadata, '$.chunk'), 0) = 0"
            ).fetchone()[0]
            topics = conn.execute("SELECT COUNT(DISTINCT json_extract(metadata, '$.focus')) FROM passages").fetchone()[
                0
            ]
            sources = conn.execute(
                "SELECT source, COUNT(DISTINCT doc_id) FROM passages GROUP BY source ORDER BY 2 DESC"
            ).fetchall()
            qtypes = conn.execute(
                "SELECT json_extract(metadata, '$.qtype'), COUNT(*) FROM passages "
                "WHERE COALESCE(json_extract(metadata, '$.chunk'), 0) = 0 GROUP BY 1 ORDER BY 2 DESC"
            ).fetchall()
            return {
                "questions": questions,
                "topics": topics,
                "sources": [
                    {"name": MEDQUAD_SITES.get(s.split("·")[-1].strip(), s.split("·")[-1].strip()), "count": n}
                    for s, n in sources
                ],
                "question_types": [{"name": q, "count": n} for q, n in qtypes if q and n >= 20],
            }
        if corpus == "arxiv":
            years = conn.execute(
                "SELECT MIN(CAST(json_extract(metadata, '$.year') AS INTEGER)), "
                "MAX(CAST(json_extract(metadata, '$.year') AS INTEGER)) FROM passages"
            ).fetchone()
            categories = conn.execute(
                "SELECT json_extract(metadata, '$.primary_category'), COUNT(*) FROM passages "
                "GROUP BY 1 ORDER BY 2 DESC LIMIT 12"
            ).fetchall()
            seen: dict[str, int] = {}
            for code, n in categories:
                name = ARXIV_CATEGORIES.get(code or "", code or "Other")
                seen[name] = seen.get(name, 0) + n
            return {
                "years": list(years) if years and years[0] else None,
                "categories": [{"name": k, "count": v} for k, v in list(seen.items())[:10]],
            }
        documents = conn.execute("SELECT doc_id, COUNT(*) FROM passages GROUP BY doc_id ORDER BY doc_id").fetchall()
        return {"documents": [{"name": d.rsplit("/", 1)[-1], "count": n} for d, n in documents]}
    finally:
        conn.close()


def clean_title(corpus: str, title: str) -> str:
    """'What is (are) Glaucoma ?' -> 'What is Glaucoma?'; section paths keep their last part."""
    if corpus == "medquad":
        title = title.replace("What is (are)", "What is")
        treatment = re.match(r"What are the treatments for (.+?)\s*\?$", title)
        if treatment:
            title = f"How is {treatment.group(1)} treated?"
        title = re.sub(r"\s+\?", "?", title).strip()
        return re.sub(r"\?{2,}$", "?", title)
    if corpus == "knowledge":
        return title.split("›")[-1].strip()
    return " ".join(title.split())


def question_for(corpus: str, title: str) -> str:
    """The question to send when a user picks a suggestion."""
    if corpus == "arxiv":
        return f'What does the paper "{title}" propose?'
    if corpus == "knowledge":
        return f"What does the documentation say about {title}?"
    return title


def suggest(corpora: CorpusService, mode: str, text: str, limit: int = 6) -> list[dict[str, str]]:
    """Titles from the index that start with what the user is typing (prefix match on titles)."""
    corpus = CORPUS_FOR_MODE[mode]
    tokens = [t for t in _TOKEN.findall(text.casefold()) if t]
    if not tokens or len(text.strip()) < 2:
        return []
    index = corpora.index(corpus)
    if not index.exists():
        return []
    # Every complete word must appear in the title; the word being typed matches as a prefix.
    *words, last = tokens[-6:]
    query = " ".join(f'"{w}"' for w in words) + f' "{last}"*'
    sql = (
        "SELECT p.title FROM passages_fts f JOIN passages p ON p.rowid = f.rowid "
        "WHERE passages_fts MATCH ? ORDER BY bm25(passages_fts, 10.0, 0.0) LIMIT 200"
    )
    conn = sqlite3.connect(f"file:{index.db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(sql, (f"title : ({query})",)).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    labels: list[str] = []
    seen: set[str] = set()
    for (title,) in rows:
        label = clean_title(corpus, title)
        if label.casefold() not in seen:
            seen.add(label.casefold())
            labels.append(label)
    if corpus == "medquad":
        # Short titles are the core topics ("What is Diabetes?"); long ones are page-title quirks.
        labels.sort(key=len)
    return [{"title": label, "question": question_for(corpus, label)} for label in labels[:limit]]
