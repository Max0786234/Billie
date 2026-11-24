"""arXiv metadata snapshot -> passages (computer-science papers only).

The Kaggle snapshot is ~5.4 GB of JSON lines (~2.9M papers). We stream it once,
keep papers with at least one ``cs.*`` category, and store one passage per paper
(title + abstract). Nothing is loaded into memory at once.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from billie.retrieval.base import Passage


def iter_arxiv(path: Path, *, category_prefix: str = "cs.", limit: int | None = None) -> Iterator[Passage]:
    count = 0
    seen: set[str] = set()  # the snapshot contains a few papers more than once
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        rows = json.load(handle) if first == "[" else (json.loads(line) for line in handle if line.strip())
        for row in rows:
            categories = (row.get("categories") or "").split()
            if not any(c.startswith(category_prefix) for c in categories):
                continue
            passage = _to_passage(row, categories)
            if passage is None or passage.doc_id in seen:
                continue
            seen.add(passage.doc_id)
            yield passage
            count += 1
            if limit and count >= limit:
                return


def _to_passage(row: dict, categories: list[str]) -> Passage | None:
    paper_id = str(row.get("id") or row.get("paper_id") or "").strip()
    title = " ".join((row.get("title") or "").split())
    abstract = " ".join((row.get("abstract") or "").split())
    if not paper_id or not title or not abstract:
        return None
    versions = row.get("versions") or []
    published = (
        versions[0].get("created", "") if versions and isinstance(versions[0], dict) else row.get("published", "")
    )
    year = _year(row.get("update_date") or "", published)
    authors = " ".join((row.get("authors") or "").split())
    return Passage(
        id=f"arxiv:{paper_id}",
        doc_id=paper_id,
        title=title,
        text=abstract,
        source="arXiv",
        url=f"https://arxiv.org/abs/{paper_id}" if not paper_id.startswith("sample") else "",
        metadata={
            "authors": authors[:300],
            "categories": " ".join(categories),
            "primary_category": categories[0] if categories else "",
            "year": year,
            "published": published,
        },
    )


def _year(update_date: str, published: str) -> int | None:
    for value in (published[-17:-13] if published else "", update_date[:4]):
        if value.isdigit():
            return int(value)
    return None
