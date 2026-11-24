"""General-mode knowledge base: Markdown/text/PDF files -> passages.

Sources are the repository's ``knowledge/`` folder plus documents users upload
through the UI (``data/uploads/docs``). A content fingerprint over all files
decides whether the index needs rebuilding, so restarts are instant.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from billie.retrieval.base import Passage
from billie.retrieval.chunking import chunk_text

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$", re.MULTILINE)


def knowledge_files(directories: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for directory in directories:
        if directory.exists():
            files.extend(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    return sorted(files)


def fingerprint(files: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - pypdf is a core dependency
        raise RuntimeError("PDF support needs pypdf") from exc
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def iter_knowledge(files: Iterable[Path], root: Path, max_chars: int = 900) -> Iterator[Passage]:
    for path in files:
        text = read_document(path)
        rel = _relative(path, root)
        title_default = _title_from_name(path)
        for section_title, section_text in _sections(text, title_default):
            for n, chunk in enumerate(chunk_text(section_text, max_chars=max_chars, overlap_chars=150)):
                digest = hashlib.sha1(f"{rel}|{section_title}|{n}".encode()).hexdigest()[:12]
                yield Passage(
                    id=f"kb:{digest}",
                    doc_id=rel,
                    title=section_title,
                    text=chunk,
                    source=path.name,
                    metadata={"path": rel, "section": section_title},
                )


def _sections(text: str, default_title: str) -> list[tuple[str, str]]:
    """Split Markdown on headings so each chunk carries its section as a title."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [(default_title, text)]
    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    doc_title = matches[0].group(2) if matches[0].group(1) == "#" else default_title
    if preamble:
        sections.append((doc_title, preamble))
    for i, match in enumerate(matches):
        body = text[match.end() : matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()
        if body:
            heading = match.group(2).strip()
            title = heading if heading == doc_title else f"{doc_title} › {heading}"
            sections.append((title, body))
    return sections


def _title_from_name(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip().title()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name
