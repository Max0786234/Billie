"""Build Billie's search indexes.

    python -m billie.ingest status
    python -m billie.ingest knowledge
    python -m billie.ingest medquad [--no-embed]
    python -m billie.ingest arxiv [--limit N] [--embed-limit N] [--embed-since YEAR] [--no-embed]

Embedding runs are checkpointed: if interrupted (Ctrl+C, power loss), rerun the
same command with ``--embed-only`` and it continues where it stopped.
"""

from __future__ import annotations

import argparse
import sys
import time

from billie.corpora.arxiv import iter_arxiv
from billie.corpora.medquad import iter_medquad
from billie.retrieval.index import build_index, embed_index
from billie.runtime import build_corpus_service, load_settings


def _progress(label: str):
    started = time.perf_counter()
    last = [0.0]

    def report(done: int, total: int) -> None:
        now = time.perf_counter()
        if now - last[0] < 2 and done != total:
            return
        last[0] = now
        elapsed = now - started
        if total:
            rate = done / elapsed if elapsed else 0
            eta = (total - done) / rate if rate else 0
            print(f"\r{label}: {done:,}/{total:,} ({done / total:.0%}) {rate:,.0f}/s eta {eta / 60:.1f} min", end="")
        else:
            print(f"\r{label}: {done:,} ({elapsed:.0f}s)", end="")
        sys.stdout.flush()

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m billie.ingest", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show index status")
    sub.add_parser("knowledge", help="rebuild the general knowledge index")
    med = sub.add_parser("medquad", help="index the full MedQuAD corpus")
    med.add_argument("--no-embed", action="store_true")
    med.add_argument("--embed-only", action="store_true", help="skip the text index rebuild; (re)embed only")
    med.add_argument("--embed-chars", type=int, default=400, help="characters of each passage to embed")
    arx = sub.add_parser("arxiv", help="index computer-science papers from the arXiv snapshot")
    arx.add_argument("--limit", type=int, default=None, help="max papers to index (default: all CS papers)")
    arx.add_argument("--embed-limit", type=int, default=20000, help="max papers to embed (dense search subset)")
    arx.add_argument("--embed-chars", type=int, default=400, help="characters of each abstract to embed")
    arx.add_argument("--embed-since", type=int, default=2022, help="embed only papers from this year onward")
    arx.add_argument("--no-embed", action="store_true")
    arx.add_argument("--embed-only", action="store_true")
    args = parser.parse_args(argv)

    settings = load_settings()
    service = build_corpus_service(settings)
    paths = settings.paths

    if args.command == "status":
        for status in service.status():
            print(
                f"{status.name:10} {status.variant:9} passages={status.passages:>9,} vectors={status.vectors:>7,} "
                f"search={status.search_mode:6} model={status.embedding_model or '-'} built={status.built_at or '-'}"
            )
        return 0

    if args.command == "knowledge":
        rebuilt = service.sync_knowledge(force=True)
        print("knowledge index rebuilt" if rebuilt else "knowledge index up to date")
        return 0

    embedder = service.embedder
    index_dir = paths.index_dir

    if args.command == "medquad":
        if not paths.medquad_dir.exists():
            print(f"MedQuAD not found at {paths.medquad_dir}. See datasets/README.md.")
            return 1
        if not args.embed_only:
            count = build_index(
                index_dir,
                "medquad",
                iter_medquad(paths.medquad_dir),
                meta={"corpus": "medquad", "variant": "full"},
                progress=_progress("indexing MedQuAD"),
            )
            print(f"\nindexed {count:,} MedQuAD passages")
        if not args.no_embed and embedder is not None:
            # First chunk of each answer (+ its question as title) identifies the topic; BM25 covers the rest.
            n = embed_index(
                index_dir,
                "medquad",
                embedder,
                where="json_extract(metadata, '$.chunk') = 0",
                max_chars=args.embed_chars,
                progress=_progress(f"embedding ({embedder.model})"),
            )
            print(f"\nembedded {n:,} passages")
        return 0

    if args.command == "arxiv":
        if not paths.arxiv_snapshot.exists():
            print(f"arXiv snapshot not found at {paths.arxiv_snapshot}. See datasets/README.md.")
            return 1
        if not args.embed_only:
            count = build_index(
                index_dir,
                "arxiv",
                iter_arxiv(paths.arxiv_snapshot, limit=args.limit),
                meta={"corpus": "arxiv", "variant": "full"},
                progress=_progress("indexing arXiv CS papers"),
            )
            print(f"\nindexed {count:,} arXiv CS papers")
        if not args.no_embed and embedder is not None:
            where = f"CAST(json_extract(metadata, '$.year') AS INTEGER) >= {int(args.embed_since)}"
            n = embed_index(
                index_dir,
                "arxiv",
                embedder,
                where=where,
                limit=args.embed_limit,
                newest_first=True,
                max_chars=args.embed_chars,
                progress=_progress(f"embedding ({embedder.model})"),
            )
            print(f"\nembedded {n:,} papers (since {args.embed_since}, limit {args.embed_limit:,})")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
