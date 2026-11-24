"""Typed runtime settings (from environment / ``.env``) and service construction.

All configuration lives here; nothing else reads ``os.environ``. See
``.env.example`` at the repository root for every variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from billie import paths as P
from billie.corpora.service import CorpusService
from billie.retrieval.embeddings import OllamaEmbedder

Environment = Literal["development", "production"]


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    knowledge_dir: Path
    data_dir: Path
    index_dir: Path
    uploads_dir: Path
    doc_uploads_dir: Path
    image_uploads_dir: Path
    sessions_db: Path
    llm_settings: Path
    medquad_dir: Path
    medquad_sample: Path
    arxiv_snapshot: Path
    arxiv_sample: Path

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> Paths:
        uploads = data_dir / "uploads"
        return cls(
            repo_root=P.REPO_ROOT,
            knowledge_dir=P.KNOWLEDGE_DIR,
            data_dir=data_dir,
            index_dir=data_dir / "index",
            uploads_dir=uploads,
            doc_uploads_dir=uploads / "docs",
            image_uploads_dir=uploads / "images",
            sessions_db=data_dir / "sessions.sqlite",
            llm_settings=data_dir / "llm_settings.json",
            medquad_dir=P.MEDQUAD_DIR,
            medquad_sample=P.MEDQUAD_SAMPLE,
            arxiv_snapshot=P.ARXIV_SNAPSHOT,
            arxiv_sample=P.ARXIV_SAMPLE,
        )


@dataclass(frozen=True)
class Settings:
    paths: Paths
    environment: Environment = "development"
    api_key: str | None = None
    admin_key: str | None = None
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
    ollama_url: str = "http://127.0.0.1:11434"
    embedding_model: str | None = "embeddinggemma"
    history_turns: int = 3
    max_image_mb: int = 8
    max_document_mb: int = 20
    rate_limit_per_minute: int = 30
    # Public demo mode (e.g. a Hugging Face Space): visitors chat without keys, each sees only
    # their own conversations, and daily caps protect the free cloud-model quota.
    demo: bool = False
    demo_daily_limit: int = 150
    demo_visitor_daily_limit: int = 25
    # Sites allowed to embed the UI in an iframe (the Space page frames the app); empty = none.
    frame_ancestors: tuple[str, ...] = ()
    top_k: dict[str, int] = field(default_factory=lambda: {"general": 4, "medical": 4, "research": 5})

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


def load_settings(env: dict[str, str] | None = None, *, data_dir: Path | None = None) -> Settings:
    if env is None:
        load_dotenv(P.ENV_FILE)
        env = dict(os.environ)
    environment = env.get("BILLIE_ENV", "development").lower()
    if environment not in ("development", "production"):
        raise ValueError(f"BILLIE_ENV must be 'development' or 'production', got {environment!r}")
    embed = env.get("BILLIE_EMBED_MODEL", "embeddinggemma").strip()
    origins = env.get("BILLIE_CORS_ORIGINS")
    settings = Settings(
        paths=Paths.from_data_dir(data_dir or Path(env.get("BILLIE_DATA_DIR", P.DATA_DIR))),
        environment=environment,  # type: ignore[arg-type]
        api_key=env.get("BILLIE_API_KEY") or None,
        admin_key=env.get("BILLIE_ADMIN_KEY") or None,
        cors_origins=tuple(o.strip() for o in origins.split(",")) if origins else Settings.cors_origins,
        ollama_url=env.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        embedding_model=None if embed.lower() in ("", "off", "none") else embed,
        history_turns=int(env.get("BILLIE_HISTORY_TURNS", "3")),
        max_image_mb=int(env.get("BILLIE_MAX_IMAGE_MB", "8")),
        max_document_mb=int(env.get("BILLIE_MAX_DOCUMENT_MB", "20")),
        rate_limit_per_minute=int(env.get("BILLIE_RATE_LIMIT_PER_MINUTE", "30")),
        demo=_flag(env.get("BILLIE_DEMO")),
        demo_daily_limit=int(env.get("BILLIE_DEMO_DAILY_LIMIT", "150")),
        demo_visitor_daily_limit=int(env.get("BILLIE_DEMO_VISITOR_DAILY_LIMIT", "25")),
        frame_ancestors=tuple(a.strip() for a in env.get("BILLIE_FRAME_ANCESTORS", "").split(",") if a.strip()),
    )
    if settings.is_production and not settings.admin_key:
        # Fail closed: settings and the knowledge base must never be open to anyone.
        raise RuntimeError("BILLIE_ENV=production requires BILLIE_ADMIN_KEY to be set.")
    if settings.is_production and not settings.demo and not settings.api_key:
        # A private deployment also needs a chat key; a public demo is open on purpose (with quotas).
        raise RuntimeError("BILLIE_ENV=production requires BILLIE_API_KEY (or BILLIE_DEMO=1 for a public demo).")
    return settings


def _flag(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def build_corpus_service(settings: Settings) -> CorpusService:
    embedder = OllamaEmbedder(settings.embedding_model, settings.ollama_url) if settings.embedding_model else None
    return CorpusService(
        settings.paths.index_dir,
        knowledge_dirs=[settings.paths.knowledge_dir, settings.paths.doc_uploads_dir],
        repo_root=settings.paths.repo_root,
        medquad_sample=settings.paths.medquad_sample,
        arxiv_sample=settings.paths.arxiv_sample,
        embedder=embedder,
    )
