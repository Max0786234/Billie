"""Dependency container: every long-lived object the API needs, built once."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from billie.analysis import MessageAnalyzer, ModelAnalyzer
from billie.api.metrics import Metrics
from billie.chat.cache import AnswerCache
from billie.chat.pipeline import ChatPipeline
from billie.conversations import ConversationStore
from billie.corpora.service import CorpusService
from billie.llm.base import GenerationOptions, LLMError, Message, complete
from billie.llm.manager import LLMManager
from billie.runtime import Settings, build_corpus_service
from billie.uploads import DocumentStore, ImageStore


@dataclass
class Services:
    settings: Settings
    llm: LLMManager
    corpora: CorpusService
    conversations: ConversationStore
    images: ImageStore
    documents: DocumentStore
    pipeline: ChatPipeline
    metrics: Metrics
    cache: AnswerCache | None = None


def build_services(
    settings: Settings,
    *,
    corpora: CorpusService | None = None,
    llm: LLMManager | None = None,
    analyzer: MessageAnalyzer | None = None,
) -> Services:
    paths = settings.paths
    llm = llm or LLMManager(paths.llm_settings)
    corpora = corpora or build_corpus_service(settings)
    conversations = ConversationStore(paths.sessions_db)
    images = ImageStore(paths.image_uploads_dir, settings.max_image_mb * 1024 * 1024)
    documents = DocumentStore(paths.doc_uploads_dir, settings.max_document_mb * 1024 * 1024)
    cache = AnswerCache(paths.data_dir / "answer_cache.sqlite")
    pipeline = ChatPipeline(
        llm=llm,
        corpora=corpora,
        conversations=conversations,
        analyzer=analyzer or ModelAnalyzer.from_embedder(corpora.embedder),
        images=images,
        history_turns=settings.history_turns,
        top_k=settings.top_k,
        cache=cache,
    )
    return Services(settings, llm, corpora, conversations, images, documents, pipeline, Metrics(), cache)


def warm_up(services: Services) -> None:
    """Load the chat and embedding models into memory so the first question doesn't pay for it.

    Uses the same generation options as real requests: Ollama reloads a model when
    ``num_ctx`` changes, which would defeat the purpose.
    """
    log = logging.getLogger("billie")
    try:
        provider = services.llm.provider()
        if provider.name == "ollama":
            complete(provider, [Message("user", "hi")], GenerationOptions(max_tokens=1))
    except LLMError as exc:
        log.info("chat model warm-up skipped: %s", exc)
    embedder = services.corpora.embedder
    if embedder is not None:
        try:
            embedder.embed_queries(["warm up"])
        except LLMError as exc:
            log.info("embedding warm-up skipped: %s", exc)
