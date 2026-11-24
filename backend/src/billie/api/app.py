"""FastAPI application.

Run (development):  uvicorn billie.api.app:create_app --factory --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

import anyio
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import iterate_in_threadpool

from billie import paths as P
from billie.api.auth import require_admin_key, require_api_key
from billie.api.logging_setup import configure_logging, request_id_var
from billie.api.middleware import MetricsMiddleware, RequestIdMiddleware, SecurityMiddleware
from billie.api.schemas import (
    ChatRequest,
    DocumentInfo,
    FeedbackRequest,
    LLMTestRequest,
    RenameRequest,
    UploadedImage,
)
from billie.api.security import TokenBucketLimiter
from billie.api.services import Services, build_services, warm_up
from billie.chat.pipeline import ChatTurn
from billie.conversations import ConversationNotFound, MessageNotFound
from billie.llm.base import LLMError
from billie.llm.manager import LLMManager, LLMSettingsUpdate
from billie.runtime import Settings, load_settings
from billie.uploads import UploadError

log = logging.getLogger("billie")

FRONTEND_DIST = P.REPO_ROOT / "frontend" / "dist"


def create_app(settings: Settings | None = None, *, services: Services | None = None, warm: bool = True) -> FastAPI:
    settings = settings or load_settings()
    configure_logging()
    container = services or build_services(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = container
        # Indexing (and embedding new documents) can take a while; do not block startup.
        app.state.ready = threading.Event()

        def prepare() -> None:
            try:
                container.corpora.prepare()
            except Exception:  # pragma: no cover - logged, surfaced via /api/status
                log.exception("corpus preparation failed")
            finally:
                app.state.ready.set()
            if warm:
                warm_up(container)

        threading.Thread(target=prepare, name="corpus-prepare", daemon=True).start()
        yield

    app = FastAPI(
        title="Billie API",
        version="0.2.0",
        description="Retrieval-augmented multimodal assistant: general documents, MedQuAD medical Q&A and arXiv research.",
        lifespan=lifespan,
    )
    app.state.services = container
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Pure ASGI middleware (see api/middleware.py): BaseHTTPMiddleware would break client-disconnect
    # detection for streamed answers. Added last = outermost, so every log line gets the request ID.
    app.add_middleware(MetricsMiddleware, metrics=container.metrics)
    app.add_middleware(SecurityMiddleware, limiter=TokenBucketLimiter(settings.rate_limit_per_minute))
    app.add_middleware(RequestIdMiddleware)

    def services_of(request: Request) -> Services:
        return request.app.state.services

    # ------------------------------------------------------------- health
    @app.get("/api/health", tags=["ops"])
    def health() -> dict[str, str]:
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/api/status", tags=["ops"])
    def readiness(request: Request) -> dict[str, Any]:
        """Readiness: model reachable, indexes built, embeddings available."""
        s = services_of(request)
        try:
            llm_health = asdict(s.llm.provider().health())
        except LLMError as exc:
            llm_health = {
                "ok": False,
                "provider": s.llm.settings.provider,
                "model": s.llm.settings.model,
                "detail": str(exc),
            }
        embedder = s.corpora.embedder
        ready_event = getattr(request.app.state, "ready", None)
        return {
            "indexing": bool(ready_event and not ready_event.is_set()),
            "llm": llm_health,
            "embedding": {
                "model": getattr(embedder, "model", None),
                "available": bool(embedder and getattr(embedder, "available", lambda: True)()),
            },
            "corpora": [asdict(c) for c in s.corpora.status()],
            "cache": s.cache.stats() if s.cache else None,
            "feedback": s.conversations.feedback_stats(),
            "environment": s.settings.environment,
            "auth_required": bool(s.settings.api_key),
        }

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    def metrics(request: Request):
        return services_of(request).metrics.render()

    # --------------------------------------------------------------- chat
    @app.post("/api/chat", tags=["chat"], dependencies=[Depends(require_api_key)])
    def chat(request: Request, body: ChatRequest, stream: bool = True):
        """Answer a question. Streams Server-Sent Events by default; ``?stream=false`` returns JSON."""
        s = services_of(request)
        for image_id in body.image_ids:  # validate up front so errors are a clean 400, not a broken stream
            try:
                s.images.load(image_id)
            except UploadError as exc:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        turn = ChatTurn(body.question, body.mode, body.conversation_id, body.image_ids, body.regenerate)
        request_id = request_id_var.get()  # the streamed body runs after the middleware returns

        def events() -> Iterator[dict[str, Any]]:
            for event in s.pipeline.run(turn):
                if event["event"] == "done":
                    done = event["data"]
                    s.metrics.observe_turn(body.mode, done)
                    log.info("chat_turn", extra={"fields": _turn_log_fields(request_id, body.mode, done)})
                yield event

        if not stream:
            final: dict[str, Any] = {}
            meta: dict[str, Any] = {}
            for event in events():
                if event["event"] == "meta":
                    meta = event["data"]
                elif event["event"] == "done":
                    final = event["data"]
            return {**meta, **final}
        return StreamingResponse(
            stream_and_close(_sse(events())),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------ conversations
    @app.get("/api/conversations", tags=["conversations"], dependencies=[Depends(require_api_key)])
    def list_conversations(request: Request, limit: int = 50):
        return [asdict(c) for c in services_of(request).conversations.recent(min(limit, 200))]

    @app.get("/api/conversations/{conversation_id}", tags=["conversations"], dependencies=[Depends(require_api_key)])
    def get_conversation(request: Request, conversation_id: str):
        try:
            summary, messages = services_of(request).conversations.get(conversation_id)
        except ConversationNotFound as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found.") from exc
        return {**asdict(summary), "messages": [asdict(m) for m in messages]}

    @app.patch("/api/conversations/{conversation_id}", tags=["conversations"], dependencies=[Depends(require_api_key)])
    def rename_conversation(request: Request, conversation_id: str, body: RenameRequest):
        try:
            services_of(request).conversations.rename(conversation_id, body.title)
        except ConversationNotFound as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found.") from exc
        return {"ok": True}

    @app.delete("/api/conversations/{conversation_id}", tags=["conversations"], dependencies=[Depends(require_api_key)])
    def delete_conversation(request: Request, conversation_id: str):
        try:
            services_of(request).conversations.delete(conversation_id)
        except ConversationNotFound as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found.") from exc
        return {"ok": True}

    @app.put("/api/messages/{message_id}/feedback", tags=["conversations"], dependencies=[Depends(require_api_key)])
    def message_feedback(request: Request, message_id: int, body: FeedbackRequest):
        """Thumbs up/down on an answer. A thumbs-down also evicts the answer from the cache."""
        s = services_of(request)
        try:
            meta = s.conversations.set_feedback(message_id, body.value or None)
        except MessageNotFound as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Message not found.") from exc
        if body.value == -1 and s.cache and meta.get("cache_key"):
            s.cache.forget(meta["cache_key"])
        s.metrics.feedback.labels(meta.get("analysis", {}).get("language", "unknown"), str(body.value)).inc()
        return {"ok": True}

    # ------------------------------------------------------------ uploads
    @app.post(
        "/api/uploads/images", tags=["uploads"], response_model=UploadedImage, dependencies=[Depends(require_api_key)]
    )
    async def upload_image(request: Request, file: UploadFile = File(...)):
        s = services_of(request)
        data = await file.read(s.images.max_bytes + 1)
        try:
            stored = s.images.save(data)
        except UploadError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        return UploadedImage(id=stored.id, media_type=stored.media_type, size=stored.size)

    @app.get("/api/uploads/images/{image_id}", tags=["uploads"], dependencies=[Depends(require_api_key)])
    def get_image(request: Request, image_id: str):
        s = services_of(request)
        try:
            s.images.load(image_id)  # validates the id format and existence
        except UploadError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        return FileResponse(s.images.directory / image_id)

    # ---------------------------------------------------------- documents
    @app.get(
        "/api/documents", tags=["documents"], response_model=list[DocumentInfo], dependencies=[Depends(require_api_key)]
    )
    def list_documents(request: Request):
        return [DocumentInfo(**asdict(d)) for d in services_of(request).documents.list()]

    @app.post(
        "/api/documents", tags=["documents"], response_model=DocumentInfo, dependencies=[Depends(require_admin_key)]
    )
    async def upload_document(request: Request, file: UploadFile = File(...)):
        s = services_of(request)
        data = await file.read(s.documents.max_bytes + 1)
        try:
            stored = s.documents.save(file.filename or "document.txt", data)
        except UploadError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        try:
            s.corpora.sync_knowledge()
        except Exception as exc:  # e.g. an unreadable PDF
            s.documents.delete(stored.name)
            s.corpora.sync_knowledge()
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Could not index this document: {exc}") from exc
        return DocumentInfo(**asdict(stored))

    @app.delete("/api/documents/{name}", tags=["documents"], dependencies=[Depends(require_admin_key)])
    def delete_document(request: Request, name: str):
        s = services_of(request)
        try:
            s.documents.delete(name)
        except UploadError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        s.corpora.sync_knowledge()
        return {"ok": True}

    # ----------------------------------------------------------- settings
    @app.get("/api/settings/llm", tags=["settings"], dependencies=[Depends(require_api_key)])
    def get_llm_settings(request: Request):
        return services_of(request).llm.public_settings()

    @app.put("/api/settings/llm", tags=["settings"], dependencies=[Depends(require_admin_key)])
    def update_llm_settings(request: Request, body: LLMSettingsUpdate):
        try:
            return services_of(request).llm.update(body)
        except LLMError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    @app.get("/api/llm/presets", tags=["settings"])
    def llm_presets():
        return LLMManager.presets()

    @app.get("/api/llm/models", tags=["settings"], dependencies=[Depends(require_api_key)])
    def llm_models(request: Request):
        """Models offered by the currently configured provider (e.g. models pulled in Ollama)."""
        try:
            return {"models": services_of(request).llm.provider().list_models()}
        except LLMError as exc:
            return JSONResponse({"models": [], "error": str(exc)}, status_code=200)

    @app.post("/api/llm/test", tags=["settings"], dependencies=[Depends(require_admin_key)])
    def llm_test(request: Request, body: LLMTestRequest):
        """Check a provider/model/key combination without saving it."""
        manager = services_of(request).llm
        try:
            provider = manager.preview(body.provider, body.model, body.base_url, body.api_key)
            health = provider.health()
        except LLMError as exc:
            return {"ok": False, "detail": str(exc)}
        return asdict(health)

    # ------------------------------------------------------------ frontend
    if FRONTEND_DIST.exists():  # production build served by the API itself (single process)
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith(("api/", "metrics")):  # unknown API routes are real 404s, not the SPA
                raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found.")
            candidate = (FRONTEND_DIST / path).resolve()
            if path and candidate.is_file() and candidate.is_relative_to(FRONTEND_DIST.resolve()):
                return FileResponse(candidate)
            return FileResponse(FRONTEND_DIST / "index.html")

    return app


def _turn_log_fields(request_id: str, mode: str, done: dict[str, Any]) -> dict[str, Any]:
    """One structured line per answer - enough to debug quality and latency, no user text."""
    return {
        "request_id": request_id,
        "mode": mode,
        "language": (done.get("analysis") or {}).get("language"),
        "grounded": done.get("grounded"),
        "abstained": done.get("abstained"),
        "cached": done.get("cached"),
        "used_llm": done.get("used_llm"),
        "model": done.get("model"),
        "sources_cited": len(done.get("citations") or []),
        "alert": bool(done.get("alert")),
        "timings_ms": done.get("timings_ms"),
    }


async def stream_and_close(chunks: Iterator[str]) -> AsyncIterator[str]:
    """Stream a sync generator and *always* close it, including on client disconnect.

    When the browser disconnects (Stop button, closed tab), Starlette cancels the response
    but leaves the sync generator suspended - still holding its HTTP stream to Ollama,
    which keeps generating and blocks the next question (measured: a 0.4 s request waited
    21 s). Closing the generator raises GeneratorExit at its paused ``yield``; that unwinds
    the pipeline and provider, closes the stream, and Ollama stops at once.
    """
    completed = False
    try:
        async for chunk in iterate_in_threadpool(chunks):
            yield chunk
        completed = True
    finally:
        close = getattr(chunks, "close", None)
        if close is not None:
            with anyio.CancelScope(shield=True):  # the response is being cancelled; finish cleanup anyway
                await anyio.to_thread.run_sync(close)
        if not completed:
            log.info("stream closed early (client disconnected); generation stopped")


def _sse(events: Iterator[dict[str, Any]]) -> Iterator[str]:
    try:
        for event in events:
            yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
    except Exception as exc:  # surface failures to the client instead of a silently truncated stream
        log.exception("chat stream failed")
        yield f"event: error\ndata: {json.dumps({'message': f'Internal error: {type(exc).__name__}'})}\n\n"
