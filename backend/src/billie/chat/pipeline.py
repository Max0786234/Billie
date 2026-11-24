"""The chat pipeline: one turn = analyse -> (read image) -> retrieve -> generate -> verify -> remember.

``ChatPipeline.run`` is a generator of events so the API can stream them as
Server-Sent Events and the UI can show progress while a CPU-bound local model works:

    status  {stage}                          retrieving / reading_image / generating
    meta    {conversation_id, sources, analysis, retrieval, ...}
    token   {text}                           incremental answer text
    done    {answer, citations, grounded, used_llm, model, timings_ms, ...}
    error   {message}                        unrecoverable request error
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

from billie.analysis import Analysis, MessageAnalyzer
from billie.chat.cache import AnswerCache, cache_key
from billie.chat.citations import repair_citations
from billie.chat.enrich import enrich
from billie.chat.not_found import is_small_talk, not_found_message
from billie.chat.prompts import (
    IMAGE_DESCRIBE_PROMPT,
    SYSTEM_PROMPTS,
    TRANSLATE_QUERY_PROMPT,
    build_messages,
    cited_numbers,
    split_image_description,
)
from billie.conversations import ConversationStore
from billie.corpora.service import CORPUS_FOR_MODE, CorpusService
from billie.llm.base import GenerationOptions, ImagePart, LLMError, Message, complete
from billie.llm.manager import LLMManager
from billie.nlp.safety import PROMPT_HINT, EmergencyAlert, detect_emergency
from billie.retrieval.base import SearchHit, SearchResult
from billie.uploads import ImageStore

log = logging.getLogger(__name__)

Mode = Literal["general", "medical", "research"]
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pipeline")
Event = dict[str, Any]

_FOLLOW_UP = re.compile(
    r"\b(it|its|they|them|their|this|that|these|those|he|she|his|her|the same|more|also|what about|and)\b", re.I
)


@dataclass
class ChatTurn:
    question: str
    mode: Mode = "general"
    conversation_id: str | None = None
    image_ids: list[str] = field(default_factory=list)
    # Replace the last question/answer pair instead of appending (regenerate / edit question).
    regenerate: bool = False
    owner: str | None = None  # public demo: the visitor this chat belongs to (None = local user)


def contextual_query(question: str, history: list[tuple[str, str]]) -> str:
    """Search query for a turn. Short or pronoun-heavy follow-ups borrow the previous question's topic."""
    if not history:
        return question
    if len(question.split()) <= 6 or _FOLLOW_UP.search(question):
        return f"{history[-1][0]} {question}"
    return question


def source_payload(n: int, hit: SearchHit) -> dict[str, Any]:
    p = hit.passage
    return {
        "n": n,
        "id": p.id,
        "title": p.title,
        "source": p.source,
        "url": p.url,
        "text": p.text,
        "score": round(hit.score, 5),
        "components": hit.components,
        "metadata": p.metadata,
    }


class ChatPipeline:
    def __init__(
        self,
        *,
        llm: LLMManager,
        corpora: CorpusService,
        conversations: ConversationStore,
        analyzer: MessageAnalyzer,
        images: ImageStore,
        history_turns: int = 3,
        top_k: dict[str, int] | None = None,
        cache: AnswerCache | None = None,
        citation_threshold: float = 0.60,
    ) -> None:
        self.llm = llm
        self.corpora = corpora
        self.conversations = conversations
        self.analyzer = analyzer
        self.images = images
        self.history_turns = history_turns
        self.top_k = top_k or {"general": 4, "medical": 4, "research": 5}
        self.cache = cache
        # Calibrated on judged baseline answers: at 0.60, 89% of added citations point to the right
        # passage (54% of true pairs recovered). Precision over recall: a wrong citation is worse than none.
        self.citation_threshold = citation_threshold

    def run(self, turn: ChatTurn) -> Iterator[Event]:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        question = turn.question.strip()

        conversation_id = turn.conversation_id
        if not conversation_id or not self.conversations.exists(conversation_id, owner=turn.owner):
            conversation_id = self.conversations.create(turn.mode, title=_title(question), owner=turn.owner)
        elif turn.regenerate:
            self.conversations.pop_last_turn(conversation_id)
        history = self.conversations.recent_turns(conversation_id, self.history_turns)

        analysis = self.analyzer.analyze(question)
        alert = detect_emergency(question)  # deterministic safety net, independent of the LLM
        images = [self.images.load(image_id) for image_id in turn.image_ids]

        # ---- answer cache: stand-alone text questions only ----------------------
        key = self._cache_key(turn, question) if self.cache and not history and not turn.image_ids else None
        if key and self.cache and not turn.regenerate:
            cached = self.cache.get(key)
            if cached is not None:
                yield from self._replay(cached, turn, question, conversation_id, analysis, alert, started)
                return

        # ---- image understanding -------------------------------------------------
        image_description: str | None = None
        image_terms = ""
        direct_images: tuple[ImagePart, ...] = ()
        notes: list[str] = []
        if images:
            yield _event("status", stage="reading_image")
            raw_description, direct_images, note = self._read_images(question, images, timings)
            if raw_description:
                image_description, image_terms = split_image_description(raw_description)
            if note:
                notes.append(note)

        # ---- retrieval ----------------------------------------------------------
        # ---- cross-lingual retrieval: corpora are English, so translate the search query ----
        search_question = question
        translated: str | None = None
        if analysis.language != "en":
            yield _event("status", stage="translating")
            translated = self._translate_query(question, timings, notes)
            search_question = translated or question

        yield _event("status", stage="retrieving")
        query = contextual_query(search_question, history)
        if image_description:
            # Explicit keywords from the vision model retrieve far better than its prose description.
            query = f"{query} {image_terms or image_description[:300]}"
        t = time.perf_counter()
        result: SearchResult = self.corpora.retriever(CORPUS_FOR_MODE[turn.mode]).search(
            query, top_k=self.top_k[turn.mode]
        )
        timings["retrieval_ms"] = _ms(t)
        hits = result.hits
        sources = [source_payload(n, hit) for n, hit in enumerate(hits, start=1)]

        enrichment = enrich(turn.mode, question, hits)
        meta_event: dict[str, Any] = dict(
            conversation_id=conversation_id,
            mode=turn.mode,
            question=question,
            analysis=_analysis_payload(analysis),
            sources=sources,
            retrieval={
                "query": query,
                "translated_query": translated,
                "mode": result.mode,
                "degraded": result.degraded,
                "note": result.note,
                "timings_ms": result.timings_ms,
            },
            image_description=image_description,
            enrichment=enrichment,
            alert=_alert_payload(alert),
            cached=False,
        )
        yield _event("meta", **meta_event)

        # Embed the sources for citation repair *while* the model writes: they are known now, and
        # the work would otherwise run after generation (the slowest step on a fast cloud model).
        embedder = self.corpora.embedder
        source_vectors_future = (
            _POOL.submit(embedder.embed_documents, [h.passage.text for h in hits])
            if hits and embedder is not None
            else None
        )

        # ---- generation ---------------------------------------------------------
        yield _event("status", stage="generating")
        messages = build_messages(
            mode=turn.mode,
            question=question,
            hits=hits,
            history=history,
            language_name=analysis.language_name,
            sentiment=analysis.sentiment,
            image_description=image_description,
            images=direct_images,
            interpreted_question=translated,
            safety_hint=PROMPT_HINT if alert.triggered else None,
        )
        provider = None
        chunks: list[str] = []
        used_llm = True
        llm_error: str | None = None
        t = time.perf_counter()
        # No evidence + not small talk + no emergency: a fixed reply, so the model can't answer from memory.
        not_found = not hits and not alert.triggered and not is_small_talk(question)
        if not_found:
            used_llm = False
            text = not_found_message(turn.mode, analysis.language, analysis.language_name)
            chunks.append(text)
            yield _event("token", text=text)
        try:
            if not_found:
                raise _Skip
            provider = self.llm.provider()
            settings = self.llm.settings
            options = GenerationOptions(temperature=settings.temperature, max_tokens=settings.max_tokens)
            for token in provider.stream(messages, options):
                if not chunks:
                    timings["first_token_ms"] = _ms(t)
                chunks.append(token)
                yield _event("token", text=token)
                if len(chunks) % 8 == 0 and is_looping("".join(chunks[-120:])):
                    notes.append("Stopped a repetitive answer early.")
                    break
        except _Skip:
            pass
        except LLMError as exc:
            log.warning("LLM unavailable, using extractive fallback: %s", exc)
            used_llm = False
            llm_error = str(exc)
            fallback = _extractive_answer(hits, llm_error) if not chunks else "\n\n_(The answer was cut off.)_"
            chunks.append(fallback)
            yield _event("token", text=fallback)
        timings["generation_ms"] = _ms(t)
        fallback_reason = getattr(provider, "fallback_reason", None)
        if fallback_reason:  # cloud model failed or was rate-limited; the local model answered
            notes.append(fallback_reason)
        answer = "".join(chunks).strip()

        # ---- citation repair: link uncited factual sentences to their source --------
        auto_citations: list[dict[str, Any]] = []
        unverified: list[str] = []
        if used_llm and hits and embedder is not None:
            t = time.perf_counter()
            try:
                source_vectors = source_vectors_future.result() if source_vectors_future else None
                repaired = repair_citations(
                    answer, [h.passage.text for h in hits], embedder, self.citation_threshold, source_vectors
                )
                answer, auto_citations, unverified = repaired.answer, repaired.added, repaired.unverified
            except LLMError as exc:
                notes.append(f"Citation repair skipped: {exc}")
            timings["citation_repair_ms"] = _ms(t)

        # ---- verification -------------------------------------------------------
        valid, invalid = cited_numbers(answer, len(hits))
        # grounded: True = cites only real sources; False = cites sources that do not exist;
        # None = not applicable, or abstained (no citations, e.g. "I could not find this").
        # A refusal can still cite the sources it describes ("the sources only cover X [1]"); stronger
        # models do this, and it is still a refusal, not a grounded answer (stress test, P2.16).
        abstained = bool(hits) and used_llm and not invalid and (not valid or opens_with_refusal(answer))
        grounded: bool | None = None if (not hits or not used_llm or abstained) else not invalid
        cited_sources = [sources[n - 1] for n in valid]

        # ---- memory ---------------------------------------------------------------
        timings["total_ms"] = _ms(started)
        meta = {
            "analysis": _analysis_payload(analysis),
            "grounded": grounded,
            "abstained": abstained,
            "citations": valid,
            "invalid_citations": invalid,
            "used_llm": used_llm,
            "provider": getattr(provider, "name", None),
            "model": getattr(provider, "model", None),
            "retrieval_mode": result.mode,
            "translated_query": translated,
            "timings_ms": timings,
            "image_ids": turn.image_ids,
            "image_description": image_description,
            "enrichment": enrichment,
            "alert": _alert_payload(alert),
            "notes": notes + ([f"LLM unavailable: {llm_error}"] if llm_error else []),
            "cache_key": key,
            "not_found": not_found,
            "auto_citations": auto_citations,
            "unverified_sentences": len(unverified),
        }
        self.conversations.append(conversation_id, "user", question, turn.mode, meta={"image_ids": turn.image_ids})
        message_id = self.conversations.append(
            conversation_id, "assistant", answer, turn.mode, sources=sources, meta=meta
        )
        done_event = dict(
            conversation_id=conversation_id,
            message_id=message_id,
            answer=answer,
            cited_sources=cited_sources,
            cached=False,
            **meta,
        )
        if key and self.cache and used_llm and grounded is not False and not notes:
            self.cache.put(key, meta_event, done_event)
        yield _event("done", **done_event)

    # ------------------------------------------------------------------ cache
    def _cache_key(self, turn: ChatTurn, question: str) -> str:
        settings = self.llm.settings
        info = self.corpora.index(CORPUS_FOR_MODE[turn.mode]).info()
        corpus_stamp = f"{info.meta.get('built_at')}|{info.vectors}|{info.meta.get('embedding_model')}"
        prompt_stamp = hashlib.sha1(SYSTEM_PROMPTS[turn.mode].encode()).hexdigest()[:12]
        return cache_key(turn.mode, question, f"{settings.provider}:{settings.model}", f"{corpus_stamp}|{prompt_stamp}")

    def _replay(
        self,
        cached: tuple[dict[str, Any], dict[str, Any]],
        turn: ChatTurn,
        question: str,
        conversation_id: str,
        analysis: Analysis,
        alert: EmergencyAlert,
        started: float,
    ) -> Iterator[Event]:
        meta_event, done_event = cached
        fresh = {"analysis": _analysis_payload(analysis), "alert": _alert_payload(alert)}
        yield _event("status", stage="retrieving")
        yield _event(
            "meta", **{**meta_event, **fresh, "conversation_id": conversation_id, "question": question, "cached": True}
        )
        answer = done_event["answer"]
        yield _event("token", text=answer)
        timings = {"total_ms": _ms(started)}
        stored = {
            k: v for k, v in done_event.items() if k not in ("conversation_id", "message_id", "answer", "cited_sources")
        }
        stored.update(fresh, timings_ms=timings, cached=True)
        self.conversations.append(conversation_id, "user", question, turn.mode, meta={"image_ids": []})
        message_id = self.conversations.append(
            conversation_id, "assistant", answer, turn.mode, sources=meta_event["sources"], meta=stored
        )
        yield _event(
            "done",
            **{
                **stored,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "answer": answer,
                "cited_sources": done_event.get("cited_sources", []),
            },
        )

    # ------------------------------------------------------------------ helpers
    def _translate_query(self, question: str, timings: dict[str, float], notes: list[str]) -> str | None:
        t = time.perf_counter()
        try:
            text = complete(
                self.llm.provider(),
                [Message("system", TRANSLATE_QUERY_PROMPT), Message("user", question)],
                GenerationOptions(temperature=0.0, max_tokens=80),
            )
        except LLMError as exc:
            notes.append(f"Query translation unavailable: {exc}")
            return None
        finally:
            timings["translation_ms"] = _ms(t)
        return " ".join(text.split())[:300] or None

    def _read_images(
        self, question: str, images: list[ImagePart], timings: dict[str, float]
    ) -> tuple[str | None, tuple[ImagePart, ...], str]:
        """Either hand images straight to a vision-capable chat model, or describe them with the vision model."""
        try:
            chat = self.llm.provider()
            vision = self.llm.vision_provider()
        except LLMError as exc:
            return None, (), f"Images ignored: {exc}"
        if vision is None:
            return None, (), "Images ignored: no vision-capable model is configured."
        t = time.perf_counter()
        try:
            description = complete(
                vision,
                [Message("user", IMAGE_DESCRIBE_PROMPT.format(question=question), images=tuple(images))],
                GenerationOptions(temperature=0.1, max_tokens=300),
            )
        except LLMError as exc:
            return None, (), f"Could not read the image: {exc}"
        finally:
            timings["vision_ms"] = _ms(t)
        # A vision-capable chat model also gets the pixels, so it can look again while answering.
        direct = tuple(images) if vision is chat else ()
        return description, direct, ""


class _Skip(Exception):
    """Internal: skip generation (deterministic reply already produced)."""


def is_looping(text: str, repeats: int = 5) -> bool:
    """True if the text ends with the same 3-60 character phrase repeated ``repeats`` times."""
    for period in range(3, 61):
        span = period * repeats
        if len(text) >= span and text[-span:] == text[-period:] * repeats:
            return True
    return False


def _extractive_answer(hits: list[SearchHit], error: str) -> str:
    if not hits:
        return f"I can't generate an answer right now because the language model is unavailable ({error})."
    lines = [f"_The language model is unavailable ({error}). The most relevant passages are:_", ""]
    for n, hit in enumerate(hits[:2], start=1):
        text = hit.passage.text
        snippet = text if len(text) <= 400 else text[:400].rsplit(" ", 1)[0] + " …"
        lines.append(f"- **{hit.passage.title}**: {snippet} [{n}]")
    return "\n".join(lines)


def _alert_payload(alert: EmergencyAlert) -> dict[str, Any] | None:
    return {"categories": alert.categories, "message": alert.message} if alert.triggered else None


def _analysis_payload(analysis: Analysis) -> dict[str, Any]:
    return {
        "language": analysis.language,
        "language_name": analysis.language_name,
        "language_confidence": round(analysis.language_confidence, 3),
        "sentiment": analysis.sentiment,
        "sentiment_score": round(analysis.sentiment_score, 3),
        "method": analysis.method,
    }


_REFUSAL_OPENER = re.compile(
    r"^\W*(i (could not|couldn't|cannot|can't|can not|am unable to|was unable to|did not|didn't) (find|provide|recommend|help|answer|give|write)"
    r"|i'm (unable|not able) to|i am not able to|unfortunately,? (i|the (provided )?(sources|documents))"
    r"|the (provided |available )?(sources|documents) (do not|don't|does not|doesn't)|none of the (provided )?(sources|documents)"
    r"|no encontr|je n'ai pas trouv|ich konnte keine|mujhe .{0,40}nahi? mil)",
    re.IGNORECASE,
)


def opens_with_refusal(answer: str) -> bool:
    """True if the answer's first sentence declines (not found / won't help)."""
    return bool(_REFUSAL_OPENER.search(answer.strip()[:200]))


def _title(question: str) -> str:
    title = " ".join(question.split())
    return title if len(title) <= 60 else title[:60].rsplit(" ", 1)[0] + "…"


def _event(name: str, **data: Any) -> Event:
    return {"event": name, "data": data}


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
