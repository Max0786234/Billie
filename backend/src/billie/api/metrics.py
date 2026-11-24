"""Prometheus metrics. Each app instance owns its registry (keeps tests isolated)."""

from __future__ import annotations

from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest

_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 80, 160)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.http_requests = Counter(
            "billie_http_requests_total", "HTTP requests", ["method", "route", "status"], registry=self.registry
        )
        self.http_latency = Histogram(
            "billie_http_request_seconds", "HTTP latency", ["route"], buckets=_LATENCY_BUCKETS, registry=self.registry
        )
        self.chats = Counter("billie_chat_turns_total", "Chat turns", ["mode", "language"], registry=self.registry)
        self.llm_fallbacks = Counter(
            "billie_llm_fallbacks_total", "Answers served without the LLM", ["mode"], registry=self.registry
        )
        self.ungrounded = Counter(
            "billie_ungrounded_answers_total", "Answers without valid citations", ["mode"], registry=self.registry
        )
        self.feedback = Counter(
            "billie_answer_feedback_total",
            "Thumbs up (1) / down (-1) / cleared (0)",
            ["language", "value"],
            registry=self.registry,
        )
        self.stage_latency = Histogram(
            "billie_chat_stage_seconds",
            "Chat stage latency",
            ["stage"],
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )

    def observe_turn(self, mode: str, done: dict) -> None:
        self.chats.labels(mode, done.get("analysis", {}).get("language", "unknown")).inc()
        if not done.get("used_llm"):
            self.llm_fallbacks.labels(mode).inc()
        if done.get("grounded") is False:
            self.ungrounded.labels(mode).inc()
        for stage, ms in (done.get("timings_ms") or {}).items():
            self.stage_latency.labels(stage.removesuffix("_ms")).observe(ms / 1000)

    def render(self) -> Response:
        return Response(generate_latest(self.registry), media_type=CONTENT_TYPE_LATEST)
