"""Shared helpers for evaluation scripts."""

from __future__ import annotations

from pathlib import Path

from billie.llm.manager import LLMManager


def eval_llm(tmp: Path, use_app_model: bool, app_settings_path: Path) -> LLMManager:
    """Evaluations run on the local Ollama model by default, so they never spend cloud quota
    (e.g. Groq's free tier). ``--use-app-model`` opts into whatever the app has configured."""
    if use_app_model:
        return LLMManager(app_settings_path)
    return LLMManager(tmp / "eval_llm_settings.json", env={})  # defaults: Ollama qwen3:4b-instruct
