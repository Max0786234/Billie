"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["general", "medical", "research"]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    mode: Mode = "general"
    conversation_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    image_ids: list[str] = Field(default_factory=list, max_length=4)
    regenerate: bool = False


class FeedbackRequest(BaseModel):
    value: Literal[1, -1, 0]  # 0 clears


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class UploadedImage(BaseModel):
    id: str
    media_type: str
    size: int


class DocumentInfo(BaseModel):
    name: str
    size: int
    modified: float


class LLMTestRequest(BaseModel):
    """Optional overrides to test before saving; omitted fields use the saved settings."""

    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
