from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    dataset_id: str
    message: str
    top_k: int = Field(default=5, ge=1, le=50)
    doc_types: list[str] | None = None
    response_format: str = "markdown"


class ChatCitation(BaseModel):
    citation: str
    doc_type: str | None
    row_start: int | None
    row_end: int | None
    score: float | None


class ChatContext(BaseModel):
    text: str
    source: dict[str, Any]
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[ChatCitation]
    contexts: list[ChatContext]
    latency_ms: int
    prompt_tokens: int | None = None
    response_tokens: int | None = None
