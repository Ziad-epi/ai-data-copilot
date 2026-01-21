from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status

from app.core.config import settings
from app.llm.provider import get_llm_client
from app.schemas.chat import ChatCitation, ChatContext, ChatRequest, ChatResponse
from app.services.datasets import load_dataset
from app.services.rag import search_dataset

SYSTEM_PROMPT = (
    "You are a data assistant for dataset Q&A. "
    "Use only the provided context. Do not make up information. "
    "If the answer is not in the context, say you do not have enough information. "
    "Always cite sources using the provided citation strings."
)


def _index_metadata_path(dataset_id: str) -> Path:
    return Path(settings.storage_dir) / dataset_id / "index_metadata.json"


def _build_prompt(
    message: str,
    contexts: list[dict[str, Any]],
    response_format: str,
) -> list[dict[str, str]]:
    summary_chunks: list[str] = []
    row_chunks: list[str] = []

    for item in contexts:
        source = item.get("source") or {}
        doc_type = source.get("doc_type")
        citation = item.get("citation", "unknown")
        text = item.get("text", "")
        block = f"Source: {citation}\n{text}"
        if doc_type == "summary":
            summary_chunks.append(block)
        else:
            row_chunks.append(block)

    prompt_parts = [
        f"Question: {message}",
        f"Response format: {response_format}",
        "Context:",
    ]

    if summary_chunks:
        prompt_parts.append("Dataset summary:")
        prompt_parts.extend(summary_chunks)

    if row_chunks:
        prompt_parts.append("Rows context:")
        prompt_parts.extend(row_chunks)

    prompt_parts.append(
        "Answer clearly and concisely. Use citations inline like [dataset:...]."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(prompt_parts)},
    ]


def _build_citations(contexts: list[dict[str, Any]]) -> list[ChatCitation]:
    citations: list[ChatCitation] = []
    for item in contexts:
        source = item.get("source") or {}
        citations.append(
            ChatCitation(
                citation=item.get("citation", ""),
                doc_type=source.get("doc_type"),
                row_start=source.get("row_start"),
                row_end=source.get("row_end"),
                score=item.get("score"),
            )
        )
    return citations


def _build_contexts(contexts: list[dict[str, Any]]) -> list[ChatContext]:
    return [
        ChatContext(
            text=item.get("text", ""),
            source=item.get("source") or {},
            score=float(item.get("score", 0.0)),
        )
        for item in contexts
    ]


def chat_with_dataset(payload: ChatRequest) -> ChatResponse:
    logger = logging.getLogger("ai-data-copilot")
    request_start = time.perf_counter()

    load_dataset(payload.dataset_id)
    if not _index_metadata_path(payload.dataset_id).exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset not indexed.",
        )

    if not settings.llm_api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="LLM not configured",
        )

    retrieval_start = time.perf_counter()
    contexts = search_dataset(
        dataset_id=payload.dataset_id,
        query=payload.message,
        top_k=payload.top_k,
        doc_types=payload.doc_types,
    )
    retrieval_ms = int((time.perf_counter() - retrieval_start) * 1000)

    if not contexts:
        latency_ms = int((time.perf_counter() - request_start) * 1000)
        logger.info(
            "chat dataset_id=%s top_k=%s retrieval_ms=%s llm_ms=%s provider=%s model=%s",
            payload.dataset_id,
            payload.top_k,
            retrieval_ms,
            0,
            settings.llm_provider,
            settings.llm_model,
        )
        return ChatResponse(
            answer="Je n'ai pas assez d'infos dans le contexte fourni.",
            citations=[],
            contexts=[],
            latency_ms=latency_ms,
            prompt_tokens=None,
            response_tokens=None,
        )

    messages = _build_prompt(
        message=payload.message,
        contexts=contexts,
        response_format=payload.response_format,
    )
    llm_client = get_llm_client()
    llm_start = time.perf_counter()
    llm_response = llm_client.generate(messages)
    llm_ms = int((time.perf_counter() - llm_start) * 1000)
    latency_ms = int((time.perf_counter() - request_start) * 1000)

    logger.info(
        "chat dataset_id=%s top_k=%s retrieval_ms=%s llm_ms=%s provider=%s model=%s",
        payload.dataset_id,
        payload.top_k,
        retrieval_ms,
        llm_ms,
        settings.llm_provider,
        settings.llm_model,
    )

    return ChatResponse(
        answer=llm_response.text,
        citations=_build_citations(contexts),
        contexts=_build_contexts(contexts),
        latency_ms=latency_ms,
        prompt_tokens=llm_response.prompt_tokens,
        response_tokens=llm_response.response_tokens,
    )
