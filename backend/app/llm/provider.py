from fastapi import HTTPException, status

from app.core.config import settings
from app.llm.client import OpenAICompatibleClient


def get_llm_client() -> OpenAICompatibleClient:
    if settings.llm_provider != "openai_compatible":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported LLM provider: {settings.llm_provider}",
        )
    if not settings.llm_api_key or not settings.llm_base_url or not settings.llm_model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="LLM not configured",
        )
    return OpenAICompatibleClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
