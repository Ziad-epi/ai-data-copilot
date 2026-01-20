from typing import Any

from pydantic import BaseModel, Field


class DatasetIndexRequest(BaseModel):
    columns: list[str] | None = None
    max_rows: int | None = None
    rows_per_doc: int | None = None
    reindex: bool = False


class DatasetIndexResponse(BaseModel):
    dataset_id: str
    nb_docs: int
    vectors_upserted: int
    duration_ms: int


class DatasetSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    doc_types: list[str] | None = None


class DatasetSearchResult(BaseModel):
    score: float
    text: str
    source: dict[str, Any]
    citation: str


class DatasetSearchResponse(BaseModel):
    dataset_id: str
    results: list[DatasetSearchResult]
