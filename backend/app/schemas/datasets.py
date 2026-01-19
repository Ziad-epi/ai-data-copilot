from typing import Any

from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    dataset_id: str
    filename: str
    file_size_bytes: int
    created_at: str
    nb_rows: int
    nb_columns: int
    columns: list[str]
    dtypes: dict[str, str]
    preview: list[dict[str, Any]]
    delimiter: str | None = None
    encoding: str | None = None
    missing_values_count: dict[str, int] = Field(default_factory=dict)
    numeric_columns_summary: dict[str, dict[str, float]] = Field(default_factory=dict)
    top_values: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    inferred_primary_key_candidate: str | None = None
    warnings: list[str] = Field(default_factory=list)


class DatasetSummary(BaseModel):
    dataset_id: str
    created_at: str
    filename: str
    nb_rows: int
    nb_columns: int


class DatasetSchema(BaseModel):
    dataset_id: str
    columns: list[str]
    dtypes: dict[str, str]
    missing_values_count: dict[str, int]
    numeric_columns_summary: dict[str, dict[str, float]]
    top_values: dict[str, list[dict[str, Any]]]
    inferred_primary_key_candidate: str | None
    warnings: list[str]


class DatasetPreview(BaseModel):
    dataset_id: str
    columns: list[str]
    rows: list[dict[str, Any]]
    limit: int


class DatasetQueryRequest(BaseModel):
    columns: list[str] | None = None
    filters: dict[str, Any] | None = None
    limit: int | None = None


class DatasetQueryResponse(BaseModel):
    dataset_id: str
    columns: list[str]
    rows: list[dict[str, Any]]
