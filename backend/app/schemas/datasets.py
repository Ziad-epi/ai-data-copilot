from typing import Any

from pydantic import BaseModel


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


class DatasetSummary(BaseModel):
    dataset_id: str
    created_at: str
    filename: str
    nb_rows: int
    nb_columns: int
