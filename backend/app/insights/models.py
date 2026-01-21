from typing import Any

from pydantic import BaseModel, Field


class InsightsRequest(BaseModel):
    sample_rows: int | None = Field(default=None, ge=1)
    target_column: str | None = None
    force_recompute: bool = False


class TopValue(BaseModel):
    value: str
    count: int


class NumericSummary(BaseModel):
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    p50: float | None
    p95: float | None


class ColumnProfile(BaseModel):
    type: str
    missing_rate: float
    unique_count: int
    unique_rate: float
    top_values: list[TopValue] | None = None
    numeric_summary: NumericSummary | None = None


class DatasetOverview(BaseModel):
    rows: int
    cols: int
    memory_estimate: int
    missing_rate_global: float


class MissingColumnAnomaly(BaseModel):
    column: str
    missing_rate: float


class OutlierAnomaly(BaseModel):
    column: str
    method: str
    indices: list[int]


class SuspectValueAnomaly(BaseModel):
    column: str
    issue: str
    example: str | None = None


class InsightsAnomalies(BaseModel):
    missing_columns: list[MissingColumnAnomaly]
    outliers: list[OutlierAnomaly]
    suspect_values: list[SuspectValueAnomaly]


class InsightsResponse(BaseModel):
    dataset_id: str
    generated_at: str
    sample_rows_used: int
    target_column: str | None
    dataset_overview: DatasetOverview
    column_profiles: dict[str, ColumnProfile]
    anomalies: InsightsAnomalies
    recommendations: list[str]


class ChartsSuggestRequest(BaseModel):
    question: str | None = None
    max_charts: int = Field(default=3, ge=1, le=10)


class ChartSpec(BaseModel):
    title: str
    type: str
    x: str | None
    y: str | None
    aggregation: str | None
    filters: dict[str, Any] = Field(default_factory=dict)
    data_preview: dict[str, list[Any]] = Field(default_factory=dict)
    notes: str


class ChartsSuggestResponse(BaseModel):
    dataset_id: str
    charts: list[ChartSpec]


class ReportResponse(BaseModel):
    dataset_id: str
    report_markdown: str
    used_llm: bool
    generated_at: str
