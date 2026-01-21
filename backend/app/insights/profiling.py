from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.insights.models import ColumnProfile, DatasetOverview, NumericSummary, TopValue
from app.schemas.datasets import DatasetMetadata


def load_sample_frame(
    metadata: DatasetMetadata,
    file_path: Path,
    sample_rows: int,
) -> pd.DataFrame:
    delimiter = metadata.delimiter or ","
    encoding = metadata.encoding or "utf-8"
    df = pd.read_csv(
        file_path,
        sep=delimiter,
        encoding=encoding,
        header=0,
        names=metadata.columns,
        nrows=sample_rows,
        low_memory=False,
    )
    df.index = df.index + 1
    return df


def _try_parse_datetime(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    sample = non_null.head(1000)
    parsed = pd.to_datetime(sample, errors="coerce", utc=True)
    return float(parsed.notna().mean())


def _infer_column_type(series: pd.Series, unique_count: int, unique_rate: float) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    parse_rate = _try_parse_datetime(series)
    if parse_rate >= 0.8:
        return "datetime"

    if unique_rate <= 0.2 or unique_count <= 20:
        return "categorical"
    return "text"


def _numeric_summary(series: pd.Series) -> NumericSummary | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return NumericSummary(
        min=float(numeric.min()),
        max=float(numeric.max()),
        mean=float(numeric.mean()),
        std=float(numeric.std(ddof=0)),
        p50=float(numeric.quantile(0.5)),
        p95=float(numeric.quantile(0.95)),
    )


def _top_values(series: pd.Series) -> list[TopValue]:
    counts = series.dropna().astype(str).value_counts().head(5)
    return [TopValue(value=str(idx), count=int(val)) for idx, val in counts.items()]


def build_dataset_overview(
    metadata: DatasetMetadata,
    sample_df: pd.DataFrame,
) -> DatasetOverview:
    rows = metadata.nb_rows
    cols = metadata.nb_columns
    sample_rows = len(sample_df)
    sample_cells = max(sample_rows * cols, 1)
    missing_rate_global = float(sample_df.isna().sum().sum() / sample_cells)
    sample_memory = int(sample_df.memory_usage(deep=True).sum())
    if sample_rows > 0 and rows > sample_rows:
        memory_estimate = int(sample_memory * (rows / sample_rows))
    else:
        memory_estimate = sample_memory
    return DatasetOverview(
        rows=rows,
        cols=cols,
        memory_estimate=memory_estimate,
        missing_rate_global=missing_rate_global,
    )


def build_column_profiles(sample_df: pd.DataFrame) -> dict[str, ColumnProfile]:
    profiles: dict[str, ColumnProfile] = {}
    total_rows = len(sample_df)
    for column in sample_df.columns:
        series = sample_df[column]
        missing_rate = float(series.isna().mean()) if total_rows else 0.0
        unique_count = int(series.nunique(dropna=True))
        unique_rate = float(unique_count / total_rows) if total_rows else 0.0
        col_type = _infer_column_type(series, unique_count, unique_rate)
        top_values = _top_values(series) if col_type == "categorical" else []
        numeric_summary = _numeric_summary(series) if col_type == "numeric" else None
        profiles[column] = ColumnProfile(
            type=col_type,
            missing_rate=missing_rate,
            unique_count=unique_count,
            unique_rate=unique_rate,
            top_values=top_values or None,
            numeric_summary=numeric_summary,
        )
    return profiles
