from __future__ import annotations

from typing import Iterable

import pandas as pd

from app.insights.models import (
    ColumnProfile,
    InsightsAnomalies,
    MissingColumnAnomaly,
    OutlierAnomaly,
    SuspectValueAnomaly,
)


def _detect_outlier_indices(series: pd.Series, method: str) -> list[int]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return []
    if method == "zscore":
        mean = numeric.mean()
        std = numeric.std(ddof=0)
        if std == 0:
            return []
        zscores = (numeric - mean) / std
        outliers = zscores[zscores.abs() > 3]
    else:
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return []
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = numeric[(numeric < lower) | (numeric > upper)]
    return [int(idx) for idx in outliers.index[:20]]


def _detect_suspect_values(
    sample_df: pd.DataFrame,
    column_profiles: dict[str, ColumnProfile],
) -> list[SuspectValueAnomaly]:
    suspects: list[SuspectValueAnomaly] = []
    for column, profile in column_profiles.items():
        series = sample_df[column]
        if profile.type == "text":
            lengths = series.dropna().astype(str).map(len)
            if not lengths.empty and lengths.max() > 200:
                example = series.dropna().astype(str).iloc[0]
                suspects.append(
                    SuspectValueAnomaly(
                        column=column,
                        issue="very long strings detected",
                        example=example[:120],
                    )
                )
        elif profile.type == "datetime":
            parsed = pd.to_datetime(series, errors="coerce", utc=True)
            invalid_rate = float(parsed.isna().mean())
            if invalid_rate > 0.2:
                suspects.append(
                    SuspectValueAnomaly(
                        column=column,
                        issue=f"invalid datetime rate {invalid_rate:.2f}",
                    )
                )
    return suspects


def build_anomalies(
    sample_df: pd.DataFrame,
    column_profiles: dict[str, ColumnProfile],
    missing_threshold: float,
    outlier_method: str,
) -> InsightsAnomalies:
    missing_columns: list[MissingColumnAnomaly] = []
    outliers: list[OutlierAnomaly] = []

    for column, profile in column_profiles.items():
        if profile.missing_rate > missing_threshold:
            missing_columns.append(
                MissingColumnAnomaly(column=column, missing_rate=profile.missing_rate)
            )
        if profile.type == "numeric":
            indices = _detect_outlier_indices(sample_df[column], outlier_method)
            if indices:
                outliers.append(
                    OutlierAnomaly(
                        column=column,
                        method=outlier_method,
                        indices=indices,
                    )
                )

    suspect_values = _detect_suspect_values(sample_df, column_profiles)
    return InsightsAnomalies(
        missing_columns=missing_columns,
        outliers=outliers,
        suspect_values=suspect_values,
    )


def build_recommendations(anomalies: InsightsAnomalies) -> list[str]:
    recommendations: list[str] = []
    if anomalies.missing_columns:
        columns = ", ".join([item.column for item in anomalies.missing_columns])
        recommendations.append(f"Review missing data for columns: {columns}.")
    if anomalies.outliers:
        columns = ", ".join([item.column for item in anomalies.outliers])
        recommendations.append(f"Inspect outliers in numeric columns: {columns}.")
    if anomalies.suspect_values:
        columns = ", ".join([item.column for item in anomalies.suspect_values])
        recommendations.append(f"Validate suspect values in columns: {columns}.")
    if not recommendations:
        recommendations.append("No major issues detected in the sampled data.")
    return recommendations
