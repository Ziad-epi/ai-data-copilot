from __future__ import annotations

from typing import Iterable

import pandas as pd

from app.insights.models import ChartSpec, ColumnProfile, InsightsResponse


def _normalize_question(question: str) -> str:
    return " ".join(question.lower().split())


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _build_bar_chart(
    df: pd.DataFrame,
    category_col: str,
    numeric_col: str | None,
    max_points: int,
    notes: str,
) -> ChartSpec | None:
    if category_col not in df.columns:
        return None
    if numeric_col and numeric_col in df.columns:
        data = df[[category_col, numeric_col]].dropna()
        if data.empty:
            return None
        grouped = (
            data.groupby(category_col)[numeric_col]
            .mean()
            .sort_values(ascending=False)
            .head(max_points)
        )
        x_vals = [str(idx) for idx in grouped.index]
        y_vals = [float(val) for val in grouped.values]
        title = f"Average {numeric_col} by {category_col}"
        return ChartSpec(
            title=title,
            type="bar",
            x=category_col,
            y=numeric_col,
            aggregation="avg",
            data_preview={"x": x_vals, "y": y_vals},
            notes=notes,
        )

    counts = df[category_col].dropna().astype(str).value_counts().head(max_points)
    if counts.empty:
        return None
    title = f"Top {category_col} values"
    return ChartSpec(
        title=title,
        type="bar",
        x=category_col,
        y=None,
        aggregation="count",
        data_preview={"x": [str(idx) for idx in counts.index], "y": [int(val) for val in counts.values]},
        notes=notes,
    )


def _build_pie_chart(
    df: pd.DataFrame,
    category_col: str,
    max_points: int,
    notes: str,
) -> ChartSpec | None:
    counts = df[category_col].dropna().astype(str).value_counts().head(max_points)
    if counts.empty:
        return None
    title = f"Share of {category_col}"
    return ChartSpec(
        title=title,
        type="pie",
        x=category_col,
        y=None,
        aggregation="count",
        data_preview={"labels": [str(idx) for idx in counts.index], "values": [int(val) for val in counts.values]},
        notes=notes,
    )


def _build_histogram_chart(
    df: pd.DataFrame,
    numeric_col: str,
    max_points: int,
    notes: str,
) -> ChartSpec | None:
    series = pd.to_numeric(df[numeric_col], errors="coerce").dropna()
    if series.empty:
        return None
    bins = min(10, max_points, max(series.nunique(), 2))
    hist = pd.cut(series, bins=bins)
    counts = hist.value_counts().sort_index()
    labels = [f"{interval.left:.2f}-{interval.right:.2f}" for interval in counts.index]
    return ChartSpec(
        title=f"Distribution of {numeric_col}",
        type="histogram",
        x=numeric_col,
        y=None,
        aggregation="count",
        data_preview={"x": labels[:max_points], "y": [int(val) for val in counts.values][:max_points]},
        notes=notes,
    )


def _build_line_chart(
    df: pd.DataFrame,
    datetime_col: str,
    numeric_col: str | None,
    max_points: int,
    notes: str,
) -> ChartSpec | None:
    if datetime_col not in df.columns:
        return None
    parsed = pd.to_datetime(df[datetime_col], errors="coerce", utc=True)
    if parsed.isna().all():
        return None
    if numeric_col and numeric_col in df.columns:
        data = df[[datetime_col, numeric_col]].copy()
        data["__dt"] = parsed
        data = data.dropna(subset=["__dt", numeric_col])
        if data.empty:
            return None
        data["__date"] = data["__dt"].dt.date
        grouped = data.groupby("__date")[numeric_col].mean().sort_index()
        x_vals = [str(idx) for idx in grouped.index][:max_points]
        y_vals = [float(val) for val in grouped.values][:max_points]
        title = f"Average {numeric_col} over time"
        return ChartSpec(
            title=title,
            type="line",
            x=datetime_col,
            y=numeric_col,
            aggregation="avg",
            data_preview={"x": x_vals, "y": y_vals},
            notes=notes,
        )

    data = parsed.dropna()
    if data.empty:
        return None
    grouped = data.dt.date.value_counts().sort_index()
    x_vals = [str(idx) for idx in grouped.index][:max_points]
    y_vals = [int(val) for val in grouped.values][:max_points]
    title = f"Count over time ({datetime_col})"
    return ChartSpec(
        title=title,
        type="line",
        x=datetime_col,
        y=None,
        aggregation="count",
        data_preview={"x": x_vals, "y": y_vals},
        notes=notes,
    )


def _build_scatter_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    max_points: int,
    notes: str,
) -> ChartSpec | None:
    if x_col not in df.columns or y_col not in df.columns:
        return None
    data = df[[x_col, y_col]].dropna().head(max_points)
    if data.empty:
        return None
    return ChartSpec(
        title=f"{y_col} vs {x_col}",
        type="scatter",
        x=x_col,
        y=y_col,
        aggregation=None,
        data_preview={
            "x": [float(val) for val in data[x_col].tolist()],
            "y": [float(val) for val in data[y_col].tolist()],
        },
        notes=notes,
    )


def suggest_charts(
    insights: InsightsResponse,
    sample_df: pd.DataFrame,
    question: str | None,
    max_charts: int,
    max_points: int,
) -> list[ChartSpec]:
    profiles = insights.column_profiles
    numeric_cols = [col for col, profile in profiles.items() if profile.type == "numeric"]
    categorical_cols = [col for col, profile in profiles.items() if profile.type == "categorical"]
    datetime_cols = [col for col, profile in profiles.items() if profile.type == "datetime"]

    charts: list[ChartSpec] = []
    chart_keys: set[tuple[str, str | None, str | None, str | None]] = set()

    def add_chart(chart: ChartSpec | None) -> None:
        if chart is None:
            return
        key = (chart.type, chart.x, chart.y, chart.aggregation)
        if key in chart_keys:
            return
        chart_keys.add(key)
        charts.append(chart)

    if question:
        normalized = _normalize_question(question)
        if _contains_any(normalized, ["evolution", "trend", "over time", "time"]):
            if datetime_cols:
                add_chart(
                    _build_line_chart(
                        sample_df,
                        datetime_cols[0],
                        numeric_cols[0] if numeric_cols else None,
                        max_points,
                        "Requested trend over time.",
                    )
                )
        elif _contains_any(normalized, ["distribution", "histogram", "repartition"]):
            if numeric_cols:
                add_chart(
                    _build_histogram_chart(
                        sample_df,
                        numeric_cols[0],
                        max_points,
                        "Requested distribution view.",
                    )
                )
        elif _contains_any(normalized, ["compare", "comparison", "comparaison", "top"]):
            if categorical_cols:
                add_chart(
                    _build_bar_chart(
                        sample_df,
                        categorical_cols[0],
                        numeric_cols[0] if numeric_cols else None,
                        max_points,
                        "Requested comparison between categories.",
                    )
                )
        elif _contains_any(normalized, ["share", "ratio", "percentage", "proportion"]):
            if categorical_cols:
                add_chart(
                    _build_pie_chart(
                        sample_df,
                        categorical_cols[0],
                        max_points,
                        "Requested proportional view.",
                    )
                )

    if datetime_cols and len(charts) < max_charts:
        add_chart(
            _build_line_chart(
                sample_df,
                datetime_cols[0],
                numeric_cols[0] if numeric_cols else None,
                max_points,
                "Time-based overview.",
            )
        )

    if categorical_cols and len(charts) < max_charts:
        add_chart(
            _build_bar_chart(
                sample_df,
                categorical_cols[0],
                numeric_cols[0] if numeric_cols else None,
                max_points,
                "Category comparison.",
            )
        )

    if len(numeric_cols) >= 2 and len(charts) < max_charts:
        add_chart(
            _build_scatter_chart(
                sample_df,
                numeric_cols[0],
                numeric_cols[1],
                max_points,
                "Relationship between numeric features.",
            )
        )

    if numeric_cols and len(charts) < max_charts:
        add_chart(
            _build_histogram_chart(
                sample_df,
                numeric_cols[0],
                max_points,
                "Distribution of numeric values.",
            )
        )

    return charts[:max_charts]
