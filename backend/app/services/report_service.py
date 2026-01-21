from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.insights.models import ChartsSuggestRequest, InsightsRequest, ReportResponse
from app.llm.provider import get_llm_client
from app.services.charts_service import suggest_charts
from app.services.datasets import load_dataset
from app.services.insights_service import get_dataset_insights

SYSTEM_PROMPT = (
    "You are an analytics assistant. Use only the provided insights and chart specs. "
    "Do not invent facts. If information is missing, say so explicitly."
)


def _report_path(dataset_id: str) -> Path:
    return Path(settings.storage_dir) / dataset_id / "report.md"


def _should_use_llm() -> bool:
    if settings.llm_provider != "openai_compatible":
        return False
    return bool(settings.llm_api_key and settings.llm_base_url and settings.llm_model)


def _compact_insights(insights) -> dict:
    missing_cols = sorted(
        insights.anomalies.missing_columns,
        key=lambda item: item.missing_rate,
        reverse=True,
    )[:5]
    numeric_columns: dict[str, dict[str, float | None]] = {}
    categorical_columns: dict[str, list[dict[str, str | int]]] = {}

    for column, profile in insights.column_profiles.items():
        if profile.type == "numeric" and profile.numeric_summary:
            numeric_columns[column] = profile.numeric_summary.model_dump()
        if profile.type == "categorical" and profile.top_values:
            categorical_columns[column] = [item.model_dump() for item in profile.top_values]
        if len(numeric_columns) >= 5 and len(categorical_columns) >= 3:
            break

    return {
        "dataset_overview": insights.dataset_overview.model_dump(),
        "missing_columns": [item.model_dump() for item in missing_cols],
        "outliers": [item.model_dump() for item in insights.anomalies.outliers],
        "suspect_values": [item.model_dump() for item in insights.anomalies.suspect_values],
        "recommendations": insights.recommendations,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }


def _build_report_template(insights, charts) -> str:
    overview = insights.dataset_overview
    lines = [
        "# Executive Report",
        "",
        "## Dataset Summary",
        f"- Rows: {overview.rows}",
        f"- Columns: {overview.cols}",
        f"- Missing rate (global): {overview.missing_rate_global:.2%}",
        f"- Memory estimate (bytes): {overview.memory_estimate}",
        "",
        "## Key Insights",
    ]

    key_insights: list[str] = []
    if insights.anomalies.missing_columns:
        cols = ", ".join(item.column for item in insights.anomalies.missing_columns[:5])
        key_insights.append(f"High missing rate in columns: {cols}.")
    if insights.anomalies.outliers:
        cols = ", ".join(item.column for item in insights.anomalies.outliers[:5])
        key_insights.append(f"Outliers detected in numeric columns: {cols}.")
    if insights.anomalies.suspect_values:
        cols = ", ".join(item.column for item in insights.anomalies.suspect_values[:5])
        key_insights.append(f"Suspect values detected in columns: {cols}.")

    numeric_profiles = [
        (col, profile.numeric_summary)
        for col, profile in insights.column_profiles.items()
        if profile.type == "numeric" and profile.numeric_summary
    ]
    for col, summary in numeric_profiles[:2]:
        key_insights.append(
            f"{col} ranges from {summary.min} to {summary.max} (p95 {summary.p95})."
        )

    while len(key_insights) < 5:
        key_insights.append("No additional insights from the sample.")

    for insight in key_insights[:5]:
        lines.append(f"- {insight}")

    lines.extend(
        [
            "",
            "## Anomalies",
            f"- Missing columns: {len(insights.anomalies.missing_columns)}",
            f"- Outlier columns: {len(insights.anomalies.outliers)}",
            f"- Suspect values: {len(insights.anomalies.suspect_values)}",
            "",
            "## Recommended Charts",
        ]
    )

    if charts:
        for chart in charts:
            lines.append(f"- {chart.title} ({chart.type})")
    else:
        lines.append("- No charts suggested.")

    lines.extend(
        [
            "",
            "## Recommendations",
        ]
    )
    for rec in insights.recommendations:
        lines.append(f"- {rec}")

    return "\n".join(lines).strip() + "\n"


def generate_report(dataset_id: str) -> ReportResponse:
    load_dataset(dataset_id)
    insights = get_dataset_insights(
        dataset_id,
        InsightsRequest(sample_rows=settings.insights_sample_max, force_recompute=False),
    )
    charts_response = suggest_charts(
        dataset_id,
        ChartsSuggestRequest(question=None, max_charts=3),
    )

    used_llm = False
    if _should_use_llm():
        llm_client = get_llm_client()
        compact = _compact_insights(insights)
        payload = {
            "insights": compact,
            "charts": [chart.model_dump() for chart in charts_response.charts],
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Write an executive report in Markdown with sections: "
                    "Dataset summary, Key insights (5 bullets), Anomalies, "
                    "Recommended charts. Use only the provided JSON.\n\n"
                    f"{json.dumps(payload, indent=2)}"
                ),
            },
        ]
        llm_response = llm_client.generate(messages)
        report_markdown = llm_response.text.strip() or _build_report_template(
            insights, charts_response.charts
        )
        used_llm = True
    else:
        report_markdown = _build_report_template(insights, charts_response.charts)

    report_path = _report_path(dataset_id)
    report_path.write_text(report_markdown, encoding="utf-8")

    return ReportResponse(
        dataset_id=dataset_id,
        report_markdown=report_markdown,
        used_llm=used_llm,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
