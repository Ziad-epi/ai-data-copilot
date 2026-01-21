from __future__ import annotations

from app.core.config import settings
from app.insights.charts import suggest_charts as build_charts
from app.insights.models import ChartsSuggestRequest, ChartsSuggestResponse, InsightsRequest
from app.insights.profiling import load_sample_frame
from app.services.datasets import load_dataset
from app.services.insights_service import get_dataset_insights


def suggest_charts(dataset_id: str, payload: ChartsSuggestRequest) -> ChartsSuggestResponse:
    metadata, file_path = load_dataset(dataset_id)
    insights = get_dataset_insights(
        dataset_id,
        InsightsRequest(sample_rows=settings.insights_sample_max, force_recompute=False),
    )
    sample_rows = min(
        insights.sample_rows_used or settings.insights_sample_max,
        settings.insights_sample_max,
        metadata.nb_rows,
    )
    sample_df = load_sample_frame(metadata, file_path, sample_rows)
    charts = build_charts(
        insights=insights,
        sample_df=sample_df,
        question=payload.question,
        max_charts=payload.max_charts,
        max_points=settings.charts_max_points,
    )
    return ChartsSuggestResponse(dataset_id=dataset_id, charts=charts)
