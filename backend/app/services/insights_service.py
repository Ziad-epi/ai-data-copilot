from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException, status

from app.core.config import settings
from app.insights.anomalies import build_anomalies, build_recommendations
from app.insights.models import InsightsRequest, InsightsResponse
from app.insights.profiling import build_column_profiles, build_dataset_overview, load_sample_frame
from app.services.datasets import load_dataset


def _insights_path(dataset_id: str) -> Path:
    return Path(settings.storage_dir) / dataset_id / "insights.json"


def _load_cached_insights(path: Path) -> InsightsResponse:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Insights not found.") from exc
    return InsightsResponse(**data)


def get_dataset_insights(dataset_id: str, payload: InsightsRequest) -> InsightsResponse:
    metadata, file_path = load_dataset(dataset_id)
    if payload.target_column and payload.target_column not in metadata.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unknown target_column.",
        )

    insights_path = _insights_path(dataset_id)
    if insights_path.exists() and not payload.force_recompute:
        return _load_cached_insights(insights_path)

    sample_rows = payload.sample_rows or settings.insights_sample_max
    sample_rows = min(sample_rows, settings.insights_sample_max, metadata.nb_rows)
    if sample_rows <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="sample_rows must be positive.")

    sample_df = load_sample_frame(metadata, file_path, sample_rows)
    overview = build_dataset_overview(metadata, sample_df)
    column_profiles = build_column_profiles(sample_df)
    anomalies = build_anomalies(
        sample_df,
        column_profiles,
        settings.insights_missing_threshold,
        settings.insights_outlier_method,
    )
    recommendations = build_recommendations(anomalies)

    insights = InsightsResponse(
        dataset_id=dataset_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        sample_rows_used=len(sample_df),
        target_column=payload.target_column,
        dataset_overview=overview,
        column_profiles=column_profiles,
        anomalies=anomalies,
        recommendations=recommendations,
    )

    insights_path.write_text(
        json.dumps(insights.model_dump(), indent=2),
        encoding="utf-8",
    )
    return insights
