from fastapi import APIRouter, File, Query, UploadFile

from app.core.config import settings
from app.insights.models import (
    ChartsSuggestRequest,
    ChartsSuggestResponse,
    InsightsRequest,
    InsightsResponse,
    ReportResponse,
)
from app.schemas.datasets import (
    DatasetMetadata,
    DatasetPreview,
    DatasetQueryRequest,
    DatasetQueryResponse,
    DatasetSchema,
    DatasetSummary,
)
from app.schemas.rag import (
    DatasetIndexRequest,
    DatasetIndexResponse,
    DatasetSearchRequest,
    DatasetSearchResponse,
)
from app.services.charts_service import suggest_charts
from app.services.datasets import (
    create_dataset,
    get_dataset,
    list_datasets,
    load_dataset,
    read_dataframe,
)
from app.services.insights_service import get_dataset_insights
from app.services.rag import index_dataset, search_dataset
from app.services.report_service import generate_report

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload", response_model=DatasetMetadata)
def upload_dataset(
    file: UploadFile = File(...),
    delimiter: str | None = Query(default=None, description="Optional CSV delimiter."),
) -> DatasetMetadata:
    return create_dataset(file, delimiter=delimiter)


@router.get("/{dataset_id}", response_model=DatasetMetadata)
def get_dataset_metadata(dataset_id: str) -> DatasetMetadata:
    return get_dataset(dataset_id)


@router.get("", response_model=list[DatasetSummary])
def list_all_datasets() -> list[DatasetSummary]:
    return list_datasets()


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
def preview_dataset(
    dataset_id: str,
    limit: int = Query(default=20, ge=1),
) -> DatasetPreview:
    max_rows = settings.preview_max_rows
    columns, rows = read_dataframe(dataset_id, limit=limit, max_rows=max_rows)
    return DatasetPreview(dataset_id=dataset_id, columns=columns, rows=rows, limit=min(limit, max_rows))


@router.get("/{dataset_id}/schema", response_model=DatasetSchema)
def get_dataset_schema(dataset_id: str) -> DatasetSchema:
    metadata, _ = load_dataset(dataset_id)
    return DatasetSchema(
        dataset_id=metadata.dataset_id,
        columns=metadata.columns,
        dtypes=metadata.dtypes,
        missing_values_count=metadata.missing_values_count,
        numeric_columns_summary=metadata.numeric_columns_summary,
        top_values=metadata.top_values,
        inferred_primary_key_candidate=metadata.inferred_primary_key_candidate,
        warnings=metadata.warnings,
    )


@router.post("/{dataset_id}/query", response_model=DatasetQueryResponse)
def query_dataset(dataset_id: str, payload: DatasetQueryRequest) -> DatasetQueryResponse:
    columns, rows = read_dataframe(
        dataset_id,
        limit=payload.limit,
        columns=payload.columns,
        filters=payload.filters,
        max_rows=settings.query_max_rows,
    )
    return DatasetQueryResponse(dataset_id=dataset_id, columns=columns, rows=rows)


@router.post("/{dataset_id}/index", response_model=DatasetIndexResponse)
def index_dataset_endpoint(dataset_id: str, payload: DatasetIndexRequest) -> DatasetIndexResponse:
    result = index_dataset(
        dataset_id,
        columns=payload.columns,
        max_rows=payload.max_rows,
        rows_per_doc=payload.rows_per_doc,
        reindex=payload.reindex,
    )
    return DatasetIndexResponse(dataset_id=dataset_id, **result)


@router.post("/{dataset_id}/search", response_model=DatasetSearchResponse)
def search_dataset_endpoint(dataset_id: str, payload: DatasetSearchRequest) -> DatasetSearchResponse:
    results = search_dataset(
        dataset_id,
        query=payload.query,
        top_k=payload.top_k,
        doc_types=payload.doc_types,
    )
    return DatasetSearchResponse(dataset_id=dataset_id, results=results)


@router.post("/{dataset_id}/insights", response_model=InsightsResponse)
def insights_dataset_endpoint(
    dataset_id: str,
    payload: InsightsRequest | None = None,
) -> InsightsResponse:
    return get_dataset_insights(dataset_id, payload or InsightsRequest())


@router.post("/{dataset_id}/charts/suggest", response_model=ChartsSuggestResponse)
def charts_suggest_dataset_endpoint(
    dataset_id: str,
    payload: ChartsSuggestRequest | None = None,
) -> ChartsSuggestResponse:
    return suggest_charts(dataset_id, payload or ChartsSuggestRequest())


@router.post("/{dataset_id}/report", response_model=ReportResponse)
def report_dataset_endpoint(dataset_id: str) -> ReportResponse:
    return generate_report(dataset_id)
