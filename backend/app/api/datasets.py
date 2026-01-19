from fastapi import APIRouter, File, UploadFile

from app.schemas.datasets import DatasetMetadata, DatasetSummary
from app.services.datasets import create_dataset, get_dataset, list_datasets

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload", response_model=DatasetMetadata)
def upload_dataset(file: UploadFile = File(...)) -> DatasetMetadata:
    return create_dataset(file)


@router.get("/{dataset_id}", response_model=DatasetMetadata)
def get_dataset_metadata(dataset_id: str) -> DatasetMetadata:
    return get_dataset(dataset_id)


@router.get("", response_model=list[DatasetSummary])
def list_all_datasets() -> list[DatasetSummary]:
    return list_datasets()
