import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, UploadFile, status

from app.core.config import settings
from app.schemas.datasets import DatasetMetadata, DatasetSummary


def _ensure_storage_dir() -> Path:
    storage_dir = Path(settings.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


def _validate_filename(filename: str) -> None:
    if not filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing filename.")
    if filename != os.path.basename(filename):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename.")
    if ".." in Path(filename).parts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename.")
    if Path(filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only .csv files are allowed.")


def _save_upload_file(upload: UploadFile, dest_path: Path, max_bytes: int) -> int:
    total_bytes = 0
    with dest_path.open("wb") as buffer:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="File too large.",
                )
            buffer.write(chunk)
    return total_bytes


def _read_metadata(metadata_path: Path) -> DatasetMetadata:
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        return DatasetMetadata(**data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Corrupted metadata.",
        ) from exc


def create_dataset(upload: UploadFile) -> DatasetMetadata:
    _validate_filename(upload.filename or "")
    storage_dir = _ensure_storage_dir()

    dataset_id = str(uuid.uuid4())
    dataset_dir = storage_dir / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=False)

    file_path = dataset_dir / "raw.csv"
    max_bytes = settings.max_upload_mb * 1024 * 1024

    try:
        file_size = _save_upload_file(upload, file_path, max_bytes)
        df = pd.read_csv(file_path)
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            filename=upload.filename or "raw.csv",
            file_size_bytes=file_size,
            created_at=datetime.now(timezone.utc).isoformat(),
            nb_rows=int(df.shape[0]),
            nb_columns=int(df.shape[1]),
            columns=[str(col) for col in df.columns.tolist()],
            dtypes={str(k): str(v) for k, v in df.dtypes.astype(str).to_dict().items()},
            preview=df.head(5).to_dict(orient="records"),
        )
        (dataset_dir / "metadata.json").write_text(
            metadata.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return metadata
    except HTTPException:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process CSV: {exc}",
        ) from exc


def get_dataset(dataset_id: str) -> DatasetMetadata:
    dataset_dir = Path(settings.storage_dir) / dataset_id
    metadata_path = dataset_dir / "metadata.json"
    return _read_metadata(metadata_path)


def list_datasets() -> list[DatasetSummary]:
    storage_dir = _ensure_storage_dir()
    results: list[DatasetSummary] = []
    for dataset_dir in storage_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        metadata_path = dataset_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = _read_metadata(metadata_path)
        results.append(
            DatasetSummary(
                dataset_id=metadata.dataset_id,
                created_at=metadata.created_at,
                filename=metadata.filename,
                nb_rows=metadata.nb_rows,
                nb_columns=metadata.nb_columns,
            )
        )
    return results
