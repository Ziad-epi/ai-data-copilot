import csv
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def _validate_dataset_id(dataset_id: str) -> None:
    try:
        uuid.UUID(dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid dataset_id.") from exc


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


def _read_sample_bytes(file_path: Path, size: int = 65536) -> bytes:
    with file_path.open("rb") as handle:
        return handle.read(size)


def _detect_encoding(sample: bytes) -> tuple[str, list[str]]:
    warnings: list[str] = []
    try:
        sample.decode("utf-8")
        return "utf-8", warnings
    except UnicodeDecodeError:
        warnings.append("encoding fallback used: latin-1")
        return "latin-1", warnings


def _detect_delimiter(sample_text: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=[",", ";", "\t"])
        return dialect.delimiter, warnings
    except csv.Error:
        warnings.append("delimiter detection failed, defaulting to ','")
        return ",", warnings


def _detect_header(sample_text: str) -> bool:
    try:
        return csv.Sniffer().has_header(sample_text)
    except csv.Error:
        return True


def _read_header_row(file_path: Path, encoding: str, delimiter: str) -> list[str]:
    with file_path.open("r", encoding=encoding) as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            if any(cell.strip() for cell in row):
                return [item.strip() for item in row]
    return []


def _normalize_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    if not columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing header row.")

    cleaned: list[str] = []
    seen: dict[str, int] = {}
    duplicate_found = False
    for raw in columns:
        name = raw.strip()
        if not name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid header column name.")
        count = seen.get(name, 0)
        if count:
            new_name = f"{name}_dup{count}"
            duplicate_found = True
            cleaned.append(new_name)
        else:
            cleaned.append(name)
        seen[name] = count + 1
    if duplicate_found:
        warnings.append("duplicate column names renamed")
    return cleaned, warnings


def _iter_csv_chunks(
    file_path: Path,
    delimiter: str,
    encoding: str,
    columns: list[str],
    chunk_size: int = 10000,
) -> pd.io.parsers.TextFileReader:
    return pd.read_csv(
        file_path,
        sep=delimiter,
        encoding=encoding,
        header=0,
        names=columns,
        skip_blank_lines=True,
        skipinitialspace=True,
        chunksize=chunk_size,
    )


def _summarize_numeric(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    numeric_df = df.select_dtypes(include="number")
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue
        result[str(col)] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
        }
    return result


def _summarize_top_values(df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    categorical_df = df.select_dtypes(exclude="number")
    for col in categorical_df.columns:
        counts = categorical_df[col].dropna().value_counts().head(5)
        if counts.empty:
            continue
        result[str(col)] = [{"value": str(idx), "count": int(val)} for idx, val in counts.items()]
    return result


def _infer_primary_key(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    for col in df.columns:
        series = df[col]
        if series.isna().any():
            continue
        unique_ratio = series.nunique(dropna=True) / len(series)
        if unique_ratio >= 0.95:
            return str(col)
    return None


def _build_metadata(
    file_path: Path,
    filename: str,
    file_size: int,
    delimiter: str,
    encoding: str,
    columns: list[str],
    warnings: list[str],
) -> DatasetMetadata:
    sample_rows = settings.sample_rows
    preview_rows: list[dict[str, Any]] = []
    sample_frames: list[pd.DataFrame] = []
    missing_counts: dict[str, int] = {col: 0 for col in columns}
    total_rows = 0

    for chunk in _iter_csv_chunks(file_path, delimiter, encoding, columns):
        total_rows += len(chunk)
        chunk_missing = chunk.isna().sum().to_dict()
        for col, count in chunk_missing.items():
            missing_counts[str(col)] = missing_counts.get(str(col), 0) + int(count)

        if len(preview_rows) < 5:
            needed = 5 - len(preview_rows)
            preview_rows.extend(chunk.head(needed).to_dict(orient="records"))

        if sum(len(frame) for frame in sample_frames) < sample_rows:
            remaining = sample_rows - sum(len(frame) for frame in sample_frames)
            sample_frames.append(chunk.head(remaining))

    if total_rows == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="CSV contains no data.")

    sample_df = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    numeric_summary = _summarize_numeric(sample_df)
    top_values = _summarize_top_values(sample_df)
    inferred_pk = _infer_primary_key(sample_df)

    for col, count in missing_counts.items():
        if total_rows and (count / total_rows) > 0.5:
            warnings.append(f"high missing rate in col {col}")

    metadata = DatasetMetadata(
        dataset_id=file_path.parent.name,
        filename=filename,
        file_size_bytes=file_size,
        created_at=datetime.now(timezone.utc).isoformat(),
        nb_rows=total_rows,
        nb_columns=len(columns),
        columns=columns,
        dtypes={str(k): str(v) for k, v in sample_df.dtypes.astype(str).to_dict().items()},
        preview=preview_rows,
        delimiter=delimiter,
        encoding=encoding,
        missing_values_count=missing_counts,
        numeric_columns_summary=numeric_summary,
        top_values=top_values,
        inferred_primary_key_candidate=inferred_pk,
        warnings=warnings,
    )
    return metadata


def create_dataset(upload: UploadFile, delimiter: str | None = None) -> DatasetMetadata:
    _validate_filename(upload.filename or "")
    storage_dir = _ensure_storage_dir()

    dataset_id = str(uuid.uuid4())
    dataset_dir = storage_dir / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=False)

    file_path = dataset_dir / "raw.csv"
    max_bytes = settings.max_upload_mb * 1024 * 1024

    try:
        file_size = _save_upload_file(upload, file_path, max_bytes)
        sample_bytes = _read_sample_bytes(file_path)
        encoding, encoding_warnings = _detect_encoding(sample_bytes)
        sample_text = sample_bytes.decode(encoding)

        if delimiter:
            if len(delimiter) != 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Delimiter must be a single character.",
                )
            delimiter_value = delimiter
            delimiter_warnings: list[str] = []
        else:
            delimiter_value, delimiter_warnings = _detect_delimiter(sample_text)

        if not _detect_header(sample_text):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing header row.")

        raw_columns = _read_header_row(file_path, encoding, delimiter_value)
        normalized_columns, normalization_warnings = _normalize_columns(raw_columns)

        warnings = encoding_warnings + delimiter_warnings + normalization_warnings
        metadata = _build_metadata(
            file_path=file_path,
            filename=upload.filename or "raw.csv",
            file_size=file_size,
            delimiter=delimiter_value,
            encoding=encoding,
            columns=normalized_columns,
            warnings=warnings,
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


def load_dataset(dataset_id: str) -> tuple[DatasetMetadata, Path]:
    _validate_dataset_id(dataset_id)
    dataset_dir = Path(settings.storage_dir) / dataset_id
    metadata_path = dataset_dir / "metadata.json"
    metadata = _read_metadata(metadata_path)
    file_path = dataset_dir / "raw.csv"
    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")
    return metadata, file_path


def get_dataset(dataset_id: str) -> DatasetMetadata:
    metadata, _ = load_dataset(dataset_id)
    return metadata


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


def read_dataframe(
    dataset_id: str,
    limit: int | None = None,
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    max_rows: int | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    metadata, file_path = load_dataset(dataset_id)
    selected_columns = columns or metadata.columns

    if limit is not None and limit <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Limit must be positive.")

    for col in selected_columns:
        if col not in metadata.columns:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown column: {col}")

    filters = filters or {}
    for col in filters.keys():
        if col not in metadata.columns:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown column: {col}")

    if max_rows is None:
        max_rows = settings.query_max_rows
    if limit is None:
        limit = max_rows
    limit = min(limit, max_rows)

    results: list[dict[str, Any]] = []
    delimiter = metadata.delimiter or ","
    encoding = metadata.encoding or "utf-8"
    for chunk in _iter_csv_chunks(file_path, delimiter=delimiter, encoding=encoding, columns=metadata.columns):
        if filters:
            for col, value in filters.items():
                series = chunk[col]
                if pd.api.types.is_numeric_dtype(series):
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        pass
                chunk = chunk[series == value]

        if selected_columns:
            chunk = chunk[selected_columns]

        if not chunk.empty:
            remaining = limit - len(results)
            rows = chunk.head(remaining).to_dict(orient="records")
            results.extend(rows)
            if len(results) >= limit:
                break

    return selected_columns, results
