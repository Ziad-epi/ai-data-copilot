from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status
from qdrant_client.models import PointStruct

from app.core.config import settings
from app.rag.chunking import RagDocument, build_summary_doc, iter_row_docs
from app.rag.embeddings import get_embedding_service
from app.rag.vector_store import VectorStore
from app.schemas.datasets import DatasetMetadata
from app.services.datasets import load_dataset


def _index_metadata_path(dataset_id: str) -> Path:
    return Path(settings.storage_dir) / dataset_id / "index_metadata.json"


def _validate_columns(metadata: DatasetMetadata, columns: list[str]) -> None:
    for col in columns:
        if col not in metadata.columns:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown column: {col}")


def _validate_doc_types(doc_types: list[str]) -> None:
    allowed = {"summary", "rows"}
    unknown = [doc_type for doc_type in doc_types if doc_type not in allowed]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown doc_types: {', '.join(unknown)}",
        )


def _iter_documents(
    metadata: DatasetMetadata,
    file_path: Path,
    columns: list[str],
    max_rows: int,
    rows_per_doc: int,
) -> list[RagDocument]:
    docs: list[RagDocument] = [build_summary_doc(metadata, columns)]
    docs.extend(iter_row_docs(file_path, metadata, columns, max_rows, rows_per_doc))
    return docs


def _write_index_metadata(
    dataset_id: str,
    nb_docs: int,
    params: dict[str, Any],
    status_value: str = "indexed",
) -> None:
    payload = {
        "dataset_id": dataset_id,
        "embedding_model": settings.rag_embedding_model,
        "vector_store": "qdrant",
        "collection": settings.rag_collection_name,
        "nb_docs": nb_docs,
        "params": params,
        "status": status_value,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _index_metadata_path(dataset_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def index_dataset(
    dataset_id: str,
    columns: list[str] | None = None,
    max_rows: int | None = None,
    rows_per_doc: int | None = None,
    reindex: bool = False,
) -> dict[str, Any]:
    metadata, file_path = load_dataset(dataset_id)
    selected_columns = columns or metadata.columns
    if not selected_columns:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="columns must not be empty.")
    _validate_columns(metadata, selected_columns)

    max_rows_value = max_rows if max_rows is not None else settings.rag_max_rows_to_index
    rows_per_doc_value = rows_per_doc if rows_per_doc is not None else settings.rag_rows_per_doc
    if max_rows_value <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="max_rows must be positive.")
    if rows_per_doc_value <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="rows_per_doc must be positive.")

    if reindex:
        VectorStore().delete(dataset_id)
        metadata_path = _index_metadata_path(dataset_id)
        if metadata_path.exists():
            metadata_path.unlink()

    documents = _iter_documents(metadata, file_path, selected_columns, max_rows_value, rows_per_doc_value)
    if not documents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No documents to index.")

    start_time = time.perf_counter()
    embedder = get_embedding_service()

    vectors_upserted = 0
    doc_index = 0
    vector_store: VectorStore | None = None
    batch_size = settings.rag_embed_batch_size

    for batch_start in range(0, len(documents), batch_size):
        batch = documents[batch_start : batch_start + batch_size]
        texts = [doc.text for doc in batch]
        vectors = embedder.embed_texts(texts)
        if not vectors:
            continue
        if vector_store is None:
            try:
                vector_store = VectorStore(vector_size=len(vectors[0]))
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Embedding size mismatch. Reindex the dataset.",
                ) from exc
        points: list[PointStruct] = []
        for doc, vector in zip(batch, vectors, strict=True):
            point_id = f"{dataset_id}:{doc_index}"
            payload = {"text": doc.text, **doc.metadata}
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
            doc_index += 1
        vectors_upserted += vector_store.upsert(points)

    duration_ms = int((time.perf_counter() - start_time) * 1000)
    _write_index_metadata(
        dataset_id,
        nb_docs=len(documents),
        params={
            "columns": selected_columns,
            "max_rows": max_rows_value,
            "rows_per_doc": rows_per_doc_value,
            "reindex": reindex,
        },
    )
    return {"nb_docs": len(documents), "vectors_upserted": vectors_upserted, "duration_ms": duration_ms}


def search_dataset(
    dataset_id: str,
    query: str,
    top_k: int,
    doc_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    load_dataset(dataset_id)
    if not _index_metadata_path(dataset_id).exists():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Dataset not indexed yet.")

    doc_types_value = doc_types or ["summary", "rows"]
    _validate_doc_types(doc_types_value)

    embedder = get_embedding_service()
    query_vector = embedder.embed_texts([query])
    if not query_vector:
        return []
    try:
        vector_store = VectorStore(vector_size=len(query_vector[0]), recreate_on_mismatch=False)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Embedding size mismatch. Reindex the dataset.",
        ) from exc
    hits = vector_store.search(
        query_vector=query_vector[0],
        top_k=top_k,
        dataset_id=dataset_id,
        doc_types=doc_types_value,
    )

    results: list[dict[str, Any]] = []
    for hit in hits:
        payload = hit.payload or {}
        row_start = payload.get("row_start")
        row_end = payload.get("row_end")
        citation = f"dataset:{dataset_id}"
        if row_start is not None and row_end is not None:
            citation = f"{citation} rows:{row_start}-{row_end}"
        results.append(
            {
                "score": float(hit.score),
                "text": payload.get("text", ""),
                "source": {
                    "dataset_id": payload.get("dataset_id", dataset_id),
                    "doc_type": payload.get("doc_type"),
                    "row_start": row_start,
                    "row_end": row_end,
                },
                "citation": citation,
            }
        )
    return results
