from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, MatchValue, PointStruct, VectorParams

from app.core.config import settings


class VectorStore:
    def __init__(
        self,
        vector_size: int | None = None,
        collection_name: str | None = None,
        recreate_on_mismatch: bool = False,
    ) -> None:
        self.collection_name = collection_name or settings.rag_collection_name
        if settings.qdrant_path:
            self.client = QdrantClient(path=settings.qdrant_path)
        else:
            self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self._recreate_on_mismatch = recreate_on_mismatch
        if vector_size is not None:
            self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection_name):
            info = self.client.get_collection(self.collection_name)
            existing_size = info.config.params.vectors.size
            if existing_size != vector_size:
                if not self._recreate_on_mismatch:
                    raise ValueError("Vector size mismatch. Reindex required.")
                self.client.delete_collection(self.collection_name)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert(self, points: list[PointStruct]) -> int:
        if not points:
            return 0
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def delete(self, dataset_id: str) -> None:
        if not self.client.collection_exists(self.collection_name):
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="dataset_id", match=MatchValue(value=dataset_id))]
            ),
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        dataset_id: str,
        doc_types: list[str] | None = None,
    ) -> list[Any]:
        conditions = [FieldCondition(key="dataset_id", match=MatchValue(value=dataset_id))]
        if doc_types:
            conditions.append(FieldCondition(key="doc_type", match=MatchAny(any=doc_types)))
        query_filter = Filter(must=conditions)
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
