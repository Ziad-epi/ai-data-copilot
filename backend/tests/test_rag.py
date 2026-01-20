import io
import json

from fastapi.testclient import TestClient

from app.core.config import settings
from main import app


class FakeEmbeddingService:
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            seed = sum(text.encode("utf-8")) % 1000
            vectors.append([float((seed + i) % 100) / 100.0 for i in range(self.dim)])
        return vectors


def _client_with_temp_storage(tmp_path, monkeypatch) -> TestClient:
    settings.storage_dir = str(tmp_path)
    settings.max_upload_mb = 1
    settings.preview_max_rows = 50
    settings.query_max_rows = 100
    settings.sample_rows = 100
    settings.rag_rows_per_doc = 2
    settings.rag_max_rows_to_index = 10
    settings.rag_embed_batch_size = 4
    settings.qdrant_path = str(tmp_path / "qdrant")
    monkeypatch.setattr(
        "app.rag.embeddings.get_embedding_service",
        lambda: FakeEmbeddingService(),
    )
    return TestClient(app)


def _upload_dataset(client: TestClient, csv_content: str, delimiter: str | None = None) -> dict:
    files = {"file": ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    params = {"delimiter": delimiter} if delimiter else None
    response = client.post("/datasets/upload", files=files, params=params)
    assert response.status_code == 200
    return response.json()


def test_index_creates_metadata(tmp_path, monkeypatch) -> None:
    client = _client_with_temp_storage(tmp_path, monkeypatch)
    csv_content = "col1;col2;country\n1;2;FR\n3;;US\n4;5;FR\n"
    dataset = _upload_dataset(client, csv_content, delimiter=";")

    response = client.post(f"/datasets/{dataset['dataset_id']}/index", json={})

    assert response.status_code == 200
    data = response.json()
    assert data["nb_docs"] > 0
    metadata_path = tmp_path / dataset["dataset_id"] / "index_metadata.json"
    assert metadata_path.exists()


def test_search_returns_results(tmp_path, monkeypatch) -> None:
    client = _client_with_temp_storage(tmp_path, monkeypatch)
    csv_content = "col1;col2;country\n1;2;FR\n3;;US\n4;5;FR\n"
    dataset = _upload_dataset(client, csv_content, delimiter=";")

    index_response = client.post(f"/datasets/{dataset['dataset_id']}/index", json={})
    assert index_response.status_code == 200

    search_response = client.post(
        f"/datasets/{dataset['dataset_id']}/search",
        json={"query": "country FR", "top_k": 3, "doc_types": ["summary", "rows"]},
    )

    assert search_response.status_code == 200
    results = search_response.json()["results"]
    assert 0 < len(results) <= 3
    for result in results:
        assert result["source"]["dataset_id"] == dataset["dataset_id"]
        assert "dataset:" in result["citation"]


def test_search_without_index_returns_400(tmp_path, monkeypatch) -> None:
    client = _client_with_temp_storage(tmp_path, monkeypatch)
    csv_content = "col1;col2;country\n1;2;FR\n3;;US\n4;5;FR\n"
    dataset = _upload_dataset(client, csv_content, delimiter=";")

    response = client.post(
        f"/datasets/{dataset['dataset_id']}/search",
        json={"query": "country FR", "top_k": 3},
    )

    assert response.status_code == 400


def test_reindex_replaces_index(tmp_path, monkeypatch) -> None:
    client = _client_with_temp_storage(tmp_path, monkeypatch)
    csv_content = "col1;col2;country\n1;2;FR\n3;;US\n4;5;FR\n5;6;DE\n"
    dataset = _upload_dataset(client, csv_content, delimiter=";")

    first = client.post(
        f"/datasets/{dataset['dataset_id']}/index",
        json={"rows_per_doc": 2},
    )
    assert first.status_code == 200
    metadata_path = tmp_path / dataset["dataset_id"] / "index_metadata.json"
    first_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    second = client.post(
        f"/datasets/{dataset['dataset_id']}/index",
        json={"rows_per_doc": 1, "reindex": True},
    )
    assert second.status_code == 200
    second_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert first_metadata["nb_docs"] != second_metadata["nb_docs"]
