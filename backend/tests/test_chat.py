import io

from fastapi.testclient import TestClient

from app.core.config import settings
from app.llm.client import LLMResponse
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


class FakeLLMClient:
    def generate(self, messages: list[dict[str, str]]) -> LLMResponse:
        return LLMResponse(
            text="Answer with citations [dataset:fake rows:1-2].",
            prompt_tokens=12,
            response_tokens=8,
            total_tokens=20,
        )


def _client_with_temp_storage(tmp_path, monkeypatch, llm_configured: bool = True) -> TestClient:
    settings.storage_dir = str(tmp_path)
    settings.max_upload_mb = 1
    settings.preview_max_rows = 50
    settings.query_max_rows = 100
    settings.sample_rows = 100
    settings.rag_rows_per_doc = 2
    settings.rag_max_rows_to_index = 10
    settings.rag_embed_batch_size = 4
    settings.qdrant_path = str(tmp_path / "qdrant")
    settings.llm_provider = "openai_compatible"
    settings.llm_base_url = "http://llm.local"
    settings.llm_model = "test-model"
    settings.llm_api_key = "test-key" if llm_configured else None
    monkeypatch.setattr(
        "app.rag.embeddings.get_embedding_service",
        lambda: FakeEmbeddingService(),
    )
    monkeypatch.setattr("app.llm.provider.get_llm_client", lambda: FakeLLMClient())
    return TestClient(app)


def _upload_dataset(client: TestClient, csv_content: str, delimiter: str | None = None) -> dict:
    files = {"file": ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    params = {"delimiter": delimiter} if delimiter else None
    response = client.post("/datasets/upload", files=files, params=params)
    assert response.status_code == 200
    return response.json()


def _index_dataset(client: TestClient, dataset_id: str) -> None:
    response = client.post(f"/datasets/{dataset_id}/index", json={})
    assert response.status_code == 200


def test_chat_without_llm_config_returns_400(tmp_path, monkeypatch) -> None:
    client = _client_with_temp_storage(tmp_path, monkeypatch, llm_configured=False)
    csv_content = "col1;col2;country\n1;2;FR\n3;;US\n4;5;FR\n"
    dataset = _upload_dataset(client, csv_content, delimiter=";")
    _index_dataset(client, dataset["dataset_id"])

    response = client.post(
        "/chat",
        json={"dataset_id": dataset["dataset_id"], "message": "top countries"},
    )

    assert response.status_code == 400


def test_chat_dataset_not_indexed_returns_400(tmp_path, monkeypatch) -> None:
    client = _client_with_temp_storage(tmp_path, monkeypatch)
    csv_content = "col1;col2;country\n1;2;FR\n3;;US\n4;5;FR\n"
    dataset = _upload_dataset(client, csv_content, delimiter=";")

    response = client.post(
        "/chat",
        json={"dataset_id": dataset["dataset_id"], "message": "top countries"},
    )

    assert response.status_code == 400


def test_chat_with_mocked_llm_returns_structure(tmp_path, monkeypatch) -> None:
    client = _client_with_temp_storage(tmp_path, monkeypatch)
    csv_content = "col1;col2;country\n1;2;FR\n3;;US\n4;5;FR\n"
    dataset = _upload_dataset(client, csv_content, delimiter=";")
    _index_dataset(client, dataset["dataset_id"])

    response = client.post(
        "/chat",
        json={"dataset_id": dataset["dataset_id"], "message": "top countries", "top_k": 3},
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "contexts" in data
    assert "latency_ms" in data
    assert "prompt_tokens" in data
    assert "response_tokens" in data


def test_chat_returns_citations_when_context_present(tmp_path, monkeypatch) -> None:
    client = _client_with_temp_storage(tmp_path, monkeypatch)
    csv_content = "col1;col2;country\n1;2;FR\n3;;US\n4;5;FR\n"
    dataset = _upload_dataset(client, csv_content, delimiter=";")
    _index_dataset(client, dataset["dataset_id"])

    response = client.post(
        "/chat",
        json={"dataset_id": dataset["dataset_id"], "message": "country FR", "top_k": 3},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["citations"]) > 0
