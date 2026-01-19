import io

from fastapi.testclient import TestClient

from app.core.config import settings
from main import app


def _client_with_temp_storage(tmp_path) -> TestClient:
    settings.storage_dir = str(tmp_path)
    settings.max_upload_mb = 1
    settings.preview_max_rows = 50
    settings.query_max_rows = 100
    settings.sample_rows = 100
    return TestClient(app)


def _upload_dataset(client: TestClient, csv_content: str, delimiter: str | None = None) -> dict:
    files = {"file": ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    params = {"delimiter": delimiter} if delimiter else None
    response = client.post("/datasets/upload", files=files, params=params)
    assert response.status_code == 200
    return response.json()


def test_upload_dataset_with_delimiter(tmp_path) -> None:
    client = _client_with_temp_storage(tmp_path)
    csv_content = "col1;col2;country\n1;2;FR\n3;;US\n4;5;FR\n"
    data = _upload_dataset(client, csv_content, delimiter=";")

    assert data["dataset_id"]
    assert data["delimiter"] == ";"
    assert data["nb_rows"] == 3
    assert data["nb_columns"] == 3


def test_preview_endpoint_respects_limit(tmp_path) -> None:
    client = _client_with_temp_storage(tmp_path)
    csv_content = "col1;col2;country\n1;2;FR\n3;;US\n4;5;FR\n"
    dataset = _upload_dataset(client, csv_content, delimiter=";")

    response = client.get(f"/datasets/{dataset['dataset_id']}/preview", params={"limit": 2})

    assert response.status_code == 200
    data = response.json()
    assert data["columns"] == ["col1", "col2", "country"]
    assert len(data["rows"]) <= 2


def test_schema_endpoint_returns_stats(tmp_path) -> None:
    client = _client_with_temp_storage(tmp_path)
    csv_content = "col1;col2;country\n1;2;FR\n3;;US\n4;5;FR\n"
    dataset = _upload_dataset(client, csv_content, delimiter=";")

    response = client.get(f"/datasets/{dataset['dataset_id']}/schema")

    assert response.status_code == 200
    data = response.json()
    assert "dtypes" in data
    assert "missing_values_count" in data
    assert data["missing_values_count"]["col2"] == 1


def test_query_with_filters_and_unknown_column(tmp_path) -> None:
    client = _client_with_temp_storage(tmp_path)
    csv_content = "col1;col2;country\n1;2;FR\n3;;US\n4;5;FR\n"
    dataset = _upload_dataset(client, csv_content, delimiter=";")

    payload = {"columns": ["col1", "country"], "filters": {"country": "FR"}, "limit": 10}
    response = client.post(f"/datasets/{dataset['dataset_id']}/query", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["rows"]) == 2
    assert data["columns"] == ["col1", "country"]

    bad_payload = {"filters": {"unknown": "X"}}
    bad_response = client.post(f"/datasets/{dataset['dataset_id']}/query", json=bad_payload)

    assert bad_response.status_code == 400
