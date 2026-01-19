import io

from fastapi.testclient import TestClient

from app.core.config import settings
from main import app


def test_upload_dataset_returns_metadata(tmp_path) -> None:
    settings.storage_dir = str(tmp_path)
    settings.max_upload_mb = 1
    client = TestClient(app)

    csv_content = "col1,col2\n1,2\n3,4\n"
    files = {"file": ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/datasets/upload", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["dataset_id"]
    assert data["filename"] == "test.csv"
    assert data["nb_rows"] == 2
    assert data["nb_columns"] == 2
    assert data["columns"] == ["col1", "col2"]


def test_get_dataset_metadata(tmp_path) -> None:
    settings.storage_dir = str(tmp_path)
    settings.max_upload_mb = 1
    client = TestClient(app)

    csv_content = "col1,col2\n1,2\n3,4\n"
    files = {"file": ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    upload_response = client.post("/datasets/upload", files=files)
    dataset_id = upload_response.json()["dataset_id"]

    response = client.get(f"/datasets/{dataset_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["dataset_id"] == dataset_id
    assert data["filename"] == "test.csv"
