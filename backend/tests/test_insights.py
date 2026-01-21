import io
import time

from fastapi.testclient import TestClient

from app.core.config import settings
from app.llm.client import LLMResponse
from main import app


class FakeLLMClient:
    def generate(self, messages: list[dict[str, str]]) -> LLMResponse:
        return LLMResponse(
            text="# Executive Report\n\nLLM summary.\n",
            prompt_tokens=10,
            response_tokens=20,
            total_tokens=30,
        )


def _client_with_temp_storage(tmp_path, llm_configured: bool = False) -> TestClient:
    settings.storage_dir = str(tmp_path)
    settings.max_upload_mb = 1
    settings.preview_max_rows = 50
    settings.query_max_rows = 100
    settings.sample_rows = 100
    settings.insights_sample_max = 100
    settings.insights_missing_threshold = 0.3
    settings.insights_outlier_method = "iqr"
    settings.charts_max_points = 20
    settings.llm_provider = "openai_compatible"
    settings.llm_base_url = "http://llm.local"
    settings.llm_model = "test-model"
    settings.llm_api_key = "test-key" if llm_configured else None
    return TestClient(app)


def _upload_dataset(client: TestClient) -> dict:
    csv_content = (
        "col1;col2;category;date;note\n"
        "1;10;A;2024-01-01;short\n"
        "2;;B;2024-01-02;this is a long text value for testing\n"
        "3;1000;A;not_a_date;another note\n"
        "4;5;C;2024-01-04;ok\n"
    )
    files = {"file": ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/datasets/upload", files=files, params={"delimiter": ";"})
    assert response.status_code == 200
    return response.json()


def test_insights_creates_file_and_structure(tmp_path) -> None:
    client = _client_with_temp_storage(tmp_path)
    dataset = _upload_dataset(client)

    response = client.post(
        f"/datasets/{dataset['dataset_id']}/insights",
        json={"sample_rows": 10, "force_recompute": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert "dataset_overview" in data
    assert "column_profiles" in data
    insights_path = tmp_path / dataset["dataset_id"] / "insights.json"
    assert insights_path.exists()


def test_insights_cache_returns_same_generated_at(tmp_path) -> None:
    client = _client_with_temp_storage(tmp_path)
    dataset = _upload_dataset(client)

    first = client.post(
        f"/datasets/{dataset['dataset_id']}/insights",
        json={"sample_rows": 10},
    )
    assert first.status_code == 200
    first_generated = first.json()["generated_at"]

    time.sleep(0.01)

    second = client.post(
        f"/datasets/{dataset['dataset_id']}/insights",
        json={"sample_rows": 10},
    )
    assert second.status_code == 200
    assert second.json()["generated_at"] == first_generated


def test_charts_suggest_max_charts_and_preview_size(tmp_path) -> None:
    client = _client_with_temp_storage(tmp_path)
    dataset = _upload_dataset(client)

    response = client.post(
        f"/datasets/{dataset['dataset_id']}/charts/suggest",
        json={"question": "distribution", "max_charts": 2},
    )

    assert response.status_code == 200
    charts = response.json()["charts"]
    assert len(charts) <= 2
    for chart in charts:
        for values in chart["data_preview"].values():
            assert len(values) <= settings.charts_max_points


def test_report_without_llm_creates_report(tmp_path) -> None:
    client = _client_with_temp_storage(tmp_path, llm_configured=False)
    dataset = _upload_dataset(client)

    response = client.post(f"/datasets/{dataset['dataset_id']}/report")

    assert response.status_code == 200
    data = response.json()
    assert data["used_llm"] is False
    report_path = tmp_path / dataset["dataset_id"] / "report.md"
    assert report_path.exists()
    assert data["report_markdown"]


def test_report_with_llm_mocked(tmp_path, monkeypatch) -> None:
    client = _client_with_temp_storage(tmp_path, llm_configured=True)
    monkeypatch.setattr("app.llm.provider.get_llm_client", lambda: FakeLLMClient())
    dataset = _upload_dataset(client)

    response = client.post(f"/datasets/{dataset['dataset_id']}/report")

    assert response.status_code == 200
    data = response.json()
    assert data["used_llm"] is True
    report_path = tmp_path / dataset["dataset_id"] / "report.md"
    assert report_path.exists()
