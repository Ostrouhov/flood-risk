"""Smoke-тесты API. На Этапе 0 — только /health и индексная страница."""

from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "timestamp" in body


def test_index_renders(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "floodrisk" in response.text


def test_predict_not_implemented_yet(client: TestClient) -> None:
    response = client.post("/api/predict")
    assert response.status_code == 501


def test_scenarios_empty_before_seed(client: TestClient) -> None:
    response = client.get("/api/scenarios")
    assert response.status_code == 200
    assert response.json() == []
