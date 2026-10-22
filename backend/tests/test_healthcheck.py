from fastapi.testclient import TestClient
from pymongo.errors import ConnectionFailure

from app.main import app

client = TestClient(app)


def test_healthcheck():
    response = client.get("/api/v1/healthcheck")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_healthcheck_database_unavailable(monkeypatch):
    async def failing_ping() -> None:
        raise ConnectionFailure("no server")

    monkeypatch.setattr("app.main.ping_db", failing_ping)

    response = client.get("/api/v1/healthcheck/database")

    assert response.status_code == 503


def test_healthcheck_database_ok(monkeypatch):
    async def successful_ping() -> None:
        return None

    monkeypatch.setattr("app.main.ping_db", successful_ping)

    response = client.get("/api/v1/healthcheck/database")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
