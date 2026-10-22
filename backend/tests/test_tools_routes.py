"""Routes /api/v1/moves/{fen}, /api/v1/evaluate/{fen} et /api/v1/tools."""

from __future__ import annotations

from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_registry
from app.main import app
from tests.conftest import ITALIAN_FEN, LICHESS_URL, START_FEN, StubAnalyzer, load_fixture


@pytest.fixture
def client(make_registry):
    registry = make_registry(StubAnalyzer([20]))
    app.dependency_overrides[get_registry] = lambda: registry
    yield TestClient(app)
    app.dependency_overrides.clear()


def url(prefix: str, fen: str) -> str:
    return f"/api/v1/{prefix}/{quote(fen, safe='/')}"


def test_moves(client, respx_mock):
    respx_mock.get(f"{LICHESS_URL}/masters").respond(
        json=load_fixture("lichess/masters_italian.json")
    )

    response = client.get(url("moves", ITALIAN_FEN), params={"max_moves": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["opening"]["name"] == "Italian Game"
    assert len(body["moves"]) == 2


def test_moves_with_invalid_fen_is_422(client):
    response = client.get(url("moves", "8/8/8/8/8/8/8/K7 w - - 0 1"))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_arguments"


def test_moves_rate_limited_is_503_with_retry_after(client, respx_mock):
    respx_mock.get(f"{LICHESS_URL}/masters").respond(429, headers={"Retry-After": "60"})

    response = client.get(url("moves", ITALIAN_FEN))

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "60"
    assert response.json()["error"]["code"] == "rate_limited"


def test_moves_query_validation(client):
    response = client.get(url("moves", ITALIAN_FEN), params={"max_moves": 50})

    assert response.status_code == 422


def test_evaluate(client):
    response = client.get(url("evaluate", START_FEN), params={"lines": 1})

    assert response.status_code == 200
    assert response.json()["score_cp"] == 20
    assert response.json()["assessment"] == "Position équilibrée"


def test_evaluate_unavailable_engine_is_503(make_registry):
    from app.tools.errors import ToolUnavailableError

    registry = make_registry(
        StubAnalyzer(error=ToolUnavailableError(code="engine_not_found", message="absent"))
    )
    app.dependency_overrides[get_registry] = lambda: registry
    try:
        response = TestClient(app).get(url("evaluate", START_FEN))
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "Retry-After" not in response.headers


def test_tools_catalog(client):
    response = client.get("/api/v1/tools")

    assert response.status_code == 200
    names = {tool["name"] for tool in response.json()}
    assert "get_opening_theory" in names
    assert all("input_schema" in tool for tool in response.json())


def test_lifespan_builds_the_real_registry():
    with TestClient(app) as client:
        assert len(client.app.state.tool_registry.names) == 7
