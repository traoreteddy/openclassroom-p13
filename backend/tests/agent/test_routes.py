"""Routes /api/v1/agent/analyze et /api/v1/agent/ask (agent remplacé par une doublure)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agent.graph import AgentResult, ToolCallRecord
from app.api.routes.agent import get_agent
from app.main import app
from tests.conftest import ITALIAN_FEN


class FakeAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def analyze(self, fen, *, question=None, session_id=None):
        self.calls.append(("analyze", {"fen": fen, "question": question, "session_id": session_id}))
        return AgentResult(
            answer="Jouez Fc5.",
            tool_calls=[ToolCallRecord("get_opening_theory", {"fen": fen}, "success", "in_theory")],
            decisions=1,
            tokens_in=300,
            tokens_out=50,
            trace_id="ab" * 16,
            in_theory=True,
            opening="Italian Game",
            data={"theory": {"in_theory": True}},
        )

    async def ask(self, question, *, session_id=None):
        self.calls.append(("ask", {"question": question, "session_id": session_id}))
        return AgentResult(answer="Bonjour !", trace_id="cd" * 16)


@pytest.fixture
def client():
    agent = FakeAgent()
    app.dependency_overrides[get_agent] = lambda: agent
    try:
        yield TestClient(app), agent
    finally:
        app.dependency_overrides.clear()


def test_analyze_returns_the_structured_result(client):
    http, agent = client

    response = http.post(
        "/api/v1/agent/analyze", json={"fen": ITALIAN_FEN, "session_id": "partie-1"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Jouez Fc5."
    assert body["in_theory"] is True and body["opening"] == "Italian Game"
    assert body["tool_calls"][0]["name"] == "get_opening_theory"
    assert body["data"]["theory"]["in_theory"] is True
    assert body["trace_id"] == "ab" * 16
    assert agent.calls == [
        ("analyze", {"fen": ITALIAN_FEN, "question": None, "session_id": "partie-1"})
    ]


def test_analyze_validates_the_fen_before_calling_the_agent(client):
    http, agent = client

    response = http.post("/api/v1/agent/analyze", json={"fen": "8/8/8/8/8/8/8/K7 w - - 0 1"})

    assert response.status_code == 422
    assert "pas de roi noir" in response.text
    assert agent.calls == []


def test_ask(client):
    http, agent = client

    response = http.post("/api/v1/agent/ask", json={"question": "Bonjour"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Bonjour !"
    assert agent.calls == [("ask", {"question": "Bonjour", "session_id": None})]


def test_ask_rejects_empty_question(client):
    http, _ = client

    assert http.post("/api/v1/agent/ask", json={"question": ""}).status_code == 422


def test_lifespan_builds_the_agent(monkeypatch):
    from app.tools.catalog import ToolDependencies

    monkeypatch.setattr(ToolDependencies, "start_warm_up", lambda deps: None)
    with TestClient(app) as http:
        agent = http.app.state.agent
        assert agent.config.selection in ("native", "structured")
        assert set(agent.tools) == set(http.app.state.tool_registry.names)
