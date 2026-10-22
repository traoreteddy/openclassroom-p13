"""Adaptateur ToolSpec → StructuredTool LangChain (option C, § 10.5 du contrat)."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.agent.langchain_tools import build_langchain_tools, to_langchain_tool
from app.tools.errors import ToolUnavailableError
from tests.conftest import ITALIAN_FEN, LICHESS_URL, START_FEN, StubAnalyzer, load_fixture

MASTERS = f"{LICHESS_URL}/masters"
ILLEGAL_FEN = "8/8/8/8/8/8/8/K7 w - - 0 1"


def tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    """Appel d'outil tel que le produit un LLM (format LangChain)."""
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


@pytest.fixture
def registry(make_registry):
    return make_registry(StubAnalyzer([20]))


@pytest.fixture
def tools(registry):
    return {tool.name: tool for tool in build_langchain_tools(registry)}


# --- Une seule source de vérité --------------------------------------------------------


def test_one_langchain_tool_per_spec(registry, tools):
    assert list(tools) == registry.names


@pytest.mark.parametrize(
    "name",
    [
        "inspect_position",
        "get_opening_theory",
        "evaluate_position",
        "search_opening_knowledge",
        "find_opening_videos",
        "start_deep_analysis",
        "get_deep_analysis",
    ],
)
def test_generated_tool_mirrors_its_spec(registry, tools, name):
    spec, tool = registry.get(name), tools[name]

    assert tool.description == spec.description
    assert tool.args_schema is spec.input_model
    llm_schema = convert_to_openai_tool(tool)["function"]["parameters"]
    assert set(llm_schema["properties"]) == set(spec.input_model.model_fields)


# --- Succès ----------------------------------------------------------------------------


async def test_success_returns_content_for_llm_and_artifact_for_app(tools, respx_mock):
    respx_mock.get(MASTERS).respond(json=load_fixture("lichess/masters_italian.json"))

    message = await tools["get_opening_theory"].ainvoke(
        tool_call("get_opening_theory", {"fen": ITALIAN_FEN})
    )

    assert isinstance(message, ToolMessage)
    assert message.status == "success"
    assert message.tool_call_id == "call_1"
    assert json.loads(message.content)["opening"]["name"] == "Italian Game"
    assert message.artifact["ok"] is True
    assert message.artifact["data"]["moves"][0]["san"] == "Bc5"
    assert message.artifact["duration_ms"] >= 0


async def test_defaults_of_the_schema_are_applied(tools, respx_mock):
    route = respx_mock.get(MASTERS).respond(json=load_fixture("lichess/masters_italian.json"))

    await tools["get_opening_theory"].ainvoke(tool_call("get_opening_theory", {"fen": ITALIAN_FEN}))

    params = route.calls.last.request.url.params
    assert params["moves"] == "5"
    assert params["topGames"] == "0"


async def test_plain_invoke_returns_only_the_content(tools):
    content = await tools["evaluate_position"].ainvoke({"fen": START_FEN})

    assert json.loads(content)["score_cp"] == 20


# --- Arguments invalides : notre message, pas celui de LangChain ------------------------


@pytest.mark.parametrize(
    ("name", "args", "fragment"),
    [
        ("get_opening_theory", {"fen": ILLEGAL_FEN}, "pas de roi noir"),
        ("get_opening_theory", {"fen": ITALIAN_FEN, "moves": 5}, "Extra inputs"),
        ("evaluate_position", {"fen": START_FEN, "lines": 9}, "lines"),
        ("search_opening_knowledge", {"query": START_FEN}, "query contient une FEN"),
    ],
    ids=["fen-illegale", "champ-invente", "hors-bornes", "fen-comme-question"],
)
async def test_invalid_arguments_get_actionable_message(registry, tools, name, args, fragment):
    message = await tools[name].ainvoke(tool_call(name, args))

    assert message.status == "error"
    assert message.content.startswith("[invalid_arguments]")
    assert fragment in message.content
    assert "Exemple valide" in message.content
    # Même message que par le registre : le chemin d'appel ne change rien pour le modèle.
    assert message.content == (await registry.execute(name, args)).to_model_content()


async def test_invalid_arguments_never_reach_the_service(tools, respx_mock):
    route = respx_mock.get(MASTERS)

    await tools["get_opening_theory"].ainvoke(tool_call("get_opening_theory", {"fen": ILLEGAL_FEN}))

    assert not route.called


# --- Pannes : le graphe n'est pas interrompu --------------------------------------------


async def test_outage_is_reported_to_the_model_without_raising(tools, respx_mock):
    route = respx_mock.get(MASTERS).respond(503)

    message = await tools["get_opening_theory"].ainvoke(
        tool_call("get_opening_theory", {"fen": ITALIAN_FEN})
    )

    assert message.status == "error"
    assert message.content.startswith("[retries_exhausted]")
    assert "Conseil" in message.content
    assert route.call_count == 3  # la résilience du registre s'applique


async def test_unavailable_engine_is_reported(make_registry):
    registry = make_registry(
        StubAnalyzer(error=ToolUnavailableError(code="engine_not_found", message="absent"))
    )
    tool = to_langchain_tool(registry, registry.get("evaluate_position"))

    message = await tool.ainvoke(tool_call("evaluate_position", {"fen": START_FEN}))

    assert message.status == "error"
    assert message.content == "[engine_not_found] absent"


async def test_breaker_state_is_shared_with_the_registry(registry, tools, respx_mock):
    respx_mock.get(MASTERS).respond(503)

    for _ in range(5):
        await registry.execute("get_opening_theory", {"fen": ITALIAN_FEN})
    message = await tools["get_opening_theory"].ainvoke(
        tool_call("get_opening_theory", {"fen": ITALIAN_FEN})
    )

    assert message.content.startswith("[circuit_open]")
