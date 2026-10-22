"""Graphe de l'agent : parcours « position » (Lichess → Stockfish), parcours « question »
(sélection native / structurée, flot fixe), journal, garde-fous.

Le modèle est scripté (aucun appel à Ollama) ; les outils passent par le registre de test
(Lichess et YouTube simulés par respx, Milvus et Stockfish en stubs).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import Field

from app.agent.graph import AgentConfig, ChessAgent
from app.agent.journal import Pricing
from app.agent.tracing import configure_tracing
from tests.conftest import (
    ITALIAN_FEN,
    LICHESS_URL,
    START_FEN,
    YOUTUBE_URL,
    StubAnalyzer,
    StubVectorStore,
    load_fixture,
    passage,
)


class ScriptedChat(BaseChatModel):
    """Renvoie les réponses dans l'ordre ; garde la trace des appels (messages, outils liés)."""

    script: list[AIMessage]
    calls: list[dict[str, Any]] = Field(default_factory=list)
    structured_schema: dict[str, Any] | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedChat:
        self.calls.append({"bound_tools": [t.name for t in tools]})
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append({"messages": [m.content for m in messages]})
        message = self.script.pop(0) if self.script else AIMessage(content="(script épuisé)")
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def llm_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if "messages" in c]


def reply(content: str, tokens: tuple[int, int] = (100, 20)) -> AIMessage:
    message = AIMessage(content=content)
    message.usage_metadata = {
        "input_tokens": tokens[0],
        "output_tokens": tokens[1],
        "total_tokens": sum(tokens),
    }
    return message


def tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> AIMessage:
    message = reply("")
    message.tool_calls = [{"name": name, "args": args, "id": call_id}]
    return message


def structured(tool: str | None, arguments: dict[str, Any], reason: str = "parce que"):
    return reply(json.dumps({"tool": tool, "arguments": arguments, "reason": reason}))


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def tracing(exporter):
    return configure_tracing(exporter=exporter, pricing=Pricing(1.0, 2.0))


@pytest.fixture
def registry(make_registry):
    return make_registry(
        StubAnalyzer([25]),
        StubVectorStore(
            [
                passage("Les Noirs contestent le centre.", 0.8),
                passage("Le Fou c4 vise f7.", 0.75, opening="Italian Game"),
            ]
        ),
    )


def make_agent(registry, tracing, script, **config) -> tuple[ChessAgent, ScriptedChat]:
    chat = ScriptedChat(script=list(script), calls=[])

    def factory(structured_schema=None):
        if structured_schema is not None:
            chat.structured_schema = structured_schema
        return chat

    return ChessAgent(registry, tracing, factory, AgentConfig(**config)), chat


# --- Parcours « position » : Lichess d'abord, Stockfish si hors théorie --------------------


async def test_position_in_theory_runs_the_full_pipeline(registry, tracing, respx_mock):
    respx_mock.get(f"{LICHESS_URL}/masters").respond(
        json=load_fixture("lichess/masters_italian.json")
    )
    respx_mock.get(f"{YOUTUBE_URL}/search").respond(
        json=load_fixture("youtube/search_italian.json")
    )
    agent, chat = make_agent(registry, tracing, [reply("Jouez Fc5, c'est l'Italienne.")])

    result = await agent.analyze(ITALIAN_FEN, session_id="s1")

    assert result.tools_used == [
        "inspect_position",
        "get_opening_theory",
        "evaluate_position",
        "search_opening_knowledge",
        "find_opening_videos",
    ]
    assert result.in_theory is True
    assert result.opening == "Italian Game"
    assert result.tool_calls[2].arguments == {"fen": ITALIAN_FEN, "depth": 12, "lines": 1}
    assert result.tool_calls[3].arguments["opening"] == "Italian Game"
    assert result.data["theory"]["moves"][0]["san"] == "Bc5"
    assert result.data["evaluation"]["score_cp"] == 25
    assert result.data["context"]["found"] is True
    assert len(result.data["videos"]["videos"]) == 2
    assert result.answer == "Jouez Fc5, c'est l'Italienne."
    assert len(chat.llm_calls) == 1  # le modèle ne fait que rédiger
    assert "coups théoriques (Lichess)" in chat.llm_calls[0]["messages"][0]
    assert result.decisions == 1  # aucune décision d'outil n'est laissée au modèle


async def test_position_out_of_theory_uses_stockfish_only(registry, tracing, respx_mock):
    respx_mock.get(f"{LICHESS_URL}/masters").respond(json=load_fixture("lichess/lichess_rare.json"))
    agent, _ = make_agent(registry, tracing, [reply("Hors théorie : +0,25 pour les Blancs.")])

    result = await agent.analyze(START_FEN, session_id="s1")

    assert result.tools_used == ["inspect_position", "get_opening_theory", "evaluate_position"]
    assert result.in_theory is False
    assert result.opening is None
    assert result.tool_calls[2].arguments == {"fen": START_FEN, "depth": 18, "lines": 3}
    assert "context" not in result.data and "videos" not in result.data


async def test_lichess_outage_falls_back_to_stockfish(registry, tracing, respx_mock):
    route = respx_mock.get(f"{LICHESS_URL}/masters").respond(503)
    agent, _ = make_agent(registry, tracing, [reply("Lichess indisponible ; Stockfish dit…")])

    result = await agent.analyze(ITALIAN_FEN, session_id="s1")

    assert result.tool_calls[1].status == "error"
    assert result.data["theory"]["error"].startswith("[retries_exhausted]")
    assert result.in_theory is False
    assert result.tools_used[2] == "evaluate_position"
    assert route.call_count == 3  # la résilience du registre s'est appliquée


async def test_invalid_fen_still_produces_an_answer(registry, tracing, respx_mock):
    lichess = respx_mock.get(f"{LICHESS_URL}/masters")
    agent, _ = make_agent(registry, tracing, [reply("La position fournie est invalide.")])

    result = await agent.analyze("8/8/8/8/8/8/8/K7 w - - 0 1", session_id="s1")

    assert all(call.status == "error" for call in result.tool_calls)
    assert all("invalid_arguments" in call.outcome for call in result.tool_calls)
    assert result.answer == "La position fournie est invalide."
    assert lichess.call_count == 0  # Lichess n'est jamais appelé avec une FEN invalide


# --- Parcours « question », sélection native -----------------------------------------------


async def test_native_agent_calls_a_tool_then_answers(registry, tracing):
    agent, chat = make_agent(
        registry,
        tracing,
        [
            tool_call("search_opening_knowledge", {"query": "idées de la sicilienne"}),
            reply("Les Noirs contestent le centre avec c5."),
        ],
    )

    result = await agent.ask("Pourquoi c5 dans la Sicilienne ?", session_id="s1")

    assert result.tools_used == ["search_opening_knowledge"]
    assert result.tool_calls[0].status == "success"
    assert "2 passage(s)" in result.tool_calls[0].outcome
    assert result.answer == "Les Noirs contestent le centre avec c5."
    assert result.decisions == 2
    assert result.tokens_in == 200 and result.tokens_out == 40
    assert chat.calls[0]["bound_tools"] == registry.names
    assert len(result.trace_id) == 32


async def test_native_agent_answers_directly_without_tool(registry, tracing):
    agent, chat = make_agent(registry, tracing, [reply("Bonjour ! Comment puis-je aider ?")])

    result = await agent.ask("Bonjour", session_id="s1")

    assert result.tools_used == []
    assert result.answer == "Bonjour ! Comment puis-je aider ?"
    assert len(chat.llm_calls) == 1  # pas de 2e appel


async def test_tool_error_is_given_back_to_the_model(registry, tracing):
    agent, _ = make_agent(
        registry,
        tracing,
        [tool_call("evaluate_position", {"fen": "pas une fen"}), reply("FEN invalide.")],
    )

    result = await agent.ask("Évalue : pas une fen", session_id="s1")

    assert result.tool_calls[0].status == "error"
    assert result.tool_calls[0].outcome.startswith("[invalid_arguments]")
    assert result.answer == "FEN invalide."


async def test_unknown_tool_is_reported_not_raised(registry, tracing):
    agent, _ = make_agent(
        registry, tracing, [tool_call("outil_imaginaire", {}), reply("Je ne peux pas.")]
    )

    result = await agent.ask("?", session_id="s1")

    assert result.tool_calls[0].status == "error"
    assert "unknown_tool" in result.tool_calls[0].outcome


async def test_tool_rounds_are_capped(registry, tracing):
    loop = [tool_call("inspect_position", {"fen": START_FEN}, f"c{i}") for i in range(2)]
    agent, chat = make_agent(
        registry, tracing, [*loop, reply("Réponse finale.")], max_tool_rounds=2
    )

    result = await agent.ask("Boucle", session_id="s1")

    assert len(result.tool_calls) == 2
    assert result.answer == "Réponse finale."
    assert len(chat.llm_calls) == 3  # 2 décisions + réponse : la 3e décision est court-circuitée


# --- Parcours « question », sélection structurée -------------------------------------------


async def test_structured_selection_rebuilds_the_tool_call(registry, tracing):
    agent, chat = make_agent(
        registry,
        tracing,
        [
            structured("inspect_position", {"fen": START_FEN}, "vérifier la position"),
            structured(None, {}, "assez d'informations"),
            reply("20 coups légaux."),
        ],
        selection="structured",
    )

    result = await agent.ask("Quels coups sont légaux au départ ?", session_id="s1")

    assert result.tools_used == ["inspect_position"]
    assert result.tool_calls[0].status == "success"
    assert result.answer == "20 coups légaux."  # la justification n'est pas la réponse
    assert chat.structured_schema["properties"]["tool"]["enum"][:2] == registry.names[:2]
    assert "Outils disponibles" in chat.llm_calls[0]["messages"][0]


async def test_structured_unknown_tool_or_bad_json_leads_to_answer(registry, tracing):
    agent, _ = make_agent(
        registry,
        tracing,
        [structured("outil_imaginaire", {}, "hmm"), reply("Je réponds sans outil.")],
        selection="structured",
    )
    result = await agent.ask("?", session_id="s1")
    assert result.tools_used == [] and result.answer == "Je réponds sans outil."

    agent, _ = make_agent(
        registry, tracing, [reply("pas du json"), reply("Réponse.")], selection="structured"
    )
    result = await agent.ask("?", session_id="s2")
    assert result.tools_used == [] and result.answer == "Réponse."


async def test_structured_reason_is_kept_in_the_journal(registry, tracing):
    agent, _ = make_agent(
        registry,
        tracing,
        [structured(None, {}, "question de politesse"), reply("Salut !")],
        selection="structured",
    )

    await agent.ask("Bonjour", session_id="s1")

    assert tracing.journal.for_session("s1")[0].outcome == "question de politesse"


# --- Flot fixe -------------------------------------------------------------------------------


async def test_fixed_flow_always_searches(registry, tracing):
    agent, chat = make_agent(
        registry, tracing, [reply("Réponse après recherche.")], retrieval="fixed"
    )

    result = await agent.ask("Bonjour", session_id="s1")

    assert result.tools_used == ["search_opening_knowledge"]
    assert result.tool_calls[0].arguments == {"query": "Bonjour"}
    assert result.decisions == 1  # la recherche n'est pas une décision du modèle
    assert result.answer == "Réponse après recherche."
    assert len(chat.llm_calls) == 1


async def test_fixed_flow_still_routes_positions_deterministically(registry, tracing, respx_mock):
    respx_mock.get(f"{LICHESS_URL}/masters").respond(json=load_fixture("lichess/lichess_rare.json"))
    agent, _ = make_agent(registry, tracing, [reply("Hors théorie.")], retrieval="fixed")

    result = await agent.analyze(START_FEN, session_id="s1")

    assert result.tools_used == ["inspect_position", "get_opening_theory", "evaluate_position"]


# --- Première décision (évaluation Tâches-30) -----------------------------------------------


async def test_first_decision_does_not_run_tools(registry, tracing):
    agent, _ = make_agent(
        registry, tracing, [tool_call("evaluate_position", {"fen": START_FEN, "lines": 3})]
    )

    decision = await agent.first_decision("Les 3 meilleurs coups ?", session_id="s1")

    assert decision.tool == "evaluate_position"
    assert decision.arguments == {"fen": START_FEN, "lines": 3}
    assert decision.tokens_in == 100
    entries = tracing.journal.for_session("s1")
    assert entries[0].kind == "decision"
    assert not any(e.kind == "tool_call" for e in entries)


async def test_first_decision_without_tool(registry, tracing):
    agent, _ = make_agent(registry, tracing, [reply("Bonjour !")])

    assert (await agent.first_decision("Bonjour", session_id="s1")).tool is None


# --- Journal et traces -----------------------------------------------------------------------


async def test_journal_entries_follow_the_schema(registry, tracing, exporter):
    agent, _ = make_agent(
        registry,
        tracing,
        [tool_call("search_opening_knowledge", {"query": "idées"}), reply("Réponse.")],
    )

    await agent.ask("Idées de la sicilienne ?", session_id="session-42")
    tracing.flush()

    entries = tracing.journal.for_session("session-42")
    assert [(e.step, e.kind, e.choice) for e in entries] == [
        (1, "decision", "search_opening_knowledge"),
        (2, "tool_call", "search_opening_knowledge"),
        (3, "decision", "answer"),
    ]
    decision = entries[0]
    assert decision.arguments == {"query": "idées"}
    assert decision.tokens_in == 100 and decision.tokens_out == 20
    assert decision.cost_eur == pytest.approx((100 * 1.0 + 20 * 2.0) / 1e6)
    assert decision.status == "success" and decision.outcome == "1 appel(s) d'outil"
    assert len({e.trace_id for e in entries}) == 1
    assert all(e.duration_ms >= 0 for e in entries)
    assert set(json.loads(decision.to_json())) == {
        "timestamp",
        "session_id",
        "trace_id",
        "step",
        "kind",
        "choice",
        "arguments",
        "status",
        "duration_ms",
        "tokens_in",
        "tokens_out",
        "cost_eur",
        "outcome",
    }

    root = next(s for s in exporter.get_finished_spans() if s.name == "agent_turn")
    assert root.attributes["langfuse.session.id"] == "session-42"
    assert root.attributes["p13.retrieval"] == "agent"


async def test_position_trace_has_one_tool_span_per_step(registry, tracing, respx_mock):
    respx_mock.get(f"{LICHESS_URL}/masters").respond(json=load_fixture("lichess/lichess_rare.json"))
    agent, _ = make_agent(registry, tracing, [reply("Réponse.")])

    await agent.analyze(START_FEN, session_id="pos-1")

    kinds = [(e.kind, e.choice) for e in tracing.journal.for_session("pos-1")]
    assert kinds == [
        ("tool_call", "inspect_position"),
        ("tool_call", "get_opening_theory"),
        ("tool_call", "evaluate_position"),
        ("answer", "answer"),
    ]
    assert tracing.journal.for_session("pos-1")[2].outcome == "Position équilibrée"


async def test_journal_is_written_to_disk(registry, tmp_path, exporter):
    tracing = configure_tracing(journal_path=tmp_path / "j.jsonl", exporter=exporter)
    agent, _ = make_agent(registry, tracing, [reply("Salut !")])

    await agent.ask("Bonjour", session_id="s1")

    lines = (tmp_path / "j.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["choice"] == "answer"


def test_tracing_without_langfuse_has_no_callbacks(tracing):
    assert tracing.langfuse is None
    assert tracing.langchain_callbacks() == []
