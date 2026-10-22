"""Graphe LangGraph de l'agent.

Deux entrées, selon que la requête porte une position (FEN) ou non :

    START ─ FEN fournie ?
      ├─ oui ─► inspect ► theory (Lichess) ─ en théorie ?
      │                    ├─ oui ─► quick_eval (Stockfish) ► context (RAG) ► videos ► answer
      │                    └─ non, ou Lichess en panne ─► engine (Stockfish) ► answer
      └─ non (question libre) ─► decide ⇄ tools ► answer

**Parcours « position »** (énoncé, `docs/processus.md` § 2) : déterministe, aucun choix
laissé au modèle. Lichess donne les coups théoriques ; si la position est sortie de la
théorie (ou si Lichess est indisponible), Stockfish devient la source principale. En
théorie, Stockfish est aussi appelé en complément, à profondeur réduite (grille
d'évaluation : « Stockfish évalue les positions théoriques »). Le contexte (RAG) et les
vidéos ne sont cherchés que si l'ouverture a un nom. Le modèle rédige ensuite la réponse
à partir des résultats d'outils.

**Parcours « question »** : retrieval décidé par l'agent (`selection` native ou
structurée) ou à flot fixe (`retrieval="fixed"`, une recherche systématique), comparés en
phase 5 de `docs/outil-rag.md`.

Chaque étape est un span OpenTelemetry (`decision`, `tool_call`, `answer`) : le journal de
décision et Langfuse en découlent. Le registre d'outils garantit validation, délais,
tentatives et disjoncteur ; une erreur d'outil revient comme message et ne bloque rien.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agent.langchain_tools import build_langchain_tools
from app.agent.llm import LLMFactory, usage
from app.agent.tracing import Tracing, record_outcome, step_span, turn_span
from app.tools.base import ToolRegistry

Retrieval = Literal["agent", "fixed"]
Selection = Literal["native", "structured"]

SEARCH_TOOL = "search_opening_knowledge"
QUICK_EVAL_DEPTH = 12  # complément en théorie : rapide
ENGINE_DEPTH = 18  # hors théorie : Stockfish est la source principale

SYSTEM_PROMPT = """\
Tu es un coach d'échecs pour de jeunes joueurs de la Fédération française des échecs.
Tu disposes d'outils. Règles de décision :
- question sur une ouverture (idées, plans, histoire, pièges, pourquoi tel coup) :
  appelle search_opening_knowledge, ne réponds pas de mémoire ;
- une position FEN est fournie : légalité d'un coup ou état de la partie → inspect_position ;
  qui est mieux ou meilleurs coups → evaluate_position ; coups des maîtres, nom de
  l'ouverture ou parties de référence → get_opening_theory ; analyse approfondie demandée
  explicitement → start_deep_analysis ;
- demande de vidéo → find_opening_videos ;
- salutations, remerciements, questions hors échecs ou de culture générale : réponds
  directement, sans outil.
N'invente jamais le résultat d'un outil ; si un outil échoue, dis-le simplement.
Réponds en français, de façon claire et courte."""

POSITION_PROMPT = """\
Tu es un coach d'échecs pour de jeunes joueurs. On te donne les résultats d'outils sur une
position : coups théoriques (Lichess), évaluation (Stockfish), contexte de l'ouverture et
vidéos. Rédige un conseil pédagogique court en français :
1. si la position est en théorie : nomme l'ouverture, propose les 2 ou 3 coups théoriques
   les plus joués avec leurs statistiques, explique l'idée en une ou deux phrases ;
2. si elle est hors théorie : dis-le, donne l'évaluation et le meilleur coup de Stockfish,
   et explique brièvement ;
3. mentionne une vidéo si une est fournie.
N'invente rien qui ne soit pas dans les résultats ; si un outil a échoué, signale-le."""

STRUCTURED_PROMPT = """\
{system}

Décide de la prochaine action. Outils disponibles :
{catalog}

Réponds uniquement en JSON :
- "tool" : le nom de l'outil à appeler, ou null pour répondre sans outil ;
- "arguments" : les paramètres de l'outil (objet JSON, vide si aucun outil) ;
- "reason" : une justification courte."""


@dataclass(frozen=True)
class AgentConfig:
    retrieval: Retrieval = "agent"
    selection: Selection = "native"
    max_tool_rounds: int = 3  # garde-fou du parcours « question » ; budget réel en phase 6


DEFAULT_CONFIG = AgentConfig()


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    step: int
    tool_rounds: int
    fen: str | None
    in_theory: bool | None
    opening: str | None  # nom Lichess de l'ouverture, si connu
    data: dict[str, Any]  # résultats structurés par outil (parcours position)


@dataclass(frozen=True)
class Decision:
    tool: str | None  # None = répondre sans outil
    arguments: dict[str, Any]
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    status: str
    outcome: str


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    decisions: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    trace_id: str = ""
    in_theory: bool | None = None
    opening: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def tools_used(self) -> list[str]:
        return [call.name for call in self.tool_calls]


class ChessAgent:
    def __init__(
        self,
        registry: ToolRegistry,
        tracing: Tracing,
        llm_factory: LLMFactory,
        config: AgentConfig = DEFAULT_CONFIG,
    ) -> None:
        self.registry = registry
        self.tracing = tracing
        self.make_llm = llm_factory
        self.config = config
        self.tools: dict[str, StructuredTool] = {
            tool.name: tool for tool in build_langchain_tools(registry)
        }
        self._graph = self._build()

    # --- API --------------------------------------------------------------------------------

    async def ask(self, question: str, *, session_id: str | None = None) -> AgentResult:
        """Question libre : le modèle décide des outils (ou flot fixe)."""
        return await self._run(question, fen=None, session_id=session_id)

    async def analyze(
        self, fen: str, *, question: str | None = None, session_id: str | None = None
    ) -> AgentResult:
        """Position : parcours déterministe Lichess → Stockfish → contexte → vidéos."""
        prompt = question or "Que me conseilles-tu dans cette position ?"
        return await self._run(f"{prompt}\nPosition (FEN) : {fen}", fen=fen, session_id=session_id)

    async def first_decision(self, question: str, *, session_id: str | None = None) -> Decision:
        """Première décision du modèle seule, sans exécuter d'outil (évaluation Tâches-30)."""
        session_id = session_id or uuid.uuid4().hex[:12]
        with turn_span(
            self.tracing.tracer,
            session_id=session_id,
            name="first_decision",
            selection=self.config.selection,
        ):
            update = await self._decide(self._initial_state(question, None, session_id))
        message = update["messages"][0]
        tokens_in, tokens_out = usage(message)
        if message.tool_calls:
            call = message.tool_calls[0]
            return Decision(call["name"], dict(call.get("args") or {}), tokens_in, tokens_out)
        return Decision(None, {}, tokens_in, tokens_out)

    async def _run(self, text: str, *, fen: str | None, session_id: str | None) -> AgentResult:
        session_id = session_id or uuid.uuid4().hex[:12]
        with turn_span(
            self.tracing.tracer,
            session_id=session_id,
            name="analyze_position" if fen else "agent_turn",
            retrieval=self.config.retrieval,
            selection=self.config.selection,
            has_fen=fen is not None,
        ) as root:
            state = await self._graph.ainvoke(self._initial_state(text, fen, session_id))
            trace_id = format(root.get_span_context().trace_id, "032x")
        return self._result(state, trace_id)

    @staticmethod
    def _initial_state(text: str, fen: str | None, session_id: str) -> AgentState:
        return {
            "messages": [HumanMessage(content=text)],
            "session_id": session_id,
            "step": 0,
            "tool_rounds": 0,
            "fen": fen,
            "in_theory": None,
            "opening": None,
            "data": {},
        }

    # --- Graphe -----------------------------------------------------------------------------

    def _build(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("answer", self._answer)
        # Parcours « position » (déterministe)
        graph.add_node("inspect", self._inspect)
        graph.add_node("theory", self._theory)
        graph.add_node("quick_eval", self._quick_eval)
        graph.add_node("engine", self._engine)
        graph.add_node("context", self._context)
        graph.add_node("videos", self._videos)
        graph.add_edge("inspect", "theory")
        graph.add_conditional_edges("theory", self._route_theory, ["quick_eval", "engine"])
        graph.add_edge("quick_eval", "context")
        graph.add_edge("context", "videos")
        graph.add_edge("videos", "answer")
        graph.add_edge("engine", "answer")
        # Parcours « question »
        if self.config.retrieval == "fixed":
            graph.add_node("retrieve", self._retrieve)
            graph.add_edge("retrieve", "answer")
            question_entry = "retrieve"
        else:
            graph.add_node("decide", self._decide)
            graph.add_node("tools", self._tools)
            graph.add_conditional_edges("decide", self._route_decision, ["tools", "answer", END])
            graph.add_edge("tools", "decide")
            question_entry = "decide"
        graph.add_conditional_edges(
            START,
            lambda state: "inspect" if state["fen"] else question_entry,
            ["inspect", question_entry],
        )
        graph.add_edge("answer", END)
        return graph.compile()

    @staticmethod
    def _route_theory(state: AgentState) -> str:
        return "quick_eval" if state["in_theory"] else "engine"

    @staticmethod
    def _route_decision(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        if isinstance(last, AIMessage) and str(last.content).strip():
            return END  # réponse donnée directement, sans outil
        return "answer"

    # --- Parcours « position » ----------------------------------------------------------------

    async def _inspect(self, state: AgentState) -> dict[str, Any]:
        return await self._call(state, "inspect_position", {"fen": state["fen"]}, key="position")

    async def _theory(self, state: AgentState) -> dict[str, Any]:
        update = await self._call(
            state, "get_opening_theory", {"fen": state["fen"], "include_games": True}, key="theory"
        )
        theory = update["data"].get("theory") or {}
        opening = theory.get("opening") or {}
        # Lichess en panne ou position inconnue → hors théorie : Stockfish prend le relais.
        update["in_theory"] = bool(theory.get("in_theory"))
        update["opening"] = opening.get("name") if update["in_theory"] else None
        return update

    async def _quick_eval(self, state: AgentState) -> dict[str, Any]:
        return await self._call(
            state,
            "evaluate_position",
            {"fen": state["fen"], "depth": QUICK_EVAL_DEPTH, "lines": 1},
            key="evaluation",
        )

    async def _engine(self, state: AgentState) -> dict[str, Any]:
        return await self._call(
            state,
            "evaluate_position",
            {"fen": state["fen"], "depth": ENGINE_DEPTH, "lines": 3},
            key="evaluation",
        )

    async def _context(self, state: AgentState) -> dict[str, Any]:
        opening = state["opening"]
        if not opening:
            return {}
        return await self._call(
            state,
            SEARCH_TOOL,
            {"query": f"idées et plans de l'ouverture {opening}", "opening": opening, "limit": 3},
            key="context",
        )

    async def _videos(self, state: AgentState) -> dict[str, Any]:
        opening = state["opening"]
        if not opening:
            return {}
        return await self._call(
            state,
            "find_opening_videos",
            {"query": f"{opening} échecs ouverture explication", "max_results": 2},
            key="videos",
        )

    async def _call(
        self, state: AgentState, name: str, arguments: dict[str, Any], *, key: str
    ) -> dict[str, Any]:
        """Appelle un outil du parcours position ; le résultat structuré va dans `data[key]`."""
        step = state["step"] + 1
        call = {"name": name, "args": arguments, "id": f"{key}_{step}"}
        request = AIMessage(content="", tool_calls=[call])
        message = await self._run_tool(call, step, state["session_id"])
        artifact = getattr(message, "artifact", None) or {}
        data = dict(state["data"])
        if message.status == "error":
            data[key] = {"error": str(message.content)}
        else:
            data[key] = artifact.get("data")
        return {"step": step, "messages": [request, message], "data": data}

    # --- Parcours « question » ----------------------------------------------------------------

    async def _decide(self, state: AgentState) -> dict[str, Any]:
        step = state["step"] + 1
        if state["tool_rounds"] >= self.config.max_tool_rounds:
            # Garde-fou : plus d'outil, le prochain nœud sera la réponse.
            return {"step": step, "messages": [AIMessage(content="")]}
        with step_span(
            self.tracing.tracer, kind="decision", step=step, session_id=state["session_id"]
        ) as span:
            if self.config.selection == "structured":
                message = await self._decide_structured(state["messages"])
            else:
                message = await self._decide_native(state["messages"])
            tokens_in, tokens_out = usage(message)
            if message.tool_calls:
                call = message.tool_calls[0]
                record_outcome(
                    span,
                    choice=call["name"],
                    arguments=call["args"],
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    outcome=f"{len(message.tool_calls)} appel(s) d'outil",
                )
            else:
                reason = message.additional_kwargs.get("reason") or "réponse sans outil"
                record_outcome(
                    span,
                    choice="answer",
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    outcome=str(reason),
                )
        return {"step": step, "messages": [message]}

    async def _decide_native(self, messages: list[BaseMessage]) -> AIMessage:
        llm = self.make_llm().bind_tools(list(self.tools.values()))
        response = await llm.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), *messages],
            config={"callbacks": self.tracing.langchain_callbacks(), "run_name": "decide"},
        )
        return response  # type: ignore[return-value]

    async def _decide_structured(self, messages: list[BaseMessage]) -> AIMessage:
        names = list(self.tools)
        schema = {
            "type": "object",
            "properties": {
                "tool": {"type": ["string", "null"], "enum": [*names, None]},
                "arguments": {"type": "object"},
                "reason": {"type": "string"},
            },
            "required": ["tool", "arguments", "reason"],
        }
        catalog = "\n".join(
            f"- {name} : {tool.description.splitlines()[0]} Paramètres : "
            f"{json.dumps(tool.args, ensure_ascii=False)}"
            for name, tool in self.tools.items()
        )
        llm = self.make_llm(structured_schema=schema)
        response = await llm.ainvoke(
            [
                SystemMessage(
                    content=STRUCTURED_PROMPT.format(system=SYSTEM_PROMPT, catalog=catalog)
                ),
                *messages,
            ],
            config={"callbacks": self.tracing.langchain_callbacks(), "run_name": "decide"},
        )
        try:
            decision = json.loads(str(response.content))
        except ValueError:
            decision = {"tool": None, "reason": "réponse non analysable"}
        tool = decision.get("tool")
        reason = str(decision.get("reason", ""))
        # La justification n'est pas une réponse : contenu vide → le graphe exécute l'outil
        # ou passe au nœud de réponse. Elle est gardée pour le journal et Langfuse.
        tool_calls = []
        if tool in self.tools:
            arguments = decision.get("arguments") or {}
            tool_calls = [{"name": tool, "args": arguments, "id": f"call_{uuid.uuid4().hex[:8]}"}]
        message = AIMessage(content="", tool_calls=tool_calls, additional_kwargs={"reason": reason})
        message.usage_metadata = getattr(response, "usage_metadata", None)
        return message

    async def _tools(self, state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        step = state["step"]
        results: list[ToolMessage] = []
        for call in last.tool_calls:
            step += 1
            results.append(await self._run_tool(call, step, state["session_id"]))
        return {"step": step, "tool_rounds": state["tool_rounds"] + 1, "messages": results}

    async def _retrieve(self, state: AgentState) -> dict[str, Any]:
        """Flot fixe : une recherche à chaque question, sans décision du modèle."""
        question = str(state["messages"][-1].content)
        call = {"name": SEARCH_TOOL, "args": {"query": question[:300]}, "id": "fixed_retrieval"}
        request = AIMessage(content="", tool_calls=[call])
        result = await self._run_tool(call, state["step"] + 1, state["session_id"])
        return {"step": state["step"] + 1, "tool_rounds": 1, "messages": [request, result]}

    # --- Commun --------------------------------------------------------------------------------

    async def _run_tool(self, call: dict[str, Any], step: int, session_id: str) -> ToolMessage:
        name, arguments = call["name"], dict(call.get("args") or {})
        with step_span(
            self.tracing.tracer,
            kind="tool_call",
            step=step,
            session_id=session_id,
            choice=name,
            arguments=arguments,
        ) as span:
            tool = self.tools.get(name)
            if tool is None:
                outcome = f"[unknown_tool] Outil inconnu : {name}."
                record_outcome(span, status="error", outcome=outcome)
                return ToolMessage(content=outcome, tool_call_id=call["id"], status="error")
            message = await tool.ainvoke(
                {"name": name, "args": arguments, "id": call["id"], "type": "tool_call"},
                config={"callbacks": self.tracing.langchain_callbacks()},
            )
            status = "error" if message.status == "error" else "success"
            record_outcome(span, status=status, outcome=_summarize(message))
            return message

    async def _answer(self, state: AgentState) -> dict[str, Any]:
        step = state["step"] + 1
        system = POSITION_PROMPT if state["fen"] else SYSTEM_PROMPT
        with step_span(
            self.tracing.tracer, kind="answer", step=step, session_id=state["session_id"]
        ) as span:
            llm = self.make_llm()
            response = await llm.ainvoke(
                [SystemMessage(content=system), *_without_empty(state["messages"])],
                config={"callbacks": self.tracing.langchain_callbacks(), "run_name": "answer"},
            )
            tokens_in, tokens_out = usage(response)
            record_outcome(
                span,
                choice="answer",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                outcome=str(response.content)[:120],
            )
        final = AIMessage(content=str(response.content))
        final.usage_metadata = getattr(response, "usage_metadata", None)  # compté dans le résultat
        return {"step": step, "messages": [final]}

    # --- Résultat ---------------------------------------------------------------------------

    def _result(self, state: AgentState, trace_id: str) -> AgentResult:
        result = AgentResult(
            answer="",
            trace_id=trace_id,
            in_theory=state["in_theory"],
            opening=state["opening"],
            data=state["data"],
        )
        pending: dict[str, dict[str, Any]] = {}
        for message in state["messages"]:
            if isinstance(message, AIMessage):
                tokens_in, tokens_out = usage(message)
                result.tokens_in += tokens_in
                result.tokens_out += tokens_out
                if message.tool_calls:
                    for call in message.tool_calls:
                        pending[call["id"]] = call
                    if tokens_in or tokens_out:  # appel issu d'une décision du modèle
                        result.decisions += 1
                elif str(message.content).strip():
                    result.answer = str(message.content)
                    result.decisions += 1
            elif isinstance(message, ToolMessage) and message.tool_call_id in pending:
                call = pending.pop(message.tool_call_id)
                result.tool_calls.append(
                    ToolCallRecord(
                        name=call["name"],
                        arguments=dict(call.get("args") or {}),
                        status="error" if message.status == "error" else "success",
                        outcome=_summarize(message),
                    )
                )
        return result


def _summarize(message: ToolMessage) -> str:
    if message.status == "error":
        return str(message.content)[:200]
    artifact = getattr(message, "artifact", None) or {}
    data = artifact.get("data") if isinstance(artifact, dict) else None
    if isinstance(data, dict):
        if "passages" in data:
            return f"{len(data['passages'])} passage(s), found={data.get('found')}"
        if "in_theory" in data:
            name = (data.get("opening") or {}).get("name", "ouverture inconnue")
            return f"in_theory={data['in_theory']}, {name}, {len(data.get('moves', []))} coups"
        if "assessment" in data:
            return str(data["assessment"])
        if "videos" in data:
            return f"{len(data['videos'])} vidéo(s)"
    return str(message.content)[:120]


def _without_empty(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Retire les messages d'assistant vides (garde-fou) avant de demander la réponse."""
    return [
        m
        for m in messages
        if not (isinstance(m, AIMessage) and not m.tool_calls and not str(m.content).strip())
    ]
