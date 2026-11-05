"""Recherche itérative multi-sauts sous budget.

    question ──► première recherche : la question telle quelle (ou une sous-question
                 rédigée par le modèle avec `first_hop="plan"`)
                   │
                   ▼
          ┌─► search (search_opening_knowledge) ──► passages retenus (dédoublonnés,
          │                                          ajoutés par score tant que le
          │                                          budget de tokens le permet)
          │        │
          │        ▼
          │     judge : assez d'informations ?
          │        ├─ oui ──────────────────────────────────► answer
          │        ├─ non, rien trouvé → reformuler ──┐
          │        └─ non, sous-question suivante ────┤ (dépend de ce qui a été trouvé)
          └──────────────────────────────────────────┘
          arrêt forcé : 3 recherches, ou budget de contexte (6 000 tokens) atteint

Chaque sous-question est un span `decision` (choix du modèle) suivi d'un span `tool_call`
(la recherche) ; le journal de décision et Langfuse en découlent.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import LLMFactory, usage
from app.agent.tracing import Tracing, record_outcome, step_span, turn_span
from app.tools.base import ToolRegistry

SEARCH_TOOL = "search_opening_knowledge"
TokenCounter = Callable[[str], int]
StopReason = Literal["enough", "max_searches", "budget", "no_next_question"]


@dataclass(frozen=True)
class Budget:
    max_searches: int = 3
    max_context_tokens: int = 6000
    passages_per_search: int = 4
    # Première recherche : la question telle quelle, ou une sous-question rédigée par le
    # modèle ("plan", qui perd souvent des détails : coups, noms).
    first_hop: Literal["question", "plan"] = "question"


@dataclass
class HopRecord:
    sub_question: str
    reformulated: bool
    found: bool
    new_passages: int
    context_tokens: int  # cumul après ce saut
    error: str | None = None


@dataclass
class MultiHopResult:
    answer: str
    hops: list[HopRecord]
    passages: list[dict[str, Any]]
    context_tokens: int
    stop_reason: StopReason
    tokens_in: int = 0
    tokens_out: int = 0
    trace_id: str = ""

    @property
    def searches(self) -> int:
        return len(self.hops)


DEFAULT_BUDGET = Budget()

PLAN_SCHEMA = {
    "type": "object",
    "properties": {"sub_question": {"type": "string"}, "reason": {"type": "string"}},
    "required": ["sub_question", "reason"],
}
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "enough": {"type": "boolean"},
        "next_sub_question": {"type": ["string", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["enough", "next_sub_question", "reason"],
}

PLAN_PROMPT = """\
Tu prépares une recherche dans une base de connaissances sur les ouvertures d'échecs
(articles encyclopédiques). La question peut demander plusieurs informations enchaînées :
on cherche une information à la fois.

Donne la PREMIÈRE sous-question à poser à la base : courte, autonome, portant sur une
seule information, formulée avec les mots qu'un article emploierait (nom d'ouverture,
coups, joueur). Si la question est simple, la sous-question peut la reprendre.
Réponds uniquement en JSON : {"sub_question": "...", "reason": "..."}"""

JUDGE_PROMPT = """\
Tu conduis une recherche pas à pas dans une base de connaissances sur les ouvertures
d'échecs. Question initiale, sous-questions déjà posées et passages retrouvés sont fournis.

Décide :
- "enough" : true si les passages contiennent de quoi répondre complètement à la
  question initiale ;
- "next_sub_question" : sinon, la prochaine sous-question, qui utilise ce qui vient d'être
  appris (par exemple le nom trouvé au saut précédent) ; si la dernière recherche n'a
  rien donné de pertinent, REFORMULE la même demande avec d'autres mots ; null si plus
  rien ne peut être cherché ;
- "reason" : une phrase.
Réponds uniquement en JSON."""

REFORMULATE_PROMPT = """\
Tu conduis une recherche dans une base de connaissances sur les ouvertures d'échecs. La
dernière recherche n'a pas donné de passage utile. Reformule la demande avec D'AUTRES mots
(synonymes, nom de l'ouverture, coups en notation algébrique, nom de joueur) : une seule
sous-question courte, différente de toutes celles déjà posées.
Réponds uniquement en JSON : {"sub_question": "...", "reason": "..."}"""

ANSWER_PROMPT = """\
Tu es un coach d'échecs. Réponds à la question en français, de façon courte et précise,
UNIQUEMENT à partir des passages fournis ; cite l'ouverture ou l'article concerné. Si les
passages ne suffisent pas, dis ce qui manque."""


class MultiHopSearcher:
    def __init__(
        self,
        registry: ToolRegistry,
        tracing: Tracing,
        llm_factory: LLMFactory,
        count_tokens: TokenCounter,
        budget: Budget = DEFAULT_BUDGET,
    ) -> None:
        self.registry = registry
        self.tracing = tracing
        self.make_llm = llm_factory
        self.count_tokens = count_tokens
        self.budget = budget

    async def search(self, question: str, *, session_id: str | None = None) -> MultiHopResult:
        session_id = session_id or uuid.uuid4().hex[:12]
        with turn_span(
            self.tracing.tracer,
            session_id=session_id,
            name="multihop_search",
            max_searches=self.budget.max_searches,
            max_context_tokens=self.budget.max_context_tokens,
        ) as root:
            result = await self._loop(question, session_id)
            result.trace_id = format(root.get_span_context().trace_id, "032x")
        return result

    # --- Boucle ---------------------------------------------------------------------------------

    async def _loop(self, question: str, session_id: str) -> MultiHopResult:
        passages: dict[str, dict[str, Any]] = {}
        hops: list[HopRecord] = []
        tokens = [0, 0]
        context_tokens = 0
        step = 0

        step += 1
        if self.budget.first_hop == "plan":
            sub_question, reformulated = await self._plan(question, step, session_id, tokens)
        else:
            sub_question, reformulated = self._take_question(question, step, session_id)
        stop: StopReason | None = None

        while True:
            step += 1
            hits, error = await self._search(sub_question, step, session_id)
            added, context_tokens, exhausted = self._retain(hits, passages, context_tokens)
            hops.append(
                HopRecord(
                    sub_question=sub_question,
                    reformulated=reformulated,
                    found=error is None and bool(hits),
                    new_passages=added,
                    context_tokens=context_tokens,
                    error=error,
                )
            )

            if len(hops) >= self.budget.max_searches:
                stop = "max_searches"
            elif exhausted:
                stop = "budget"
            if stop is None:
                step += 1
                enough, next_question, reformulated = await self._judge(
                    question, hops, passages, step, session_id, tokens
                )
                if enough:
                    stop = "enough"
                elif not next_question:
                    stop = "no_next_question"
                else:
                    sub_question = next_question
                    continue
            break

        step += 1
        answer = await self._answer(question, passages, step, session_id, tokens)
        return MultiHopResult(
            answer=answer,
            hops=hops,
            passages=list(passages.values()),
            context_tokens=context_tokens,
            stop_reason=stop or "enough",
            tokens_in=tokens[0],
            tokens_out=tokens[1],
        )

    # --- Étapes ---------------------------------------------------------------------------------

    async def _plan(
        self, question: str, step: int, session_id: str, tokens: list[int]
    ) -> tuple[str, bool]:
        with step_span(
            self.tracing.tracer, kind="decision", step=step, session_id=session_id
        ) as span:
            raw = await self._structured(PLAN_PROMPT, f"Question : {question}", PLAN_SCHEMA, tokens)
            sub_question = str(raw.get("sub_question") or question).strip() or question
            record_outcome(
                span,
                choice="plan",
                arguments={"sub_question": sub_question},
                outcome=str(raw.get("reason", ""))[:200],
                tokens_in=raw["_tokens"][0],
                tokens_out=raw["_tokens"][1],
            )
        return sub_question, False

    def _take_question(self, question: str, step: int, session_id: str) -> tuple[str, bool]:
        """Première recherche avec la question telle quelle (aucun appel de modèle)."""
        with step_span(
            self.tracing.tracer, kind="decision", step=step, session_id=session_id
        ) as span:
            record_outcome(
                span,
                choice="plan",
                arguments={"sub_question": question},
                outcome="question posée telle quelle à la base",
            )
        return question, False

    async def _search(
        self, sub_question: str, step: int, session_id: str
    ) -> tuple[list[dict[str, Any]], str | None]:
        arguments = {"query": sub_question[:300], "limit": self.budget.passages_per_search}
        with step_span(
            self.tracing.tracer,
            kind="tool_call",
            step=step,
            session_id=session_id,
            choice=SEARCH_TOOL,
            arguments=arguments,
        ) as span:
            result = await self.registry.execute(SEARCH_TOOL, arguments)
            if not result.ok:
                outcome = result.to_model_content()
                record_outcome(span, status="error", outcome=outcome)
                return [], outcome
            hits = list((result.data or {}).get("passages", []))
            found = (result.data or {}).get("found")
            record_outcome(span, outcome=f"{len(hits)} passage(s), found={found}")
            return hits, None

    def _retain(
        self, hits: list[dict[str, Any]], passages: dict[str, dict[str, Any]], context_tokens: int
    ) -> tuple[int, int, bool]:
        """Ajoute les nouveaux passages par score décroissant tant que le budget le permet.

        Renvoie (passages ajoutés, cumul de tokens, budget épuisé ?).
        """
        added = 0
        exhausted = False
        for hit in sorted(hits, key=lambda h: h.get("score", 0), reverse=True):
            key = hit.get("chunk_id") or hit.get("text", "")[:64]
            if key in passages:
                continue
            cost = self.count_tokens(hit.get("text", ""))
            if context_tokens + cost > self.budget.max_context_tokens:
                exhausted = True
                continue
            passages[key] = hit
            context_tokens += cost
            added += 1
        return added, context_tokens, exhausted

    async def _judge(
        self,
        question: str,
        hops: list[HopRecord],
        passages: dict[str, dict[str, Any]],
        step: int,
        session_id: str,
        tokens: list[int],
    ) -> tuple[bool, str | None, bool]:
        last = hops[-1]
        asked = "\n".join(f"- {h.sub_question}" for h in hops)
        note = (
            "La dernière recherche n'a rien trouvé de pertinent : reformule."
            if not last.found or last.new_passages == 0
            else ""
        )
        user = (
            f"Question initiale : {question}\n\nSous-questions déjà posées :\n{asked}\n"
            f"{note}\n\nPassages retrouvés :\n{_render(passages)}"
        )
        with step_span(
            self.tracing.tracer, kind="decision", step=step, session_id=session_id
        ) as span:
            raw = await self._structured(JUDGE_PROMPT, user, JUDGE_SCHEMA, tokens)
            enough = bool(raw.get("enough"))
            next_question = raw.get("next_sub_question")
            next_question = str(next_question).strip() if next_question else None
            if next_question and any(next_question == h.sub_question for h in hops):
                next_question = None  # une sous-question identique ne ferait que répéter
            reformulated = bool(note) and next_question is not None
            reason = str(raw.get("reason", ""))
            tokens_in, tokens_out = raw["_tokens"]
            if not enough and next_question is None:
                # Le modèle renonce (`null`) alors qu'il reste du budget : la reformulation
                # est imposée, avec un appel dédié dont le schéma exige une sous-question.
                next_question, extra = await self._reformulate(question, hops, reason, tokens)
                reformulated = next_question is not None
                reason = extra or reason
            record_outcome(
                span,
                choice="stop" if enough else ("reformulate" if reformulated else "next_hop"),
                arguments={"next_sub_question": next_question},
                outcome=reason[:200],
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
        return enough, next_question, reformulated

    async def _reformulate(
        self, question: str, hops: list[HopRecord], missing: str, tokens: list[int]
    ) -> tuple[str | None, str]:
        """Demande une autre formulation ; None si le modèle répète une sous-question."""
        asked = "\n".join(f"- {h.sub_question}" for h in hops)
        user = (
            f"Question initiale : {question}\n\nSous-questions déjà posées :\n{asked}\n\n"
            f"Ce qui manque encore : {missing or 'la réponse'}"
        )
        raw = await self._structured(REFORMULATE_PROMPT, user, PLAN_SCHEMA, tokens)
        candidate = str(raw.get("sub_question") or "").strip()
        if not candidate or any(candidate == h.sub_question for h in hops):
            return None, str(raw.get("reason", ""))
        return candidate, str(raw.get("reason", ""))

    async def _answer(
        self,
        question: str,
        passages: dict[str, dict[str, Any]],
        step: int,
        session_id: str,
        tokens: list[int],
    ) -> str:
        with step_span(
            self.tracing.tracer, kind="answer", step=step, session_id=session_id
        ) as span:
            llm = self.make_llm()
            response = await llm.ainvoke(
                [
                    SystemMessage(content=ANSWER_PROMPT),
                    HumanMessage(
                        content=f"Question : {question}\n\nPassages :\n{_render(passages)}"
                    ),
                ],
                config={"callbacks": self.tracing.langchain_callbacks(), "run_name": "answer"},
            )
            tokens_in, tokens_out = usage(response)
            tokens[0] += tokens_in
            tokens[1] += tokens_out
            record_outcome(
                span,
                choice="answer",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                outcome=str(response.content)[:120],
            )
        return str(response.content)

    async def _structured(
        self, system: str, user: str, schema: dict[str, Any], tokens: list[int]
    ) -> dict[str, Any]:
        llm = self.make_llm(structured_schema=schema)
        response = await llm.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=user)],
            config={"callbacks": self.tracing.langchain_callbacks()},
        )
        tokens_in, tokens_out = usage(response)
        tokens[0] += tokens_in
        tokens[1] += tokens_out
        try:
            data = json.loads(str(response.content))
        except ValueError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data["_tokens"] = (tokens_in, tokens_out)
        return data


def _render(passages: dict[str, dict[str, Any]], max_chars: int = 1500) -> str:
    if not passages:
        return "(aucun passage)"
    return "\n\n".join(
        f"[{i}] {p.get('section') or p.get('title', '')}\n{str(p.get('text', ''))[:max_chars]}"
        for i, p in enumerate(passages.values(), start=1)
    )
