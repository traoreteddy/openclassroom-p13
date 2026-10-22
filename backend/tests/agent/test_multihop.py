"""Recherche multi-sauts sous budget : décomposition, arrêt, reformulation, budget."""

from __future__ import annotations

import json

import pytest

from app.agent.multihop import Budget, MultiHopSearcher
from app.agent.tracing import configure_tracing
from tests.agent.test_graph import ScriptedChat, reply
from tests.conftest import StubAnalyzer, StubVectorStore, passage


def plan(sub_question: str):
    return reply(json.dumps({"sub_question": sub_question, "reason": "première étape"}))


def judge(enough: bool, next_question: str | None = None, reason: str = "jugement"):
    return reply(
        json.dumps({"enough": enough, "next_sub_question": next_question, "reason": reason})
    )


def words(text: str) -> int:
    return len(text.split())


class RoutingStore(StubVectorStore):
    """Renvoie des passages différents selon la question (le vecteur ne sert pas ici)."""

    def __init__(self, routes: dict[str, list]) -> None:
        super().__init__()
        self.routes = routes
        self.queries: list[str] = []

    async def search(self, vector, *, limit, opening):
        self.calls.append({"limit": limit, "opening": opening})
        return list(self.passages)[:limit]


class RoutingEmbedder:
    def __init__(self, store: RoutingStore) -> None:
        self.store = store

    async def embed(self, text: str):
        self.store.queries.append(text)
        # le stub de recherche lit la dernière question posée
        self.store.passages = next(
            (hits for key, hits in self.store.routes.items() if key in text.lower()), []
        )
        return [0.1, 0.2, 0.3]


@pytest.fixture
def tracing():
    return configure_tracing()


def make_searcher(make_registry, tracing, script, routes, **budget):
    budget.setdefault("first_hop", "plan")  # les scénarios scriptés commencent par un plan
    store = RoutingStore(routes)
    registry = make_registry(StubAnalyzer(), store)
    registry.get("search_opening_knowledge").handler.__self__._embedder = RoutingEmbedder(store)
    chat = ScriptedChat(script=list(script), calls=[])
    searcher = MultiHopSearcher(
        registry, tracing, lambda structured_schema=None: chat, words, Budget(**budget)
    )
    return searcher, store, chat


EVANS = passage("Le gambit Evans commence par 4.b4.", 0.9, opening="Italian Game")
CAPTAIN = passage(
    "Le gambit porte le nom du capitaine de navire gallois William Davies Evans.",
    0.85,
    opening="Italian Game",
)


async def test_two_dependent_hops_then_stop(make_registry, tracing):
    routes = {"4.b4": [EVANS], "evans": [CAPTAIN]}
    searcher, store, chat = make_searcher(
        make_registry,
        tracing,
        [
            plan("Quel gambit de la partie italienne commence par 4.b4 ?"),
            judge(False, "Qui a donné son nom au gambit Evans ?"),
            judge(True),
            reply("Un capitaine de navire gallois, William Davies Evans."),
        ],
        routes,
    )

    result = await searcher.search(
        "Quel métier exerçait l'homme qui a donné son nom au gambit joué par 4.b4 ?",
        session_id="mh-1",
    )

    assert [h.sub_question for h in result.hops] == [
        "Quel gambit de la partie italienne commence par 4.b4 ?",
        "Qui a donné son nom au gambit Evans ?",
    ]
    assert result.searches == 2
    assert result.stop_reason == "enough"
    assert [p["text"] for p in result.passages] == [EVANS.text, CAPTAIN.text]
    assert result.context_tokens == words(EVANS.text) + words(CAPTAIN.text)
    assert result.answer.startswith("Un capitaine")
    assert store.queries == [h.sub_question for h in result.hops]
    assert "Passages retrouvés" in chat.llm_calls[1]["messages"][1]
    assert result.tokens_in == 400 and len(result.trace_id) == 32


async def test_max_searches_stops_the_loop(make_registry, tracing):
    searcher, _, chat = make_searcher(
        make_registry,
        tracing,
        [plan("q1"), judge(False, "q2"), judge(False, "q3"), reply("Réponse.")],
        {"q": [EVANS]},
        max_searches=3,
    )

    result = await searcher.search("question", session_id="mh-2")

    assert result.searches == 3
    assert result.stop_reason == "max_searches"
    assert len(chat.llm_calls) == 4  # plan, 2 jugements, réponse : pas de 3e jugement


async def test_reformulation_when_nothing_is_found(make_registry, tracing):
    routes = {"idées": [], "plans": [CAPTAIN]}
    searcher, _, chat = make_searcher(
        make_registry,
        tracing,
        [
            plan("idées du gambit Evans"),
            judge(False, "plans du gambit Evans"),
            judge(True),
            reply("Réponse."),
        ],
        routes,
    )

    result = await searcher.search("question", session_id="mh-3")

    assert result.hops[0].found is False and result.hops[0].new_passages == 0
    assert result.hops[1].reformulated is True
    assert result.hops[1].new_passages == 1
    assert "reformule" in chat.llm_calls[1]["messages"][1]
    journal = tracing.journal.for_session("mh-3")
    assert [e.choice for e in journal if e.kind == "decision"] == ["plan", "reformulate", "stop"]


async def test_context_budget_caps_retained_passages(make_registry, tracing):
    big = [passage(f"passage numéro {i} " * 10, 0.9 - i / 100) for i in range(4)]
    searcher, _, _ = make_searcher(
        make_registry,
        tracing,
        [plan("quelle idée"), reply("Réponse.")],
        {"idée": big},
        max_context_tokens=65,  # 2 passages de 30 mots tiennent, pas 3
    )

    result = await searcher.search("question", session_id="mh-4")

    assert result.stop_reason == "budget"
    assert len(result.passages) == 2
    assert result.context_tokens <= 65
    assert result.passages[0]["score"] >= result.passages[1]["score"]


async def test_duplicate_passages_are_not_counted_twice(make_registry, tracing):
    searcher, _, _ = make_searcher(
        make_registry,
        tracing,
        [plan("quelle idée"), judge(False, "quelle idée, bis"), judge(True), reply("Réponse.")],
        {"idée": [EVANS]},
    )

    result = await searcher.search("question", session_id="mh-5")

    assert result.hops[1].new_passages == 0
    assert len(result.passages) == 1


async def test_identical_next_question_stops(make_registry, tracing):
    searcher, _, _ = make_searcher(
        make_registry, tracing, [plan("q"), judge(False, "q"), reply("Réponse.")], {"q": [EVANS]}
    )

    result = await searcher.search("question", session_id="mh-6")

    assert result.stop_reason == "no_next_question"
    assert result.searches == 1


async def test_search_error_is_recorded_and_loop_continues(make_registry, tracing):
    from app.tools.errors import ToolUnavailableError

    searcher, store, _ = make_searcher(
        make_registry,
        tracing,
        [plan("q"), judge(False, "autre"), judge(True), reply("R.")],
        {"q": [EVANS], "autre": [CAPTAIN]},
    )
    original = store.search

    async def failing(vector, *, limit, opening):
        if store.queries[-1] == "q":
            raise ToolUnavailableError(code="not_indexed", message="pas d'index")
        return await original(vector, limit=limit, opening=opening)

    store.search = failing

    result = await searcher.search("question", session_id="mh-7")

    assert result.hops[0].error is not None and result.hops[0].found is False
    assert result.hops[1].new_passages == 1


async def test_bad_json_from_the_model_is_tolerated(make_registry, tracing):
    searcher, _, _ = make_searcher(
        make_registry,
        tracing,
        # plan, jugement et reformulation imposée illisibles, puis la réponse
        [reply("pas du json")] * 3 + [reply("Réponse.")],
        {"question": [EVANS]},
    )

    result = await searcher.search("question", session_id="mh-8")

    assert result.hops[0].sub_question == "question"  # repli : la question elle-même
    assert result.stop_reason == "no_next_question"
    assert result.answer == "Réponse."


async def test_reformulation_is_forced_when_the_judge_gives_up(make_registry, tracing):
    """Le juge renvoie `null` après une recherche vide : une reformulation est imposée."""
    routes = {"idées": [], "plans": [CAPTAIN]}
    searcher, _, chat = make_searcher(
        make_registry,
        tracing,
        [
            plan("idées du gambit Evans"),
            judge(False, None, "rien trouvé"),
            reply(json.dumps({"sub_question": "plans du gambit Evans", "reason": "synonyme"})),
            judge(True),
            reply("Réponse."),
        ],
        routes,
    )

    result = await searcher.search("question", session_id="mh-9")

    assert [h.sub_question for h in result.hops] == [
        "idées du gambit Evans",
        "plans du gambit Evans",
    ]
    assert result.hops[1].reformulated is True and result.hops[1].new_passages == 1
    assert "D'AUTRES mots" in chat.llm_calls[2]["messages"][0]
    assert "rien trouvé" in chat.llm_calls[2]["messages"][1]
    journal = tracing.journal.for_session("mh-9")
    assert [e.choice for e in journal if e.kind == "decision"] == ["plan", "reformulate", "stop"]


async def test_forced_reformulation_that_repeats_a_question_stops(make_registry, tracing):
    searcher, _, _ = make_searcher(
        make_registry,
        tracing,
        [
            plan("q"),
            judge(False, None),
            reply(json.dumps({"sub_question": "q", "reason": "pareil"})),
            reply("Réponse."),
        ],
        {"q": [EVANS]},
    )

    result = await searcher.search("question", session_id="mh-10")

    assert result.stop_reason == "no_next_question"
    assert result.searches == 1


async def test_first_hop_uses_the_question_verbatim_by_default(make_registry, tracing):
    """Sans plan, la première recherche reprend la question : aucun appel de modèle."""
    searcher, store, chat = make_searcher(
        make_registry,
        tracing,
        [judge(True), reply("Réponse.")],
        {"4.b4": [EVANS]},
        first_hop="question",
    )

    result = await searcher.search("Quel gambit commence par 4.b4 ?", session_id="mh-11")

    assert store.queries == ["Quel gambit commence par 4.b4 ?"]
    assert result.hops[0].sub_question == "Quel gambit commence par 4.b4 ?"
    assert len(chat.llm_calls) == 2  # jugement + réponse
    journal = tracing.journal.for_session("mh-11")
    assert [e.choice for e in journal if e.kind == "decision"] == ["plan", "stop"]
    assert Budget().first_hop == "question"
