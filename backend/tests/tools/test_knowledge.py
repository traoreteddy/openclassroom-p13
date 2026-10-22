"""Outil search_opening_knowledge : cas limites paramétrés, embedder et Milvus en stubs."""

from __future__ import annotations

import pytest

from app.tools.catalog import default_dependencies
from app.tools.errors import ToolTransientError
from tests.conftest import START_FEN, StubVectorStore, passage

QUESTION = {"query": "idées principales de la défense sicilienne"}


@pytest.fixture
def store() -> StubVectorStore:
    return StubVectorStore(
        [
            passage("Les Noirs contestent le centre avec c5.", 0.82),
            passage("Le pion c5 échange contre d4.", 0.71),
            passage("Texte hors sujet.", 0.31),
            passage("Le Fou en c4 vise f7.", 0.77, opening="Italian Game"),
        ]
    )


async def test_returns_relevant_passages_sorted(make_registry, store):
    data = (await make_registry(store=store).execute("search_opening_knowledge", QUESTION)).data

    assert data["found"] is True
    assert [p["score"] for p in data["passages"]] == [0.82, 0.77, 0.71]  # 0.31 < seuil 0.5


async def test_opening_filter_is_passed_to_store(make_registry, store):
    data = (
        await make_registry(store=store).execute(
            "search_opening_knowledge", {**QUESTION, "opening": "Sicilian Defense", "limit": 2}
        )
    ).data

    assert store.calls == [{"limit": 2, "opening": "Sicilian Defense"}]
    assert {p["opening"] for p in data["passages"]} == {"Sicilian Defense"}
    assert len(data["passages"]) == 2


async def test_nothing_relevant(make_registry):
    store = StubVectorStore([passage("Bruit", 0.2)])

    data = (await make_registry(store=store).execute("search_opening_knowledge", QUESTION)).data

    assert data == {"found": False, "passages": []}


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        ({"query": "ab"}, "at least 3 characters"),
        ({"query": "x" * 301}, "at most 300 characters"),
        ({"query": START_FEN}, "query contient une FEN"),
        ({"query": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"}, "query contient une FEN"),
        ({**QUESTION, "limit": 0}, "limit"),
        ({**QUESTION, "limit": 9}, "limit"),
        ({**QUESTION, "opening": "x" * 101}, "opening"),
    ],
    ids=[
        "requete-trop-courte",
        "requete-trop-longue",
        "fen-complete",
        "placement-seul",
        "limit-0",
        "limit-9",
        "ouverture-trop-longue",
    ],
)
async def test_edge_cases_are_rejected(make_registry, arguments, fragment):
    result = await make_registry().execute("search_opening_knowledge", arguments)

    assert result.error["code"] == "invalid_arguments"
    assert fragment in result.error["message"]


@pytest.mark.parametrize(
    "query",
    ["Pourquoi jouer 1.e4 c5 ?", "le coup Cf3/Cc3 dans la Sicilienne", "8 idées sur la Najdorf"],
)
async def test_chess_notation_in_words_is_accepted(make_registry, query):
    result = await make_registry().execute("search_opening_knowledge", {"query": query})

    assert result.ok


async def test_store_outage_is_retried_then_reported(make_registry):
    store = StubVectorStore(error=ToolTransientError(code="network_error", message="Milvus"))

    result = await make_registry(store=store).execute("search_opening_knowledge", QUESTION)

    assert result.error["code"] == "retries_exhausted"
    assert len(store.calls) == 2


def test_default_dependencies_use_milvus_and_the_embedding_model(settings):
    from app.rag.embedding import SentenceTransformerEmbedder
    from app.rag.milvus_store import MilvusVectorStore

    knowledge = default_dependencies(settings).knowledge

    assert isinstance(knowledge._embedder, SentenceTransformerEmbedder)
    assert isinstance(knowledge._store, MilvusVectorStore)
    assert knowledge._store.collection == settings.milvus_collection
    assert knowledge._min_score == settings.rag_min_score
