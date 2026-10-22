"""Embedder (faux modèle) et adaptateur Milvus (faux client) : aucun modèle ni serveur."""

from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

from app.rag.chunking import chunk_article
from app.rag.embedding import SentenceTransformerEmbedder
from app.rag.milvus_store import MilvusVectorStore, opening_filter, row
from app.tools.errors import ToolTransientError, ToolUnavailableError
from tests.rag.test_chunking import CONFIG, MARKDOWN, words
from tests.rag.test_text_and_passages import article


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str | None]] = []

    def get_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, *, prompt_name, normalize_embeddings, **kwargs):
        assert normalize_embeddings
        self.calls.append((list(texts), prompt_name))
        return np.array([[0.6, 0.8, 0.0] for _ in texts])


@pytest.fixture
def embedder():
    embedder = SentenceTransformerEmbedder("modele-test")
    embedder._model = FakeModel()
    return embedder


async def test_queries_use_the_query_prompt(embedder):
    vector = await embedder.embed("Idées de la Sicilienne ?")

    assert vector == [0.6, 0.8, 0.0]
    assert embedder._model.calls == [(["Idées de la Sicilienne ?"], "query")]


async def test_documents_have_no_prompt(embedder):
    vectors = await embedder.embed_documents(["a", "b"])

    assert len(vectors) == 2
    assert embedder._model.calls == [(["a", "b"], None)]
    assert embedder.dimension == 3


async def test_query_during_loading_fails_fast_then_works(monkeypatch):
    embedder = SentenceTransformerEmbedder("modele-test")
    release = threading.Event()

    def slow_load():
        release.wait(timeout=5)
        embedder._model = FakeModel()
        return embedder._model

    monkeypatch.setattr(embedder, "_load", slow_load)

    with pytest.raises(ToolUnavailableError) as info:
        await embedder.embed("question ?")
    assert info.value.code == "model_loading"
    assert info.value.details["retry_after_s"] == 10

    release.set()
    await embedder._warm_up_task
    assert embedder.ready
    assert await embedder.embed("question ?") == [0.6, 0.8, 0.0]


async def test_failed_loading_is_reported(monkeypatch):
    embedder = SentenceTransformerEmbedder("modele-test")
    error = ToolUnavailableError(code="embedding_unavailable", message="pas de modèle")
    monkeypatch.setattr(embedder, "_load", lambda: (_ for _ in ()).throw(error))

    with pytest.raises(ToolUnavailableError):
        await embedder.embed("question ?")
    await asyncio.gather(embedder._warm_up_task, return_exceptions=True)

    with pytest.raises(ToolUnavailableError) as info:
        await embedder.embed("question ?")
    assert info.value.code == "embedding_unavailable"


def test_missing_model_without_network_is_unavailable(monkeypatch):
    import sentence_transformers

    def offline(*args, **kwargs):
        raise OSError("pas de réseau")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", offline)

    with pytest.raises(ToolUnavailableError) as info:
        SentenceTransformerEmbedder("modele-absent")._load()
    assert info.value.code == "embedding_unavailable"


# --- Milvus ---------------------------------------------------------------------------


class FakeMilvus:
    def __init__(self, *, exists: bool = True, hits=None, error: Exception | None = None):
        self.exists = exists
        self.hits = hits or []
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def _record(self, method, **kwargs):
        self.calls.append((method, kwargs))
        if self.error:
            raise self.error

    def has_collection(self, name):
        self._record("has_collection", name=name)
        return self.exists

    def search(self, **kwargs):
        self._record("search", **kwargs)
        return [self.hits]

    def drop_collection(self, name):
        self._record("drop_collection", name=name)
        self.exists = False

    def prepare_index_params(self):
        from pymilvus import MilvusClient

        return MilvusClient.prepare_index_params()

    def create_collection(self, **kwargs):
        self._record("create_collection", **kwargs)
        self.exists = True

    def delete(self, **kwargs):
        self._record("delete", **kwargs)

    def upsert(self, **kwargs):
        self._record("upsert", **kwargs)

    def query(self, **kwargs):
        self._record("query", **kwargs)
        return [{"count(*)": 42}]


def hit(chunk_id: str, score: float) -> dict:
    return {
        "id": chunk_id,
        "distance": score,
        "entity": {
            "text": "Les Noirs contestent le centre.",
            "article_id": "defense-sicilienne",
            "title": "Défense sicilienne",
            "opening": "Sicilian Defense",
            "opening_family": "Sicilian Defense",
            "section": "Défense sicilienne > Idées",
            "source_url": "https://fr.wikipedia.org/w/index.php?oldid=1",
        },
    }


def store_with(client: FakeMilvus) -> MilvusVectorStore:
    return MilvusVectorStore(
        "http://milvus.test", "openings", dimension=3, client_factory=lambda: client
    )


async def test_search_maps_hits_to_passages():
    client = FakeMilvus(hits=[hit("recursive:defense-sicilienne:000", 0.81234)])

    (passage,) = await store_with(client).search([0.1, 0.2, 0.3], limit=4, opening=None)

    assert passage.chunk_id == "recursive:defense-sicilienne:000"
    assert passage.score == 0.8123
    assert passage.title == "Défense sicilienne"
    _, search = client.calls[-1]
    assert search["limit"] == 4
    assert search["filter"] == ""
    assert search["search_params"]["metric_type"] == "COSINE"


async def test_search_with_opening_filter():
    client = FakeMilvus()

    await store_with(client).search([0.1], limit=2, opening="Italian Game: Evans Gambit")

    assert client.calls[-1][1]["filter"] == (
        'opening == "Italian Game: Evans Gambit" or opening_family == "Italian Game"'
    )


@pytest.mark.parametrize(
    ("opening", "expected"),
    [
        (None, None),
        ("", None),
        (
            "Sicilian Defense",
            'opening == "Sicilian Defense" or opening_family == "Sicilian Defense"',
        ),
        ('Bad "quote"', 'opening == "Bad \\"quote\\"" or opening_family == "Bad \\"quote\\""'),
    ],
)
def test_opening_filter(opening, expected):
    assert opening_filter(opening) == expected


async def test_missing_collection_asks_for_ingestion():
    with pytest.raises(ToolUnavailableError) as info:
        await store_with(FakeMilvus(exists=False)).search([0.1], limit=1, opening=None)

    assert info.value.code == "not_indexed"
    assert "ingest_corpus" in info.value.hint


async def test_milvus_errors_become_transient():
    from pymilvus.exceptions import MilvusException

    client = FakeMilvus(error=MilvusException(message="service indisponible"))

    with pytest.raises(ToolTransientError) as info:
        await store_with(client).search([0.1], limit=1, opening=None)

    assert info.value.code == "vector_store_error"


async def test_unexpected_errors_are_not_hidden():
    with pytest.raises(ZeroDivisionError):
        await store_with(FakeMilvus(error=ZeroDivisionError())).count()


def test_default_factory_reports_unreachable_milvus(monkeypatch):
    import pymilvus

    def refuse(**kwargs):
        raise ConnectionError("refusé")

    monkeypatch.setattr(pymilvus, "MilvusClient", refuse)

    with pytest.raises(ToolTransientError) as info:
        MilvusVectorStore("http://nulle-part:19530", "openings")._default_factory()
    assert info.value.code == "network_error"


async def test_replace_article_deletes_then_upserts():
    client = FakeMilvus()
    chunks = chunk_article(article(MARKDOWN), words, CONFIG)

    await store_with(client).replace_article(
        "test", "test", chunks, [[0.1, 0.2, 0.3]] * len(chunks)
    )

    (delete, delete_args), (upsert, upsert_args) = client.calls
    assert delete == "delete"
    assert delete_args["filter"] == 'article_id == "test" and strategy == "test"'
    assert upsert == "upsert"
    assert len(upsert_args["data"]) == len(chunks)
    assert upsert_args["data"][0]["id"] == "test:test:000"


async def test_replace_article_with_no_chunk_only_deletes():
    client = FakeMilvus()

    await store_with(client).replace_article("test", "test", [], [])

    assert [name for name, _ in client.calls] == ["delete"]


async def test_recreate_drops_and_creates_with_hnsw_index():
    client = FakeMilvus(exists=True)

    await store_with(client).recreate()

    names = [name for name, _ in client.calls]
    assert names == ["has_collection", "drop_collection", "create_collection"]
    created = client.calls[-1][1]
    fields = {f.name for f in created["schema"].fields}
    assert {
        "id",
        "vector",
        "text",
        "date",
        "version",
        "service_emetteur",
        "confidentialite",
    } <= fields


async def test_ensure_collection_creates_only_if_missing():
    existing, missing = FakeMilvus(exists=True), FakeMilvus(exists=False)

    await store_with(existing).ensure_collection()
    await store_with(missing).ensure_collection()

    assert [n for n, _ in existing.calls] == ["has_collection"]
    assert [n for n, _ in missing.calls] == ["has_collection", "create_collection"]


async def test_count_uses_count_star_not_stats():
    client = FakeMilvus()

    assert await store_with(client).count() == 42
    assert client.calls[-1][1]["output_fields"] == ["count(*)"]


def test_row_contains_filterable_metadata():
    chunk = chunk_article(article(MARKDOWN), words, CONFIG)[0]

    data = row(chunk, [0.1, 0.2, 0.3])

    assert data["version"] == 1
    assert data["date"] == "2026-01-01T00:00:00Z"
    assert data["confidentialite"] == "public"
    assert data["type_document"] == "article"
    assert data["strategy"] == "test"
