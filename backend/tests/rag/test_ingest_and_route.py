"""Pipeline d'ingestion (doublures) et endpoint /vector-search."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.deps import get_registry
from app.main import app
from app.rag.ingest import ingest, mongo_sink
from tests.conftest import StubVectorStore, passage
from tests.rag.test_chunking import CONFIG, MARKDOWN, words
from tests.rag.test_text_and_passages import article


class RecordingStore:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.rows: dict[str, int] = {}

    async def recreate(self) -> None:
        self.calls.append("recreate")

    async def ensure_collection(self) -> None:
        self.calls.append("ensure")

    async def replace_article(self, article_id, strategy, chunks, vectors) -> None:
        assert len(chunks) == len(vectors)
        self.rows[article_id] = len(chunks)

    async def count(self) -> int:
        return sum(self.rows.values())


class CountingEmbedder:
    def __init__(self) -> None:
        self.batches: list[int] = []

    async def embed_documents(self, texts):
        self.batches.append(len(texts))
        return [[0.0, 1.0] for _ in texts]


async def test_ingest_chunks_embeds_and_indexes_each_article():
    store, embedder = RecordingStore(), CountingEmbedder()
    saved: list[str] = []

    async def sink(a):
        saved.append(a.id)

    report = await ingest(
        [article(MARKDOWN)],
        count_tokens=words,
        embedder=embedder,
        store=store,
        config=CONFIG,
        save_article=sink,
    )

    assert store.calls == ["ensure"]
    assert report.chunks_per_article == {"test": embedder.batches[0]}
    assert report.total_chunks == report.collection_count
    assert report.tokens > 0
    assert saved == ["test"]


async def test_ingest_can_recreate_the_collection():
    store = RecordingStore()

    await ingest(
        [article(MARKDOWN)],
        count_tokens=words,
        embedder=CountingEmbedder(),
        store=store,
        config=CONFIG,
        recreate=True,
    )

    assert store.calls == ["recreate"]


async def test_mongo_sink_upserts_without_html():
    class Collection:
        def __init__(self) -> None:
            self.calls = []

        async def replace_one(self, query, document, upsert):
            self.calls.append((query, document, upsert))

    collection = Collection()

    await mongo_sink({"corpus_articles": collection})(article(MARKDOWN))

    query, document, upsert = collection.calls[0]
    assert query == {"_id": "test"}
    assert upsert is True
    assert "html" not in document
    assert document["metadata"]["version"] == 1


def test_vector_search_endpoint(make_registry):
    store = StubVectorStore([passage("Les Noirs contestent le centre avec c5.", 0.8)])
    registry = make_registry(store=store)
    app.dependency_overrides[get_registry] = lambda: registry
    try:
        client = TestClient(app)
        ok = client.get(
            "/api/v1/vector-search",
            params={"query": "idées de la sicilienne", "opening": "Sicilian Defense", "limit": 2},
        )
        too_short = client.get("/api/v1/vector-search", params={"query": "ab"})
        fen = client.get(
            "/api/v1/vector-search",
            params={"query": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"},
        )
    finally:
        app.dependency_overrides.clear()

    assert ok.status_code == 200
    assert ok.json()["found"] is True
    assert ok.json()["passages"][0]["opening"] == "Sicilian Defense"
    assert store.calls == [{"limit": 2, "opening": "Sicilian Defense"}]
    assert too_short.status_code == 422  # validation FastAPI
    assert fen.status_code == 422  # validation de l'outil : une FEN n'est pas une question
    assert fen.json()["error"]["code"] == "invalid_arguments"
