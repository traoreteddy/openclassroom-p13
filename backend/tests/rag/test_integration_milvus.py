"""Test d'intégration : vrai Milvus, vrai modèle, vraie ingestion (exclu par défaut).

Prérequis : `docker compose up -d milvus`, modèle d'embedding dans le cache.

    uv run pytest -m integration tests/rag/test_integration_milvus.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import settings
from app.rag.chunking import model_token_counter
from app.rag.corpus import load_corpus
from app.rag.embedding import SentenceTransformerEmbedder
from app.rag.ingest import ingest
from app.rag.milvus_store import MilvusVectorStore

pytestmark = pytest.mark.integration

CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"
COLLECTION = "test_integration_openings"


@pytest.fixture(autouse=True)
def _allow_network(respx_mock):
    """Ce test parle à Milvus en gRPC (hors respx) : rien à simuler."""
    respx_mock.route().pass_through()


async def test_ingest_then_search_finds_the_right_article():
    articles = [a for a in load_corpus(CORPUS) if a.id in {"gambit-evans", "defense-sicilienne"}]
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    await embedder.warm_up()
    store = MilvusVectorStore(settings.milvus_uri, COLLECTION, dimension=embedder.dimension)

    report = await ingest(
        articles,
        count_tokens=model_token_counter(settings.embedding_model),
        embedder=embedder,
        store=store,
        recreate=True,
    )
    again = await ingest(
        articles,
        count_tokens=model_token_counter(settings.embedding_model),
        embedder=embedder,
        store=store,
    )

    assert report.collection_count == report.total_chunks
    assert again.collection_count == report.total_chunks  # relance sans doublon

    hits = await store.search(
        await embedder.embed("Qui a donné son nom au gambit Evans ?"), limit=3, opening=None
    )
    assert hits[0].article_id == "gambit-evans"
    assert hits[0].score > 0.6

    filtered = await store.search(
        await embedder.embed("Qui a donné son nom au gambit Evans ?"),
        limit=3,
        opening="Sicilian Defense",
    )
    assert {h.article_id for h in filtered} == {"defense-sicilienne"}
