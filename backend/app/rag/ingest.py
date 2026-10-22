"""Pipeline d'ingestion : corpus → chunks → vecteurs → Milvus (+ textes bruts dans MongoDB).

Relançable : les chunks d'un article sont supprimés puis réinsérés (identifiants
stables), donc deux ingestions successives donnent le même index, sans doublon.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.rag.chunking import BASELINE, ChunkingConfig, TokenCounter, chunk_article
from app.rag.corpus import Article


class DocumentEmbedder(Protocol):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class IndexStore(Protocol):
    async def recreate(self) -> None: ...

    async def ensure_collection(self) -> None: ...

    async def replace_article(
        self, article_id: str, strategy: str, chunks: Sequence[Any], vectors: Sequence[list[float]]
    ) -> None: ...

    async def count(self) -> int: ...


# Sauvegarde des textes bruts (MongoDB dans l'application, rien dans les tests)
ArticleSink = Callable[[Article], Awaitable[None]]


@dataclass
class IngestReport:
    collection_count: int = 0
    chunks_per_article: dict[str, int] = field(default_factory=dict)
    tokens: int = 0
    duration_s: float = 0.0

    @property
    def total_chunks(self) -> int:
        return sum(self.chunks_per_article.values())


async def ingest(
    articles: Sequence[Article],
    *,
    count_tokens: TokenCounter,
    embedder: DocumentEmbedder,
    store: IndexStore,
    config: ChunkingConfig = BASELINE,
    recreate: bool = False,
    save_article: ArticleSink | None = None,
) -> IngestReport:
    started = time.perf_counter()
    report = IngestReport()
    if recreate:
        await store.recreate()
    else:
        await store.ensure_collection()

    for article in articles:
        chunks = chunk_article(article, count_tokens, config)
        vectors = await embedder.embed_documents([chunk.text for chunk in chunks])
        await store.replace_article(article.id, config.strategy, chunks, vectors)
        if save_article is not None:
            await save_article(article)
        report.chunks_per_article[article.id] = len(chunks)
        report.tokens += sum(chunk.token_count for chunk in chunks)

    report.collection_count = await store.count()
    report.duration_s = round(time.perf_counter() - started, 1)
    return report


def mongo_sink(database: Any, collection: str = "corpus_articles") -> ArticleSink:
    """Enregistre le texte brut et les métadonnées de chaque article dans MongoDB."""

    async def save(article: Article) -> None:
        document = article.model_dump(exclude={"html"})
        await database[collection].replace_one({"_id": article.id}, document, upsert=True)

    return save
