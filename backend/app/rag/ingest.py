"""Pipeline d'ingestion : corpus → unités d'index → vecteurs → Milvus (+ textes bruts).

Unités : enfants découpés par type + résumés de section (voir `app.rag.indexing`).
Relançable : les unités d'un article sont supprimées puis réinsérées (identifiants
stables), donc deux ingestions successives donnent le même index, sans doublon.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.rag.chunking import TokenCounter
from app.rag.corpus import Article
from app.rag.indexing import IndexUnit, build_units


class DocumentEmbedder(Protocol):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class IndexStore(Protocol):
    async def recreate(self) -> None: ...

    async def ensure_collection(self) -> None: ...

    async def replace_article(
        self, article_id: str, units: Sequence[IndexUnit], vectors: Sequence[list[float]]
    ) -> None: ...

    async def count(self) -> int: ...


# Sauvegarde des textes bruts (MongoDB dans l'application, rien dans les tests)
ArticleSink = Callable[[Article], Awaitable[None]]


@dataclass
class IngestReport:
    collection_count: int = 0
    chunks_per_article: dict[str, int] = field(default_factory=dict)
    tokens: int = 0
    summaries: int = 0
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
    summaries: Mapping[str, str] | None = None,
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
        units = build_units(article, count_tokens, summaries)
        vectors = await embedder.embed_documents([unit.text for unit in units])
        await store.replace_article(article.id, units, vectors)
        if save_article is not None:
            await save_article(article)
        report.chunks_per_article[article.id] = len(units)
        report.tokens += sum(count_tokens(unit.text) for unit in units)
        report.summaries += sum(unit.kind == "resume" for unit in units)

    report.collection_count = await store.count()
    report.duration_s = round(time.perf_counter() - started, 1)
    return report


def mongo_sink(database: Any, collection: str = "corpus_articles") -> ArticleSink:
    """Enregistre le texte brut et les métadonnées de chaque article dans MongoDB."""

    async def save(article: Article) -> None:
        document = article.model_dump(exclude={"html"})
        await database[collection].replace_one({"_id": article.id}, document, upsert=True)

    return save
