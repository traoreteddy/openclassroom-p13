"""Ingestion du corpus dans Milvus (livrable « script » de l'étape RAG).

Corpus figé (data/corpus) → découpage récursif 512 tokens → embeddings
Qwen3-Embedding-0.6B → collection Milvus ; textes bruts dans MongoDB.

Dans Docker (recommandé) :
    docker compose run --rm backend python -m scripts.ingest_corpus

En local (Milvus et MongoDB lancés par docker compose) :
    uv run python -m scripts.ingest_corpus [--recreate] [--skip-mongo]
"""

from __future__ import annotations

import argparse
import asyncio

from app.core.config import settings
from app.rag.chunking import model_token_counter
from app.rag.corpus import load_corpus
from app.rag.embedding import SentenceTransformerEmbedder
from app.rag.ingest import ingest, mongo_sink
from app.rag.milvus_store import MilvusVectorStore
from scripts.fetch_corpus import CORPUS_DIR


async def main(collection: str, recreate: bool, skip_mongo: bool) -> None:
    articles = load_corpus(CORPUS_DIR)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = MilvusVectorStore(settings.milvus_uri, collection, dimension=embedder.dimension)

    sink = None
    if not skip_mongo:
        from app.core.database import close_db, get_db

        sink = mongo_sink(get_db())

    print(f"Ingestion de {len(articles)} articles dans « {collection} » ({settings.milvus_uri})")
    try:
        report = await ingest(
            articles,
            count_tokens=model_token_counter(settings.embedding_model),
            embedder=embedder,
            store=store,
            recreate=recreate,
            save_article=sink,
        )
    finally:
        if not skip_mongo:
            await close_db()

    for article_id, count in report.chunks_per_article.items():
        print(f"  {article_id:<28} {count:>3} chunks")
    print(
        f"\n{report.total_chunks} chunks ({report.tokens} tokens) en {report.duration_s} s ; "
        f"la collection contient {report.collection_count} chunks."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=settings.milvus_collection)
    parser.add_argument(
        "--recreate", action="store_true", help="supprime puis recrée la collection"
    )
    parser.add_argument("--skip-mongo", action="store_true", help="n'écrit pas dans MongoDB")
    args = parser.parse_args()
    asyncio.run(main(args.collection, args.recreate, args.skip_mongo))
