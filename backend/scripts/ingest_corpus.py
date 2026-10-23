"""Ingestion du corpus dans Milvus (livrable « script » de l'étape RAG).

Configuration retenue en phase 4 :
corpus figé → découpage par type (texte, liste, partie) avec fil d'Ariane, parents
≤ 1 024 tokens → résumé de chaque section (LLM local, en cache) → embeddings
Qwen3-Embedding-0.6B → Milvus (index multi-vecteurs) ; textes bruts dans MongoDB.

Depuis l'hôte (recommandé : le GPU accélère les embeddings), Milvus en conteneur :
    uv run python -m scripts.ingest_corpus --recreate

Options : --no-summaries (sans LLM), --skip-mongo, --collection NOM.
"""

from __future__ import annotations

import argparse
import asyncio

import httpx

from app.core.config import settings
from app.rag.chunking import model_token_counter
from app.rag.corpus import load_corpus
from app.rag.embedding import SentenceTransformerEmbedder
from app.rag.enrich import MAX_GENERATED_TOKENS, EnrichmentCache, enrich_sections
from app.rag.indexing import parent_sections
from app.rag.ingest import ingest, mongo_sink
from app.rag.llm import OllamaChat
from app.rag.milvus_store import MilvusVectorStore
from scripts.fetch_corpus import CORPUS_DIR
from scripts.generate_enrichment import ENRICHMENT_DIR


async def load_summaries(articles, count) -> dict[str, str]:
    """Résumés des sections : lus dans le cache, générés par le LLM seulement s'ils manquent."""
    cache = EnrichmentCache(ENRICHMENT_DIR / f"{settings.llm_model.replace(':', '_')}.jsonl")
    sections = [s for article in articles for s in parent_sections(article, count)]
    missing = sum(cache.get(text) is None for _, text in sections)
    if missing:
        print(
            f"{missing} sections à résumer avec {settings.llm_model} ({settings.ollama_base_url})"
        )
    async with httpx.AsyncClient() as http:
        llm = OllamaChat(
            http,
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            max_tokens=MAX_GENERATED_TOKENS,
        )
        enrichments = await enrich_sections(sections, llm, cache)
    return {text: enrichment.summary for text, enrichment in enrichments.items()}


async def main(collection: str, recreate: bool, skip_mongo: bool, summaries: bool) -> None:
    articles = load_corpus(CORPUS_DIR)
    count = model_token_counter(settings.embedding_model)
    summary_map = await load_summaries(articles, count) if summaries else None
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model, device=settings.embedding_device, dtype=settings.embedding_dtype
    )
    store = MilvusVectorStore(settings.milvus_uri, collection, dimension=embedder.dimension)

    sink = None
    if not skip_mongo:
        from app.core.database import close_db, get_db

        sink = mongo_sink(get_db())

    print(f"Ingestion de {len(articles)} articles dans « {collection} » ({settings.milvus_uri})")
    try:
        report = await ingest(
            articles,
            count_tokens=count,
            embedder=embedder,
            store=store,
            summaries=summary_map,
            recreate=recreate,
            save_article=sink,
        )
    finally:
        if not skip_mongo:
            await close_db()

    for article_id, units in report.chunks_per_article.items():
        print(f"  {article_id:<28} {units:>3} unités")
    print(
        f"\n{report.total_chunks} unités dont {report.summaries} résumés ({report.tokens} tokens) "
        f"en {report.duration_s} s ; la collection contient {report.collection_count} unités."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=settings.milvus_collection)
    parser.add_argument(
        "--recreate", action="store_true", help="supprime puis recrée la collection"
    )
    parser.add_argument("--skip-mongo", action="store_true", help="n'écrit pas dans MongoDB")
    parser.add_argument(
        "--no-summaries", action="store_true", help="indexe sans résumés (sans LLM)"
    )
    args = parser.parse_args()
    asyncio.run(main(args.collection, args.recreate, args.skip_mongo, not args.no_summaries))
