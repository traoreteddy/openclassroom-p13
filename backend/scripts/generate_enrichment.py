"""Phase 4, étape 4.2 : génère questions hypothétiques et résumés des sections (phi4-mini).

Cache : backend/data/enrichment/<modèle>.jsonl (relancer ne régénère que les manquants).

    uv run python -m scripts.generate_enrichment [--bench]
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import httpx

from app.core.config import settings
from app.rag.chunking import model_token_counter
from app.rag.corpus import load_corpus
from app.rag.enrich import MAX_GENERATED_TOKENS, EnrichmentCache, enrich_sections
from app.rag.indexing import parent_sections
from app.rag.llm import OllamaChat
from scripts.fetch_corpus import CORPUS_DIR

ENRICHMENT_DIR = Path(__file__).resolve().parents[1] / "data" / "enrichment"


async def main() -> None:
    count = model_token_counter(settings.embedding_model)
    articles = load_corpus(CORPUS_DIR)
    sections = [s for a in articles for s in parent_sections(a, count)]
    if "--bench" in sys.argv:  # sections des variantes non typées comparées en phase 4
        from app.rag.eval.strategies import hierarchical_chunks

        known = {text for _, text in sections}
        for article in articles:
            for item in hierarchical_chunks(article, count, parent_max_tokens=1024):
                if item.parent not in known:
                    known.add(item.parent)
                    sections.append((item.child.section, item.parent))
    cache = EnrichmentCache(ENRICHMENT_DIR / f"{settings.llm_model.replace(':', '_')}.jsonl")
    before = len(cache)
    started = time.perf_counter()
    async with httpx.AsyncClient() as http:
        llm = OllamaChat(
            http,
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            max_tokens=MAX_GENERATED_TOKENS,
        )
        result = await enrich_sections(sections, llm, cache)
    questions = sum(len(e.questions) for e in result.values())
    empty = sum(not e.questions for e in result.values())
    failed = len(sections) - len(result)
    if failed:
        print(f"⚠ {failed} sections non enrichies (voir les avertissements) : relancer le script")
    print(
        f"{len(sections)} sections ({len(cache) - before} générées, {before} en cache) en "
        f"{time.perf_counter() - started:.0f} s ; {questions} questions, "
        f"{empty} sections sans question valide → {cache.path}"
    )


if __name__ == "__main__":
    asyncio.run(main())
