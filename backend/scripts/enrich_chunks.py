"""Phase 4 : mesure l'effet de l'enrichissement des chunks (stratégie hiérarchique).

Base : enfants récursifs 256 tokens, parents = sections ≤ 1 024 tokens (retenue en phase 3).
Une variable à la fois : fil d'Ariane, puis questions hypothétiques, puis résumé.

    uv run python -m scripts.enrich_chunks
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.rag.chunking import model_token_counter
from app.rag.corpus import load_corpus
from app.rag.enrich import EnrichmentCache
from app.rag.eval.bench import CachedEmbedder, benchmark_queries, run_strategy
from app.rag.eval.datasets import EVAL_DIR
from app.rag.eval.strategies import add_generated_units, hierarchical_chunks
from app.rag.typed_chunking import typed_hierarchical_chunks
from scripts.fetch_corpus import CORPUS_DIR
from scripts.generate_enrichment import ENRICHMENT_DIR

RESULTS = EVAL_DIR / "results" / "phase4-enrichment.json"


def print_row(name, r) -> None:
    print(
        f"| {name} | {r.chunks} | {r.recall_ref30:.2f} | {r.recall_multi:.2f} | "
        f"{r.recall_all:.2f} | {r.mrr_all:.2f} | {r.context_tokens:.0f} | {r.recall_budget:.2f} |"
    )


async def main() -> None:
    articles = load_corpus(CORPUS_DIR)
    count = model_token_counter(settings.embedding_model)
    embedder = CachedEmbedder(SentenceTransformer(settings.embedding_model))
    queries = benchmark_queries()

    def hierarchical(**options):
        return [
            item
            for article in articles
            for item in hierarchical_chunks(article, count, parent_max_tokens=1024, **options)
        ]

    cache = EnrichmentCache(ENRICHMENT_DIR / f"{settings.llm_model.replace(':', '_')}.jsonl")
    base = hierarchical(breadcrumb=True)
    typed = [i for a in articles for i in typed_hierarchical_chunks(a, count)]
    parents = {item.parent for item in base} | {item.parent for item in typed}
    enrichments = {text: cache.get(text) for text in parents}
    missing = [t for t, e in enrichments.items() if e is None]
    if missing:
        raise SystemExit(f"{len(missing)} sections non enrichies : lancer generate_enrichment")
    questions = {t: e.questions for t, e in enrichments.items()}
    summaries = {t: [e.summary] for t, e in enrichments.items()}

    with_questions = add_generated_units(base, questions, "question", count)
    variants = {
        "Hiérarchique (phase 3)": hierarchical(),
        "+ fil d'Ariane": base,
        "+ fil d'Ariane + questions hypothétiques": with_questions,
        "+ fil d'Ariane + résumé": add_generated_units(base, summaries, "summary", count),
        "+ fil d'Ariane + questions + résumé": add_generated_units(
            with_questions, summaries, "summary", count
        ),
    }
    variants["Par type + fil d'Ariane"] = typed
    variants["Par type + fil d'Ariane + résumé"] = add_generated_units(
        typed, summaries, "summary", count
    )
    variants["Par type + fil d'Ariane + résumé avec fil d'Ariane"] = add_generated_units(
        typed, summaries, "summary", count, breadcrumb=True
    )

    print(
        "| Variante | Unités indexées | recall@5 Réf-30 | recall@5 Multi (20) | recall@5 (50) "
        "| MRR (50) | Tokens top-5 | recall à 2 000 tokens |"
    )
    print("|---|---|---|---|---|---|---|---|")
    results = {}
    for name, items in variants.items():
        result = await run_strategy(name, items, articles, embedder, count, queries)
        results[name] = asdict(result)
        print_row(name, result)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
