"""Phase 4, correction C5 : valide la configuration gelée sur Holdout-20 (jamais vu).

uv run python -m scripts.validate_holdout
"""

from __future__ import annotations

import asyncio
import json
from math import comb

from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.rag.chunking import model_token_counter
from app.rag.corpus import load_corpus
from app.rag.enrich import EnrichmentCache
from app.rag.eval.bench import CachedEmbedder, run_strategy
from app.rag.eval.datasets import EVAL_DIR, RefQuestion, load_jsonl
from app.rag.eval.strategies import add_generated_units, hierarchical_chunks, recursive_chunks
from app.rag.typed_chunking import typed_hierarchical_chunks
from scripts.fetch_corpus import CORPUS_DIR
from scripts.generate_enrichment import ENRICHMENT_DIR


def mcnemar(gained: int, lost: int) -> float:
    n, k = gained + lost, min(gained, lost)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


async def main() -> None:
    articles = load_corpus(CORPUS_DIR)
    count = model_token_counter(settings.embedding_model)
    embedder = CachedEmbedder(SentenceTransformer(settings.embedding_model))
    holdout = load_jsonl(RefQuestion, EVAL_DIR / "holdout20.jsonl")
    cache = EnrichmentCache(ENRICHMENT_DIR / f"{settings.llm_model.replace(':', '_')}.jsonl")

    typed = [i for a in articles for i in typed_hierarchical_chunks(a, count)]
    summaries = {i.parent: [cache.get(i.parent).summary] for i in typed}
    configurations = {
        "Phase 2 : récursif 512": [
            c for a in articles for c in recursive_chunks(a, count, size=512, overlap=51)
        ],
        "Phase 3 : hiérarchique (parents ≤ 1 024)": [
            i for a in articles for i in hierarchical_chunks(a, count, parent_max_tokens=1024)
        ],
        "Phase 4 : configuration gelée": add_generated_units(typed, summaries, "summary", count),
    }

    print("| Configuration | recall@5 | MRR | Tokens top-5 | recall à 2 000 tokens | vs phase 2 |")
    print("|---|---|---|---|---|---|")
    results, baseline = {}, None
    for name, items in configurations.items():
        r = await run_strategy(name, items, articles, embedder, count, holdout)
        missed = set(r.missed)
        if baseline is None:
            baseline, versus = missed, "—"
        else:
            gained, lost = len(baseline - missed), len(missed - baseline)
            versus = f"+{gained} / -{lost} (p = {mcnemar(gained, lost):.2f})"
        results[name] = {
            "recall@5": r.recall_all,
            "mrr": r.mrr_all,
            "context_tokens": r.context_tokens,
            "recall_budget": r.recall_budget,
            "missed": r.missed,
        }
        print(
            f"| {name} | {r.recall_all:.2f} | {r.mrr_all:.2f} | {r.context_tokens:.0f} "
            f"| {r.recall_budget:.2f} | {versus} |"
        )
        print(f"|   manquées | {', '.join(sorted(r.missed)) or 'aucune'} | | | | |")

    path = EVAL_DIR / "results" / "phase4-holdout.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
