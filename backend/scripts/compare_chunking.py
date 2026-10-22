"""Phase 3 : compare les stratégies de découpage sur 50 requêtes (Réf-30 + Multi-10).

Trois séries d'expériences, une variable à la fois :
  A. familles de découpage, à taille comparable (~512 tokens) ;
  B. taille des chunks récursifs : 256 / 512 / 1024 (recouvrement 10 %) ;
  C. recouvrement à 512 tokens : 0 %, 10 %, 25 %.

    uv run python -m scripts.compare_chunking
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict

from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.rag.chunking import model_token_counter
from app.rag.corpus import load_corpus
from app.rag.eval.bench import CachedEmbedder, benchmark_queries, run_strategy
from app.rag.eval.datasets import EVAL_DIR
from app.rag.eval.strategies import (
    fixed_chunks,
    hierarchical_chunks,
    html_chunks,
    markdown_chunks,
    model_tokenizer,
    recursive_chunks,
    semantic_chunks,
)
from scripts.fetch_corpus import CORPUS_DIR

RESULTS = EVAL_DIR / "results" / "phase3-chunking.json"


async def main() -> None:
    started = time.perf_counter()
    articles = load_corpus(CORPUS_DIR)
    count = model_token_counter(settings.embedding_model)
    tokenizer = model_tokenizer(settings.embedding_model)
    embedder = CachedEmbedder(SentenceTransformer(settings.embedding_model))
    queries = benchmark_queries()

    def all_chunks(fn):
        return [chunk for article in articles for chunk in fn(article)]

    def fixed(size: int, overlap: int):
        return lambda a: fixed_chunks(a, count, tokenizer=tokenizer, size=size, overlap=overlap)

    def recursive(size: int, overlap: int):
        return lambda a: recursive_chunks(a, count, size=size, overlap=overlap)

    experiments = {
        "A. Familles (~512 tokens)": [
            ("Fixe 512 / 51", fixed(512, 51), False),
            ("Récursif 512 / 51 (référence)", recursive(512, 51), True),
            ("Titres Markdown (sections)", lambda a: markdown_chunks(a, count), True),
            ("Sections HTML (brut)", lambda a: html_chunks(a, count), False),
            (
                "Sections HTML (balises de liens retirées)",
                lambda a: html_chunks(a, count, preprocess=True),
                False,
            ),
            (
                "Sémantique (percentile 95)",
                lambda a: semantic_chunks(a, count, embeddings=embedder),
                True,
            ),
            (
                "Hiérarchique (enfants 256 → section)",
                lambda a: hierarchical_chunks(a, count),
                False,
            ),
            (
                "Hiérarchique (enfants 256 → parent ≤ 1024)",
                lambda a: hierarchical_chunks(a, count, parent_max_tokens=1024),
                False,
            ),
        ],
        "B. Taille (récursif, recouvrement 10 %)": [
            ("Récursif 256", recursive(256, 25), False),
            ("Récursif 512", recursive(512, 51), False),
            ("Récursif 1024", recursive(1024, 102), False),
        ],
        "C1. Recouvrement (fixe 512 tokens)": [
            ("Fixe, 0 %", fixed(512, 0), False),
            ("Fixe, 10 % (51 tokens)", fixed(512, 51), False),
            ("Fixe, 25 % (128 tokens)", fixed(512, 128), False),
        ],
        "C2. Recouvrement (récursif 512 tokens)": [
            ("Récursif, 0 %", recursive(512, 0), False),
            ("Récursif, 10 % (51 tokens)", recursive(512, 51), False),
            ("Récursif, 25 % (128 tokens)", recursive(512, 128), False),
        ],
    }

    results: dict[str, list[dict]] = {}
    for series, runs in experiments.items():
        print(f"\n## {series}\n")
        print(
            "| Stratégie | Chunks | Tokens moy. (min-max) | Écart-type | Redondance "
            "| Atteignables | recall@5 Réf-30 | recall@5 (50) | MRR (50) | Tokens top-5 "
            "| recall à 2 000 tokens (50) |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        results[series] = []
        for name, fn, with_coherence in runs:
            result = await run_strategy(
                name,
                all_chunks(fn),
                articles,
                embedder,
                count,
                queries,
                coherence=with_coherence,
            )
            results[series].append(asdict(result))
            print(
                f"| {name} | {result.chunks} | {result.tokens_mean:.0f} "
                f"({result.tokens_min}-{result.tokens_max}) | {result.tokens_stdev:.0f} "
                f"| {result.redundancy:.0%} | {result.reachable:.0%} | {result.recall_ref30:.2f} | "
                f"{result.recall_all:.2f} | {result.mrr_all:.2f} | {result.context_tokens:.0f} "
                f"| {result.recall_budget:.2f} |"
            )

    print("\n## Cohésion et coupures\n")
    print("| Stratégie | Cohésion (phrase ↔ centre du chunk) | Similarité aux coupures |")
    print("|---|---|---|")
    for item in results["A. Familles (~512 tokens)"]:
        if item["cohesion"] is not None:
            print(f"| {item['name']} | {item['cohesion']:.3f} | {item['cut_similarity']:.3f} |")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"\n{embedder.encoded} textes encodés (le reste venait du cache) en "
        f"{time.perf_counter() - started:.0f} s → {RESULTS}"
    )


if __name__ == "__main__":
    asyncio.run(main())
