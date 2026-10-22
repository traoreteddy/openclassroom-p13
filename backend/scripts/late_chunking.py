"""Phase 3, étape 3.6 : late chunking comparé au découpage classique.

Mêmes chunks (récursif 256 tokens, recouvrement 10 %), mêmes 50 requêtes ; seule la façon
de calculer le vecteur de chaque chunk change.

    uv run python -m scripts.late_chunking
"""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.rag.chunking import model_token_counter
from app.rag.corpus import load_corpus
from app.rag.eval.bench import CachedEmbedder, MemoryIndex, benchmark_queries
from app.rag.eval.datasets import EVAL_DIR
from app.rag.eval.late_chunking import late_chunks, spans
from app.rag.eval.retrieval import evaluate
from app.rag.eval.strategies import build
from scripts.fetch_corpus import CORPUS_DIR


async def main() -> None:
    started = time.perf_counter()
    articles = load_corpus(CORPUS_DIR)
    count = model_token_counter(settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)
    embedder = CachedEmbedder(model)
    queries = benchmark_queries()

    variants = {}
    classic = [
        c
        for a in articles
        for c in build(a, [("", t) for _, t in spans(a, count)[1]], "classic", count)
    ]
    variants["Classique (chunk encodé seul)"] = (
        classic,
        embedder.encode([c.text for c in classic]),
    )
    for pooling, label in [
        ("mean", "Late chunking, moyenne"),
        ("last", "Late chunking, dernier token"),
    ]:
        chunks, vectors = [], []
        for article in articles:
            article_chunks, article_vectors = late_chunks(article, count, model, pooling)
            chunks.extend(article_chunks)
            vectors.append(article_vectors)
        variants[label] = (chunks, np.concatenate(vectors))

    print("| Variante | Chunks | recall@5 (50) | MRR (50) | Manquées |")
    print("|---|---|---|---|---|")
    results = {}
    for name, (chunks, vectors) in variants.items():
        index = MemoryIndex(chunks, vectors)

        async def search(query, k, opening, index=index):
            return index.search(embedder.encode([query], query=True)[0], k, opening)

        report = await evaluate(queries, search, k=5)
        results[name] = {"recall@5": report.recall, "mrr": report.mrr, "missed": report.missed}
        print(
            f"| {name} | {len(chunks)} | {report.recall:.2f} | {report.mrr:.2f} "
            f"| {len(report.missed)} |"
        )

    path = EVAL_DIR / "results" / "phase3-late-chunking.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nTerminé en {time.perf_counter() - started:.0f} s → {path}")


if __name__ == "__main__":
    asyncio.run(main())
