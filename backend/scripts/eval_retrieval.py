"""Mesure la recherche sur Réf-30 (recall@k, MRR, precision@k) et calibre le seuil.

Recherche brute (sans le seuil de pertinence de l'outil), dans la collection indiquée.

    uv run python -m scripts.eval_retrieval [--collection openings] [--label phase2-reference]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics

from app.core.config import settings
from app.rag.embedding import SentenceTransformerEmbedder
from app.rag.eval.datasets import EVAL_DIR, DecisionQuestion, RefQuestion, load_jsonl
from app.rag.eval.retrieval import calibrate, evaluate
from app.rag.milvus_store import MilvusVectorStore

RESULTS_DIR = EVAL_DIR / "results"


async def main(collection: str, label: str) -> None:
    ref30 = load_jsonl(RefQuestion, EVAL_DIR / "ref30.jsonl")
    decision50 = load_jsonl(DecisionQuestion, EVAL_DIR / "decision50.jsonl")
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    await embedder.warm_up()
    store = MilvusVectorStore(settings.milvus_uri, collection)

    async def search(query: str, k: int, opening: str | None):
        return await store.search(await embedder.embed(query), limit=k, opening=opening)

    plain = await evaluate(ref30, search, k=5)
    filtered = await evaluate(ref30, search, k=5, use_filter=True)

    out_scores = []
    for item in decision50:
        if not item.search_useful:
            hits = await search(item.question, 1, None)
            out_scores.append(hits[0].score if hits else 0.0)
    in_scores = [r.top_score for r in plain.results]
    thresholds = calibrate(in_scores, out_scores)
    best = max(thresholds, key=lambda t: (t.balanced_accuracy, -t.value))

    print(f"Collection « {collection} » — Réf-30 (30 questions)\n")
    print("| Variante | recall@1 | recall@3 | recall@5 | MRR | precision@5 |")
    print("|---|---|---|---|---|---|")
    for name, report in [("Sans filtre", plain), ("Avec filtre d'ouverture", filtered)]:
        print(
            f"| {name} | {report.recall_at(1):.2f} | {report.recall_at(3):.2f} | "
            f"{report.recall:.2f} | {report.mrr:.2f} | {report.precision:.2f} |"
        )
    print(f"\nManquées (sans filtre) : {plain.missed}")
    print(f"Manquées (avec filtre) : {filtered.missed}")

    print("\nMeilleur score par question :")
    print(
        f"  dans le corpus (30) : min {min(in_scores):.2f}, médiane "
        f"{statistics.median(in_scores):.2f}, max {max(in_scores):.2f}"
    )
    print(
        f"  hors corpus (20)    : min {min(out_scores):.2f}, médiane "
        f"{statistics.median(out_scores):.2f}, max {max(out_scores):.2f}"
    )
    print("\n| Seuil | Questions du corpus gardées | Questions hors corpus écartées |")
    print("|---|---|---|")
    for t in thresholds:
        mark = " ←" if t is best else ""
        print(f"| {t.value:.2f} | {t.kept_in_corpus:.0%} | {t.rejected_out_of_corpus:.0%} |{mark}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{label}.json"
    path.write_text(
        json.dumps(
            {
                "collection": collection,
                "plain": {
                    "recall@1": plain.recall_at(1),
                    "recall@3": plain.recall_at(3),
                    "recall@5": plain.recall,
                    "mrr": plain.mrr,
                    "precision@5": plain.precision,
                    "missed": plain.missed,
                    "ranks": {r.id: r.rank for r in plain.results},
                },
                "filtered": {
                    "recall@5": filtered.recall,
                    "mrr": filtered.mrr,
                    "missed": filtered.missed,
                },
                "scores": {"in_corpus": in_scores, "out_of_corpus": out_scores},
                "best_threshold": best.value,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nRésultats → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=settings.milvus_collection)
    parser.add_argument("--label", default="phase2-reference")
    args = parser.parse_args()
    asyncio.run(main(args.collection, args.label))
