"""Phase 1, étape 1.3 : génère les questions candidates de Réf-30 avec le LLM local.

3 passages rédigés par article (répartis sur tout l'article) → une question chacun.
Résultat : backend/data/eval/ref30_candidates.jsonl, à relire (étape 1.4).

    uv run python -m scripts.generate_ref_candidates
"""

from __future__ import annotations

import asyncio
import time

import httpx

from app.core.config import settings
from app.rag.corpus import load_corpus
from app.rag.eval.datasets import EVAL_DIR
from app.rag.eval.generate import OllamaChat, generate_candidate
from app.rag.eval.passages import iter_passages, spread_sample
from scripts.fetch_corpus import CORPUS_DIR

PER_ARTICLE = 3


async def main() -> None:
    passages = []
    for article in load_corpus(CORPUS_DIR):
        prose = [p for p in iter_passages(article) if p.is_prose]
        passages.extend(spread_sample(prose, PER_ARTICLE))

    out = EVAL_DIR / "ref30_candidates.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    async with httpx.AsyncClient() as http:
        llm = OllamaChat(http, base_url=settings.ollama_base_url, model=settings.llm_model)
        with out.open("w", encoding="utf-8") as file:
            for index, passage in enumerate(passages, start=1):
                candidate = await generate_candidate(llm, passage)
                file.write(candidate.model_dump_json() + "\n")
                file.flush()
                mark = "✓" if candidate.status == "valide" else "✗"
                print(f"{index:>2}/{len(passages)} {mark} {passage.breadcrumb[:60]}")

    print(f"\n{len(passages)} candidats en {time.perf_counter() - started:.0f} s → {out}")


if __name__ == "__main__":
    asyncio.run(main())
