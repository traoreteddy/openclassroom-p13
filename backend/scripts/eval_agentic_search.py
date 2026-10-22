"""Phase 7 : évalue la boucle de recherche agentique sur Réf-30.

Métriques : rappel, précision (stricte et par article), sauts moyens, tokens de contexte,
motifs d'arrêt. Objectifs : rappel > 0,8 et moins de 3 sauts par question.
Chaque exécution est une session Langfuse (`eval-agentic-<modèle>-<horodatage>`), où
chaque question est une trace avec ses sous-requêtes (spans `decision`) et ses appels
d'outil (spans `tool_call`).

    uv run python -m scripts.eval_agentic_search [--model qwen3:14b]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from app.agent.llm import ollama_factory
from app.agent.multihop import Budget, MultiHopSearcher
from app.agent.tracing import configure_tracing
from app.core.config import settings
from app.rag.chunking import model_token_counter
from app.rag.eval.agentic import AgenticReport, assess
from app.rag.eval.datasets import EVAL_DIR, RefQuestion, load_jsonl
from app.tools.catalog import build_registry, default_dependencies
from scripts.eval_agent import CLOUD_EQUIVALENT, JOURNAL_DIR, stamp

RESULTS = EVAL_DIR / "results" / "phase7-agentic-search.json"


async def main(model: str) -> None:
    questions = load_jsonl(RefQuestion, EVAL_DIR / "ref30.jsonl")
    deps = default_dependencies(settings)
    registry = build_registry(settings, deps)
    await deps.knowledge._embedder.warm_up()  # type: ignore[attr-defined]
    label = f"eval-agentic-{model.replace(':', '_')}-{stamp()}"
    tracing = configure_tracing(
        environment="eval",
        langfuse_base_url=settings.langfuse_base_url,
        langfuse_public_key=settings.langfuse_public_key,
        langfuse_secret_key=settings.langfuse_secret_key,
        journal_path=JOURNAL_DIR / f"{label}.jsonl",
        pricing=CLOUD_EQUIVALENT,
    )
    searcher = MultiHopSearcher(
        registry,
        tracing,
        ollama_factory(base_url=settings.ollama_base_url, model=model),
        model_token_counter(settings.embedding_model),
        Budget(),
    )
    outcomes = []
    tokens = [0, 0]
    started = time.perf_counter()
    try:
        for q in questions:
            result = await searcher.search(q.question, session_id=label)
            tokens[0] += result.tokens_in
            tokens[1] += result.tokens_out
            outcome = assess(
                q,
                result.passages,
                hops=result.searches,
                context_tokens=result.context_tokens,
                stop_reason=result.stop_reason,
                sub_questions=[h.sub_question for h in result.hops],
                reformulations=sum(h.reformulated for h in result.hops),
            )
            outcomes.append(outcome)
            mark = "✓" if outcome.found else "✗"
            print(
                f"  {mark} {q.id} sauts={outcome.hops} retenus={outcome.retained} "
                f"pertinents={outcome.relevant} stop={outcome.stop_reason}"
            )
        tracing.shutdown()
    finally:
        await deps.aclose()

    report = AgenticReport(outcomes)
    elapsed = time.perf_counter() - started
    header = (
        "| Rappel | Précision stricte | Précision article | Sauts moyens (max) "
        "| Tokens de contexte | Temps / question |"
    )
    print(f"\n{header}\n|---|---|---|---|---|---|")
    print(
        f"| {report.recall:.2f} | {report.precision:.2f} | {report.article_precision:.2f} "
        f"| {report.mean_hops:.2f} ({report.max_hops}) | {report.mean_context_tokens:.0f} "
        f"| {elapsed / len(questions):.1f} s |"
    )
    print(f"Arrêts : {report.stop_reasons} ; manquées : {report.missed}")
    print(f"Objectifs : {report.meets_targets}")
    payload = report.to_dict() | {
        "session": label,
        "model": model,
        "tokens_in": tokens[0],
        "tokens_out": tokens[1],
        "cloud_equivalent_eur": CLOUD_EQUIVALENT.cost(*tokens),
        "seconds": round(elapsed),
    }
    RESULTS.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"→ {RESULTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=settings.agent_model)
    asyncio.run(main(parser.parse_args().model))
