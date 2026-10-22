"""Phase 6 : dix questions multi-sauts, recherche itérative contre RAG en un seul passage.

Pour chaque question de Multi-10 :
- **un passage** : une recherche avec la question entière (5 passages), puis réponse ;
- **multi-sauts** : décomposition, boucle sous budget (3 recherches, 6 000 tokens), réponse.

Mesures : sauts attendus retrouvés (l'extrait de chaque saut est-il dans le contexte
retenu ?), nombre de recherches, tokens de contexte, et réponse contenant les termes clés
de la réponse de référence.

    uv run python -m scripts.eval_multihop [--model qwen3:14b]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import ollama_factory
from app.agent.multihop import ANSWER_PROMPT, SEARCH_TOOL, Budget, MultiHopSearcher, _render
from app.agent.tracing import configure_tracing
from app.core.config import settings
from app.rag.chunking import model_token_counter
from app.rag.eval.datasets import EVAL_DIR, MultiHopQuestion, load_jsonl
from app.rag.text import contains_excerpt
from app.tools.catalog import build_registry, default_dependencies
from scripts.eval_agent import CLOUD_EQUIVALENT, JOURNAL_DIR, stamp

RESULTS = EVAL_DIR / "results" / "phase6-multihop.json"


def key_terms(answer: str) -> list[str]:
    """Termes distinctifs de la réponse de référence : nombres, coups, noms propres."""
    terms = set()
    for token in re.findall(r"[A-Za-zÀ-ÿ0-9.…'-]+", answer):
        clean = token.strip(".,;:()'").strip()
        if len(clean) < 2:
            continue
        if any(c.isdigit() for c in clean) or (clean[0].isupper() and len(clean) >= 4):
            terms.add(clean.lower())
    return sorted(terms)


def answer_matches(answer: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    lowered = answer.lower()
    return sum(t in lowered for t in terms) / len(terms)


def hops_found(question: MultiHopQuestion, passages: list[dict[str, Any]]) -> list[bool]:
    return [
        any(
            p.get("article_id") == hop.expected.article_id
            and contains_excerpt(str(p.get("text", "")), hop.expected.excerpt)
            for p in passages
        )
        for hop in question.hops
    ]


async def single_pass(registry, llm_factory, tracing, question: str, count, session_id):
    with tracing.tracer.start_as_current_span(
        "single_pass", attributes={"langfuse.session.id": session_id}
    ):
        result = await registry.execute(SEARCH_TOOL, {"query": question[:300], "limit": 5})
        passages = list((result.data or {}).get("passages", [])) if result.ok else []
        rendered = {p["chunk_id"]: p for p in passages}
        response = await llm_factory().ainvoke(
            [
                SystemMessage(content=ANSWER_PROMPT),
                HumanMessage(content=f"Question : {question}\n\nPassages :\n{_render(rendered)}"),
            ],
            config={"callbacks": tracing.langchain_callbacks()},
        )
    return passages, str(response.content), sum(count(p.get("text", "")) for p in passages)


async def main(model: str) -> None:
    questions = load_jsonl(MultiHopQuestion, EVAL_DIR / "multi10.jsonl")
    deps = default_dependencies(settings)
    registry = build_registry(settings, deps)
    await deps.knowledge._embedder.warm_up()  # type: ignore[attr-defined]
    count = model_token_counter(settings.embedding_model)
    llm_factory = ollama_factory(base_url=settings.ollama_base_url, model=model)
    run_id = stamp()
    label = f"eval-multi10-{model.replace(':', '_')}-{run_id}"
    tracing = configure_tracing(
        environment="eval",
        langfuse_base_url=settings.langfuse_base_url,
        langfuse_public_key=settings.langfuse_public_key,
        langfuse_secret_key=settings.langfuse_secret_key,
        journal_path=JOURNAL_DIR / f"{label}.jsonl",
        pricing=CLOUD_EQUIVALENT,
    )
    searcher = MultiHopSearcher(registry, tracing, llm_factory, count, Budget())
    rows = []
    try:
        for q in questions:
            terms = key_terms(q.answer)
            started = time.perf_counter()
            sp_passages, sp_answer, sp_tokens = await single_pass(
                registry, llm_factory, tracing, q.question, count, f"{label}-single"
            )
            sp_time = time.perf_counter() - started
            started = time.perf_counter()
            mh = await searcher.search(q.question, session_id=f"{label}-multihop")
            mh_time = time.perf_counter() - started
            row = {
                "id": q.id,
                "question": q.question,
                "single": {
                    "hops_found": hops_found(q, sp_passages),
                    "context_tokens": sp_tokens,
                    "answer_match": answer_matches(sp_answer, terms),
                    "answer": sp_answer[:300],
                    "seconds": round(sp_time, 1),
                },
                "multihop": {
                    "hops_found": hops_found(q, mh.passages),
                    "sub_questions": [h.sub_question for h in mh.hops],
                    "searches": mh.searches,
                    "reformulations": sum(h.reformulated for h in mh.hops),
                    "stop_reason": mh.stop_reason,
                    "context_tokens": mh.context_tokens,
                    "answer_match": answer_matches(mh.answer, terms),
                    "answer": mh.answer[:300],
                    "tokens_in": mh.tokens_in,
                    "tokens_out": mh.tokens_out,
                    "seconds": round(mh_time, 1),
                },
                "terms": terms,
            }
            rows.append(row)
            s, m = row["single"], row["multihop"]
            print(
                f"{q.id}: un passage sauts={s['hops_found']} rép={s['answer_match']:.0%} | "
                f"multi sauts={m['hops_found']} recherches={m['searches']} "
                f"stop={m['stop_reason']} tokens={m['context_tokens']} rép={m['answer_match']:.0%}"
            )
            for sub in m["sub_questions"]:
                print(f"    ↳ {sub}")
        tracing.shutdown()
    finally:
        await deps.aclose()

    n = len(rows)
    RESULTS.with_suffix(".rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    def mean(values):
        values = list(values)
        return sum(values) / len(values)

    def hop_rate(side: dict[str, Any]) -> float:
        return sum(side["hops_found"]) / len(side["hops_found"])

    summary = {
        "single": {
            "all_hops_found": mean(all(r["single"]["hops_found"]) for r in rows),
            "hop_recall": mean(hop_rate(r["single"]) for r in rows),
            "answer_match": mean(r["single"]["answer_match"] for r in rows),
            "context_tokens": mean(r["single"]["context_tokens"] for r in rows),
            "seconds": mean(r["single"]["seconds"] for r in rows),
        },
        "multihop": {
            "all_hops_found": mean(all(r["multihop"]["hops_found"]) for r in rows),
            "hop_recall": mean(hop_rate(r["multihop"]) for r in rows),
            "answer_match": mean(r["multihop"]["answer_match"] for r in rows),
            "context_tokens": mean(r["multihop"]["context_tokens"] for r in rows),
            "searches": mean(r["multihop"]["searches"] for r in rows),
            "reformulations": sum(r["multihop"]["reformulations"] for r in rows),
            "stop_reasons": {
                k: sum(r["multihop"]["stop_reason"] == k for r in rows)
                for k in ("enough", "max_searches", "budget", "no_next_question")
            },
            "seconds": mean(r["multihop"]["seconds"] for r in rows),
        },
        "rows": rows,
        "session": label,
        "model": model,
    }
    header = (
        "| Méthode | Les 2 sauts trouvés | Rappel des sauts | Réponse (termes clés) "
        "| Recherches | Tokens de contexte | Temps |"
    )
    print(f"\n{header}\n|---|---|---|---|---|---|---|")
    s, m = summary["single"], summary["multihop"]
    print(
        f"| Un passage | {s['all_hops_found']:.0%} | {s['hop_recall']:.2f} | "
        f"{s['answer_match']:.0%} | 1 | {s['context_tokens']:.0f} | {s['seconds']:.1f} s |"
    )
    print(
        f"| Multi-sauts | {m['all_hops_found']:.0%} | {m['hop_recall']:.2f} | "
        f"{m['answer_match']:.0%} | {m['searches']:.1f} | {m['context_tokens']:.0f} "
        f"| {m['seconds']:.1f} s |"
    )
    print(f"Arrêts : {m['stop_reasons']} ; reformulations : {m['reformulations']} ({n} questions)")
    RESULTS.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"→ {RESULTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=settings.agent_model)
    asyncio.run(main(parser.parse_args().model))
