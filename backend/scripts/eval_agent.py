"""Phase 5 : mesures de l'agent (sélection d'outil, retrieval décidé ou à flot fixe).

- `--tasks`     : Tâches-30, taux d'appels d'outil corrects, pour chaque configuration
                  (modèle x mode de sélection). Seule la première décision est jouée.
- `--decisions` : Décision-50, taux d'appels inutiles de la recherche (et d'oublis),
                  flot fixe contre décision de l'agent, avec le modèle retenu.

Chaque exécution est une session Langfuse (`eval-<jeu>-<config>-<horodatage>`) et un
fichier de journal de décision (`data/journal/`).

    uv run python -m scripts.eval_agent --tasks
    uv run python -m scripts.eval_agent --decisions --model qwen3:14b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.agent.graph import SEARCH_TOOL, AgentConfig, ChessAgent
from app.agent.journal import Pricing
from app.agent.llm import ollama_factory
from app.agent.tracing import Tracing, configure_tracing
from app.core.config import settings
from app.rag.eval.datasets import EVAL_DIR, DecisionQuestion, ToolTask, load_jsonl
from app.rag.text import normalize_text
from app.tools.catalog import build_registry, default_dependencies

RESULTS_DIR = EVAL_DIR / "results"
JOURNAL_DIR = Path(__file__).resolve().parents[1] / "data" / "journal"

# Hypothèse de coût « équivalent cloud » pour un petit modèle hébergé (€ par million de
# tokens) ; le coût réel des modèles locaux est 0. Sert seulement à donner un ordre de
# grandeur dans le journal et la doc.
CLOUD_EQUIVALENT = Pricing(input_per_million=0.14, output_per_million=0.55)

CONFIGS: dict[str, tuple[str, str]] = {
    "phi4-mini/structured": ("phi4-mini", "structured"),
    "qwen3:1.7b/native": ("qwen3:1.7b", "native"),
    "qwen3:14b/native": ("qwen3:14b", "native"),
}


def stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")


def make_tracing(label: str) -> Tracing:
    return configure_tracing(
        environment="eval",
        langfuse_base_url=settings.langfuse_base_url,
        langfuse_public_key=settings.langfuse_public_key,
        langfuse_secret_key=settings.langfuse_secret_key,
        journal_path=JOURNAL_DIR / f"{label}.jsonl",
        pricing=CLOUD_EQUIVALENT,
    )


def arguments_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    for key, value in expected.items():
        got = actual.get(key)
        if key == "fen":
            if normalize_text(str(got or "")) != normalize_text(str(value)):
                return False
        elif got != value:
            return False
    return True


async def eval_tasks(configs: list[str]) -> None:
    tasks = load_jsonl(ToolTask, EVAL_DIR / "tasks30.jsonl")
    deps = default_dependencies(settings)
    registry = build_registry(settings, deps)
    run_id = stamp()
    summary: dict[str, Any] = {}
    try:
        for name in configs:
            model, selection = CONFIGS[name]
            label = f"eval-tasks30-{name.replace('/', '-').replace(':', '_')}-{run_id}"
            tracing = make_tracing(label)
            agent = ChessAgent(
                registry,
                tracing,
                ollama_factory(base_url=settings.ollama_base_url, model=model),
                AgentConfig(selection=selection),  # type: ignore[arg-type]
            )
            correct = args_ok = 0
            confusion: Counter[tuple[str, str]] = Counter()
            latencies: list[float] = []
            tokens = [0, 0]
            rows = []
            for task in tasks:
                started = time.perf_counter()
                decision = await agent.first_decision(task.request, session_id=label)
                latency = time.perf_counter() - started
                latencies.append(latency)
                tokens[0] += decision.tokens_in
                tokens[1] += decision.tokens_out
                tool_ok = decision.tool == task.expected_tool
                full_ok = tool_ok and arguments_match(task.expected_arguments, decision.arguments)
                correct += tool_ok
                args_ok += full_ok
                confusion[(task.expected_tool or "aucun", decision.tool or "aucun")] += 1
                rows.append(
                    {
                        "task": task.id,
                        "expected": task.expected_tool,
                        "got": decision.tool,
                        "arguments": decision.arguments,
                        "tool_ok": tool_ok,
                        "arguments_ok": full_ok,
                        "latency_s": round(latency, 2),
                    }
                )
                mark = "✓" if full_ok else ("~" if tool_ok else "✗")
                print(f"  {mark} {task.id} attendu={task.expected_tool} obtenu={decision.tool}")
            tracing.shutdown()
            errors = {f"{e} → {g}": n for (e, g), n in confusion.items() if e != g}
            summary[name] = {
                "model": model,
                "selection": selection,
                "tool_correct": correct / len(tasks),
                "tool_and_arguments_correct": args_ok / len(tasks),
                "mean_latency_s": round(sum(latencies) / len(latencies), 2),
                "tokens_in": tokens[0],
                "tokens_out": tokens[1],
                "cloud_equivalent_eur": CLOUD_EQUIVALENT.cost(*tokens),
                "errors": errors,
                "rows": rows,
                "session": label,
            }
            print(
                f"\n{name}: outil correct {correct}/{len(tasks)}, outil + arguments "
                f"{args_ok}/{len(tasks)}, {summary[name]['mean_latency_s']} s/décision\n"
            )
    finally:
        await deps.aclose()

    print("| Configuration | Outil correct | Outil + arguments | Latence moy. | Tokens (in/out) |")
    print("|---|---|---|---|---|")
    for name, r in summary.items():
        print(
            f"| {name} | {r['tool_correct']:.0%} | {r['tool_and_arguments_correct']:.0%} "
            f"| {r['mean_latency_s']} s | {r['tokens_in']} / {r['tokens_out']} |"
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "phase5-tool-selection.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n→ {path}")


async def eval_decisions(model: str, selection: str) -> None:
    questions = load_jsonl(DecisionQuestion, EVAL_DIR / "decision50.jsonl")
    deps = default_dependencies(settings)
    registry = build_registry(settings, deps)
    embedder = getattr(deps.knowledge, "_embedder", None)
    if embedder is not None and hasattr(embedder, "warm_up"):
        await embedder.warm_up()
    run_id = stamp()
    summary: dict[str, Any] = {}
    try:
        for mode in ("fixed", "agent"):
            label = f"eval-decision50-{mode}-{model.replace(':', '_')}-{run_id}"
            tracing = make_tracing(label)
            agent = ChessAgent(
                registry,
                tracing,
                ollama_factory(base_url=settings.ollama_base_url, model=model),
                AgentConfig(retrieval=mode, selection=selection),  # type: ignore[arg-type]
            )
            by_category: dict[str, Counter] = defaultdict(Counter)
            rows = []
            latencies = []
            tokens = [0, 0]
            for item in questions:
                started = time.perf_counter()
                result = await agent.ask(item.question, session_id=label)
                latencies.append(time.perf_counter() - started)
                tokens[0] += result.tokens_in
                tokens[1] += result.tokens_out
                searched = SEARCH_TOOL in result.tools_used
                useless = searched and not item.search_useful
                missed = item.search_useful and not searched
                by_category[item.category]["total"] += 1
                by_category[item.category]["searched"] += searched
                by_category[item.category]["useless"] += useless
                by_category[item.category]["missed"] += missed
                rows.append(
                    {
                        "id": item.id,
                        "category": item.category,
                        "searched": searched,
                        "tools": result.tools_used,
                        "useless": useless,
                        "missed": missed,
                    }
                )
                mark = "✗" if (useless or missed) else "✓"
                print(f"  {mark} {item.id} [{item.category}] outils={result.tools_used}")
            tracing.shutdown()
            not_useful = [q for q in questions if not q.search_useful]
            useful = [q for q in questions if q.search_useful]
            useless_total = sum(r["useless"] for r in rows)
            missed_total = sum(r["missed"] for r in rows)
            summary[mode] = {
                "useless_rate": useless_total / len(not_useful),
                "missed_rate": missed_total / len(useful),
                "by_category": {
                    cat: {
                        "total": c["total"],
                        "searched": c["searched"],
                        "useless_rate": (c["useless"] / c["total"])
                        if cat != "dans_corpus"
                        else None,
                        "missed_rate": (c["missed"] / c["total"]) if cat == "dans_corpus" else None,
                    }
                    for cat, c in by_category.items()
                },
                "mean_latency_s": round(sum(latencies) / len(latencies), 2),
                "tokens_in": tokens[0],
                "tokens_out": tokens[1],
                "cloud_equivalent_eur": CLOUD_EQUIVALENT.cost(*tokens),
                "rows": rows,
                "session": label,
            }
            print(
                f"\n{mode}: appels inutiles {useless_total}/{len(not_useful)}, "
                f"recherches oubliées {missed_total}/{len(useful)}\n"
            )
    finally:
        await deps.aclose()

    print("| Mode | Appels inutiles (20) | Recherches oubliées (30) | Latence moy. | Tokens |")
    print("|---|---|---|---|---|")
    for mode, r in summary.items():
        print(
            f"| {mode} | {r['useless_rate']:.0%} | {r['missed_rate']:.0%} "
            f"| {r['mean_latency_s']} s | {r['tokens_in']} / {r['tokens_out']} |"
        )
    path = RESULTS_DIR / "phase5-decision.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n→ {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", action="store_true")
    parser.add_argument("--decisions", action="store_true")
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS))
    parser.add_argument("--model", default=settings.agent_model)
    parser.add_argument("--selection", default=settings.agent_selection)
    args = parser.parse_args()
    if args.tasks:
        asyncio.run(eval_tasks(args.configs))
    if args.decisions:
        asyncio.run(eval_decisions(args.model, args.selection))
