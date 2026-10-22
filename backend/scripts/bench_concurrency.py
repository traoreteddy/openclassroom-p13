"""Mesure du gain de latence de l'exécution concurrente (asyncio.gather).

Huit appels d'outils indépendants, latences simulées réalistes, aucun réseau :
on compare l'exécution séquentielle (un `await` après l'autre) et concurrente.

    uv run python -m scripts.bench_concurrency
"""

from __future__ import annotations

import asyncio
import time

from app.tools.base import ToolCall
from scripts.simulation import ITALIAN_FEN, FlakyApi, Latencies, make_registry

CALLS = [
    ToolCall("inspect_position", {"fen": ITALIAN_FEN}),
    ToolCall("get_opening_theory", {"fen": ITALIAN_FEN}),
    ToolCall("get_opening_theory", {"fen": ITALIAN_FEN, "database": "lichess"}),
    ToolCall("get_opening_theory", {"fen": ITALIAN_FEN, "include_games": True}),
    ToolCall("evaluate_position", {"fen": ITALIAN_FEN}),
    ToolCall("search_opening_knowledge", {"query": "idées de la partie italienne"}),
    ToolCall("search_opening_knowledge", {"query": "pièges de l'italienne pour les Noirs"}),
    ToolCall("find_opening_videos", {"query": "partie italienne explication"}),
]
RUNS = 3


async def measure() -> dict[str, float]:
    latencies = Latencies()
    registry, client = make_registry(FlakyApi(latencies), latencies)
    try:
        sequential, concurrent = [], []
        for _ in range(RUNS):
            start = time.perf_counter()
            for call in CALLS:
                result = await registry.execute(call.name, call.arguments)
                assert result.ok, result.error
            sequential.append(time.perf_counter() - start)

            start = time.perf_counter()
            results = await registry.execute_many(CALLS)
            assert all(r.ok for r in results)
            concurrent.append(time.perf_counter() - start)
    finally:
        await client.aclose()

    seq, conc = min(sequential), min(concurrent)
    return {
        "calls": len(CALLS),
        "sequential_s": seq,
        "concurrent_s": conc,
        "gain_pct": 100 * (seq - conc) / seq,
        "speedup": seq / conc,
    }


def main() -> None:
    r = asyncio.run(measure())
    print(f"| Mode | Durée totale ({r['calls']} appels) |")
    print("|---|---|")
    print(f"| Séquentiel | {r['sequential_s']:.2f} s |")
    print(f"| Concurrent (asyncio.gather) | {r['concurrent_s']:.2f} s |")
    print(f"\nGain de latence : {r['gain_pct']:.0f} % (x{r['speedup']:.1f} plus rapide)")


if __name__ == "__main__":
    main()
