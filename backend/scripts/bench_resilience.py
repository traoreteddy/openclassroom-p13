"""Mesure de l'effet des politiques de résilience face à une API à 30 % d'erreurs.

Un « tour d'agent » appelle 4 outils externes (théorie, évaluation, connaissances,
vidéos). Les API Lichess et YouTube échouent aléatoirement 30 % du temps
(503, délai dépassé, connexion coupée). On compare :

- avant : aucune politique (1 tentative, pas de disjoncteur) ;
- après : politiques du contrat (retry exponentiel + disjoncteur, par outil).

    uv run python -m scripts.bench_resilience
"""

from __future__ import annotations

import asyncio

from app.tools.base import ToolCall
from app.tools.resilience import ResiliencePolicy
from scripts.simulation import ITALIAN_FEN, FlakyApi, Latencies, make_registry

ERROR_RATE = 0.30
TURNS = 300
TURN = [
    ToolCall("get_opening_theory", {"fen": ITALIAN_FEN}),
    ToolCall("evaluate_position", {"fen": ITALIAN_FEN}),
    ToolCall("search_opening_knowledge", {"query": "idées de la partie italienne"}),
    ToolCall("find_opening_videos", {"query": "partie italienne explication"}),
]
FAST = Latencies(lichess=0.0, youtube=0.0, engine=0.0, vector=0.0)


async def run(policy_override: ResiliencePolicy | None) -> dict[str, float]:
    api = FlakyApi(FAST, error_rate=ERROR_RATE, seed=2026)
    registry, client = make_registry(api, FAST, policy_override=policy_override)
    external = {"get_opening_theory", "find_opening_videos"}
    ok_calls = total_calls = full_turns = 0
    try:
        for _ in range(TURNS):
            results = await registry.execute_many(TURN)
            flaky = [r for r in results if r.tool in external]
            ok_calls += sum(r.ok for r in flaky)
            total_calls += len(flaky)
            full_turns += all(r.ok for r in results)
    finally:
        await client.aclose()
    return {
        "call_success_pct": 100 * ok_calls / total_calls,
        "turn_success_pct": 100 * full_turns / TURNS,
        "http_requests": api.requests,
        "simulated_failures": api.failures,
    }


def main() -> None:
    before = asyncio.run(run(ResiliencePolicy.none(timeout_s=5)))
    after = asyncio.run(run(None))  # politiques du contrat

    print(f"API simulée : {ERROR_RATE:.0%} d'erreurs, {TURNS} tours de 4 outils\n")
    print("| Indicateur | Avant | Après |")
    print("|---|---|---|")
    print(
        f"| Appels réussis (Lichess + YouTube) | {before['call_success_pct']:.1f} % "
        f"| {after['call_success_pct']:.1f} % |"
    )
    print(
        f"| Tours complets (4 outils OK) | {before['turn_success_pct']:.1f} % "
        f"| {after['turn_success_pct']:.1f} % |"
    )
    print(f"| Requêtes HTTP envoyées | {before['http_requests']} | {after['http_requests']} |")


if __name__ == "__main__":
    main()
