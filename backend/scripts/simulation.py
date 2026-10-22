"""Environnement simulé pour les mesures : API HTTP, moteur et base vectorielle factices.

Aucun appel réseau : les API Lichess et YouTube sont servies par un `httpx.MockTransport`
avec une latence et un taux d'erreurs réglables.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import random
from pathlib import Path
from typing import Any

import chess
import chess.engine
import httpx

from app.core.config import Settings
from app.tools.base import ToolRegistry
from app.tools.catalog import ToolDependencies, build_registry
from app.tools.knowledge import KnowledgeSearch, Passage
from app.tools.resilience import ResiliencePolicy

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
LICHESS_URL = "https://explorer.sim"
YOUTUBE_URL = "https://youtube.sim/v3"

ITALIAN_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"


def fixture(relative: str) -> Any:
    return json.loads((FIXTURES / relative).read_text(encoding="utf-8"))


@dataclasses.dataclass
class Latencies:
    """Latences simulées (secondes), ordres de grandeur observés pour chaque service."""

    lichess: float = 0.30
    youtube: float = 0.40
    engine: float = 1.00
    vector: float = 0.20


class FlakyApi:
    """Sert les fixtures Lichess / YouTube avec latence et pannes aléatoires.

    Les pannes simulées sont réparties entre HTTP 503, délai dépassé et connexion coupée.
    """

    def __init__(self, latencies: Latencies, error_rate: float = 0.0, seed: int = 13) -> None:
        self.latencies = latencies
        self.error_rate = error_rate
        self.random = random.Random(seed)
        self.requests = 0
        self.failures = 0
        self._lichess = fixture("lichess/masters_italian.json")
        self._youtube = fixture("youtube/search_italian.json")

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        is_youtube = request.url.host == "youtube.sim"
        await asyncio.sleep(self.latencies.youtube if is_youtube else self.latencies.lichess)
        if self.random.random() < self.error_rate:
            self.failures += 1
            kind = self.random.choice(["503", "timeout", "connect"])
            if kind == "timeout":
                raise httpx.ReadTimeout("simulé", request=request)
            if kind == "connect":
                raise httpx.ConnectError("simulé", request=request)
            return httpx.Response(503, request=request)
        return httpx.Response(200, json=self._youtube if is_youtube else self._lichess)


class SimAnalyzer:
    def __init__(self, latency: float) -> None:
        self.latency = latency

    async def analyse(
        self, board: chess.Board, *, depth: int, multipv: int
    ) -> list[chess.engine.InfoDict]:
        await asyncio.sleep(self.latency)
        moves = list(board.legal_moves)[:multipv]
        return [
            {
                "depth": depth,
                "score": chess.engine.PovScore(chess.engine.Cp(30 - 10 * i), chess.WHITE),
                "pv": [move],
            }
            for i, move in enumerate(moves)
        ]


class SimKnowledge:
    def __init__(self, latency: float) -> None:
        self.latency = latency

    async def embed(self, text: str) -> list[float]:
        return [0.0]

    async def search(
        self, vector: list[float], *, limit: int, opening: str | None
    ) -> list[Passage]:
        await asyncio.sleep(self.latency)
        return [
            Passage(
                text="Les Noirs développent le Fou en c5 et visent f2.",
                opening=opening or "Italian Game",
                section="Idées",
                source_url="https://fr.wikichess.org/wiki/Partie_italienne",
                score=0.8,
            )
        ][:limit]


async def no_sleep(_: float) -> None:
    """Les attentes de backoff ne sont pas réellement subies pendant la mesure."""


def make_registry(
    api: FlakyApi,
    latencies: Latencies,
    *,
    policy_override: ResiliencePolicy | None = None,
) -> tuple[ToolRegistry, httpx.AsyncClient]:
    settings = Settings(
        _env_file=None,
        lichess_explorer_url=LICHESS_URL,
        lichess_token="sim",
        youtube_api_url=YOUTUBE_URL,
        youtube_api_key="sim",
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(api))
    knowledge = SimKnowledge(latencies.vector)
    deps = ToolDependencies(
        http=client,
        analyzer=SimAnalyzer(latencies.engine),
        knowledge=KnowledgeSearch(knowledge, knowledge),
    )
    registry = build_registry(settings, deps, sleep=no_sleep)
    if policy_override is not None:
        specs = [
            dataclasses.replace(registry.get(name), policy=policy_override)
            for name in registry.names
        ]
        registry = ToolRegistry(specs, sleep=no_sleep)
    return registry, client
