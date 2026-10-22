"""Assemblage du catalogue d'outils exposé à l'agent.

Les dépendances (client HTTP, moteur, base vectorielle) sont injectées : l'application
fournit les vraies, les tests et les scripts de mesure fournissent des doubles.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.core.config import Settings
from app.tools.base import ToolRegistry
from app.tools.deep_analysis import DeepAnalysisJobs, build_deep_analysis_tools
from app.tools.errors import ToolUnavailableError
from app.tools.knowledge import (
    KnowledgeSearch,
    Passage,
    build_search_knowledge_tool,
)
from app.tools.lichess import LichessExplorer, build_opening_theory_tool
from app.tools.position import build_inspect_position_tool
from app.tools.resilience import Clock, Sleep
from app.tools.stockfish import Analyzer, StockfishAnalyzer, build_evaluate_position_tool
from app.tools.videos import YouTubeSearch, build_find_videos_tool


@dataclass
class ToolDependencies:
    http: httpx.AsyncClient
    analyzer: Analyzer
    knowledge: KnowledgeSearch

    async def aclose(self) -> None:
        """Libère les ressources : connexions HTTP et processus Stockfish."""
        await self.http.aclose()
        close = getattr(self.analyzer, "close", None)
        if close is not None:
            await close()


class _KnowledgeNotConfigured:
    """Base vectorielle pas encore branchée (étape 2) : l'outil répond « indisponible »."""

    async def embed(self, text: str) -> list[float]:
        raise ToolUnavailableError(
            code="not_configured",
            message="La base de connaissances n'est pas encore disponible.",
            hint="Répondre avec get_opening_theory et evaluate_position uniquement.",
        )

    async def search(
        self, vector: list[float], *, limit: int, opening: str | None
    ) -> list[Passage]:  # pragma: no cover - jamais atteint, embed échoue avant
        return []


def default_dependencies(settings: Settings) -> ToolDependencies:
    placeholder = _KnowledgeNotConfigured()
    return ToolDependencies(
        http=httpx.AsyncClient(timeout=settings.http_timeout_s),
        analyzer=StockfishAnalyzer(
            settings.stockfish_path,
            threads=settings.stockfish_threads,
            hash_mb=settings.stockfish_hash_mb,
        ),
        knowledge=KnowledgeSearch(placeholder, placeholder),
    )


def build_registry(
    settings: Settings,
    deps: ToolDependencies,
    *,
    sleep: Sleep | None = None,
    clock: Clock | None = None,
) -> ToolRegistry:
    explorer = LichessExplorer(
        deps.http,
        base_url=settings.lichess_explorer_url,
        token=settings.lichess_token,
        min_games=settings.theory_min_games,
    )
    youtube = YouTubeSearch(
        deps.http, api_key=settings.youtube_api_key, base_url=settings.youtube_api_url
    )
    jobs = DeepAnalysisJobs(deps.analyzer, **({"clock": clock} if clock else {}))

    specs = [
        build_inspect_position_tool(),
        build_opening_theory_tool(explorer),
        build_evaluate_position_tool(deps.analyzer),
        build_search_knowledge_tool(deps.knowledge),
        build_find_videos_tool(youtube),
        *build_deep_analysis_tools(jobs),
    ]
    options = {k: v for k, v in {"sleep": sleep, "clock": clock}.items() if v is not None}
    return ToolRegistry(specs, **options)
