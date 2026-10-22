"""Fixtures communes : aucun test n'appelle le réseau ni un vrai Stockfish."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chess
import chess.engine
import httpx
import pytest

from app.core.config import Settings
from app.tools.base import ToolRegistry
from app.tools.catalog import ToolDependencies, build_registry
from app.tools.knowledge import KnowledgeSearch, Passage

FIXTURES = Path(__file__).parent / "fixtures"

LICHESS_URL = "https://explorer.test"
YOUTUBE_URL = "https://youtube.test/v3"

START_FEN = chess.STARTING_FEN
ITALIAN_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"


@pytest.fixture(autouse=True)
def _no_network(respx_mock: Any) -> None:
    """Garde-fou : toute requête HTTP non simulée par respx fait échouer le test.

    Garantit qu'aucun test n'appelle une API réelle (et facturée).
    """


def load_fixture(relative: str) -> Any:
    """Charge une réponse d'API figée depuis tests/fixtures/."""
    return json.loads((FIXTURES / relative).read_text(encoding="utf-8"))


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        lichess_explorer_url=LICHESS_URL,
        lichess_token="test-token",
        theory_min_games=20,
        youtube_api_url=YOUTUBE_URL,
        youtube_api_key="test-key",
    )


class FakeClock:
    """Horloge contrôlée par le test (disjoncteur, durée des tâches)."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


class RecordingSleep:
    """Remplace asyncio.sleep : les retries sont instantanés, les attentes sont notées."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture
def sleep() -> RecordingSleep:
    return RecordingSleep()


class StubAnalyzer:
    """Stub du moteur : renvoie des InfoDict prévus, ou lève l'exception prévue.

    `scores` : une entrée par variante, en centipions (int) ou ("mate", n),
    du point de vue des Blancs.
    """

    def __init__(
        self,
        scores: list[int | tuple[str, int]] | None = None,
        *,
        depth: int = 16,
        error: Exception | None = None,
    ) -> None:
        self.scores = scores if scores is not None else [25]
        self.depth = depth
        self.error = error
        self.calls: list[tuple[str, int, int]] = []

    async def analyse(
        self, board: chess.Board, *, depth: int, multipv: int
    ) -> list[chess.engine.InfoDict]:
        self.calls.append((board.fen(), depth, multipv))
        if self.error is not None:
            raise self.error
        moves = list(board.legal_moves)
        infos: list[chess.engine.InfoDict] = []
        for index, raw in enumerate(self.scores[:multipv]):
            score: chess.engine.Score = (
                chess.engine.Mate(raw[1]) if isinstance(raw, tuple) else chess.engine.Cp(raw)
            )
            first = moves[index % len(moves)]
            after = board.copy(stack=False)
            after.push(first)
            pv = [first, next(iter(after.legal_moves))] if not after.is_game_over() else [first]
            infos.append(
                {
                    "depth": self.depth,
                    "multipv": index + 1,
                    "score": chess.engine.PovScore(score, chess.WHITE),
                    "pv": pv,
                }
            )
        return infos


class StubEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2, 0.3]


class StubVectorStore:
    def __init__(self, passages: list[Passage] | None = None, error: Exception | None = None):
        self.passages = passages or []
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def search(
        self, vector: list[float], *, limit: int, opening: str | None
    ) -> list[Passage]:
        self.calls.append({"limit": limit, "opening": opening})
        if self.error is not None:
            raise self.error
        return [p for p in self.passages if opening is None or p.opening == opening]


def passage(text: str, score: float, opening: str = "Sicilian Defense") -> Passage:
    return Passage(
        chunk_id=f"recursive:{opening.lower().replace(' ', '-')}:000",
        article_id=opening.lower().replace(" ", "-"),
        title=opening,
        opening=opening,
        opening_family=opening,
        section="Idées",
        text=text,
        source_url="https://fr.wikipedia.org/wiki/Défense_sicilienne",
        score=score,
    )


@pytest.fixture
async def http_client() -> Any:
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture
def make_registry(
    settings: Settings,
    http_client: httpx.AsyncClient,
    sleep: RecordingSleep,
    clock: FakeClock,
) -> Callable[..., ToolRegistry]:
    """Fabrique un catalogue complet branché sur des doubles de test."""

    def factory(
        analyzer: StubAnalyzer | None = None,
        store: StubVectorStore | None = None,
        **overrides: Any,
    ) -> ToolRegistry:
        deps = ToolDependencies(
            http=http_client,
            analyzer=analyzer or StubAnalyzer(),
            knowledge=KnowledgeSearch(StubEmbedder(), store or StubVectorStore()),
        )
        return build_registry(settings.model_copy(update=overrides), deps, sleep=sleep, clock=clock)

    return factory
