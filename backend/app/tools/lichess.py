"""Outil `get_opening_theory` : coups théoriques et parties de référence (Lichess Explorer)."""

from __future__ import annotations

from typing import Any, Literal

import httpx
from pydantic import ConfigDict, Field

from app.tools.base import SideEffect, ToolInput, ToolOutput, ToolSpec
from app.tools.fen import Fen
from app.tools.http import get_json
from app.tools.resilience import ResiliencePolicy

SERVICE = "Lichess Explorer"
ITALIAN_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
LICHESS_DB_SPEEDS = "blitz,rapid,classical"
LICHESS_DB_RATINGS = "1600,1800,2000,2200,2500"

DESCRIPTION = """\
Renvoie ce que dit la théorie des ouvertures pour une position : nom et code ECO de
l'ouverture, coups joués par les joueurs (avec nombre de parties et % de victoires) et,
en option, des parties de référence. Indique aussi si la position est encore « en
théorie » (assez de parties connues).

À utiliser : pour proposer les meilleurs coups théoriques, nommer l'ouverture, ou
savoir si la partie est sortie des sentiers battus.
Exemple : « Que jouer après 1.e4 e5 2.Cf3 Cc6 3.Fc4 ? » → get_opening_theory(fen=...)
Exemple : « Montre-moi des parties de maîtres » → get_opening_theory(fen=..., include_games=true)

Ne pas utiliser : pour une évaluation chiffrée de la position (→ evaluate_position),
ni pour expliquer les idées de l'ouverture (→ search_opening_knowledge).
Si in_theory est false, appeler evaluate_position pour conseiller un coup.
"""


class OpeningTheoryInput(ToolInput):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"fen": ITALIAN_FEN, "database": "masters"}]}
    )

    fen: Fen
    database: Literal["masters", "lichess"] = Field(
        default="masters",
        description="masters : parties de maîtres (OTB, 2200+). "
        "lichess : parties en ligne 1600-2500, plus nombreuses.",
    )
    max_moves: int = Field(default=5, ge=1, le=12, description="Nombre de coups renvoyés.")
    include_games: bool = Field(default=False, description="Ajouter des parties de référence.")


class Opening(ToolOutput):
    eco: str
    name: str


class TheoryMove(ToolOutput):
    san: str
    uci: str
    games: int
    white_pct: float
    draw_pct: float
    black_pct: float
    average_rating: int | None = None


class ReferenceGame(ToolOutput):
    id: str
    white: str
    black: str
    white_rating: int | None = None
    black_rating: int | None = None
    year: int | None = None
    winner: Literal["white", "black", "draw"]
    url: str


class OpeningTheoryOutput(ToolOutput):
    fen: str
    database: Literal["masters", "lichess"]
    opening: Opening | None
    total_games: int
    in_theory: bool
    moves: list[TheoryMove]
    reference_games: list[ReferenceGame]


class LichessExplorer:
    """Client de l'API Opening Explorer (https://lichess.org/api#tag/Opening-Explorer)."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str = "https://explorer.lichess.ovh",
        token: str | None = None,
        min_games: int = 20,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._token = token
        self.min_games = min_games

    async def theory(self, params: OpeningTheoryInput) -> OpeningTheoryOutput:
        query: dict[str, Any] = {
            "fen": params.fen,
            "moves": params.max_moves,
            "topGames": 4 if params.include_games else 0,
        }
        if params.database == "lichess":
            query |= {
                "variant": "standard",
                "speeds": LICHESS_DB_SPEEDS,
                "ratings": LICHESS_DB_RATINGS,
                "recentGames": 0,
            }
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else None
        payload = await get_json(
            self._client,
            f"{self._base_url}/{params.database}",
            service=SERVICE,
            params=query,
            headers=headers,
        )
        return self._parse(payload, params)

    def _parse(self, payload: dict[str, Any], params: OpeningTheoryInput) -> OpeningTheoryOutput:
        total = payload.get("white", 0) + payload.get("draws", 0) + payload.get("black", 0)
        moves = [_move(m) for m in payload.get("moves", [])][: params.max_moves]
        games = [_game(g) for g in payload.get("topGames", [])] if params.include_games else []
        opening = payload.get("opening")
        return OpeningTheoryOutput(
            fen=params.fen,
            database=params.database,
            opening=Opening(eco=opening["eco"], name=opening["name"]) if opening else None,
            total_games=total,
            in_theory=bool(moves) and total >= self.min_games,
            moves=moves,
            reference_games=games,
        )


def _pct(part: int, total: int) -> float:
    return round(100 * part / total, 1) if total else 0.0


def _move(raw: dict[str, Any]) -> TheoryMove:
    games = raw["white"] + raw["draws"] + raw["black"]
    return TheoryMove(
        san=raw["san"],
        uci=raw["uci"],
        games=games,
        white_pct=_pct(raw["white"], games),
        draw_pct=_pct(raw["draws"], games),
        black_pct=_pct(raw["black"], games),
        average_rating=raw.get("averageRating"),
    )


def _game(raw: dict[str, Any]) -> ReferenceGame:
    return ReferenceGame(
        id=raw["id"],
        white=raw["white"]["name"],
        black=raw["black"]["name"],
        white_rating=raw["white"].get("rating"),
        black_rating=raw["black"].get("rating"),
        year=raw.get("year"),
        winner=raw.get("winner") or "draw",
        url=f"https://lichess.org/{raw['id']}",
    )


def build_opening_theory_tool(explorer: LichessExplorer) -> ToolSpec:
    return ToolSpec(
        name="get_opening_theory",
        description=DESCRIPTION,
        input_model=OpeningTheoryInput,
        output_model=OpeningTheoryOutput,
        handler=explorer.theory,
        side_effect=SideEffect.READ_EXTERNAL,
        idempotent=True,
        policy=ResiliencePolicy(timeout_s=5.0, max_attempts=3, breaker_threshold=5),
    )
