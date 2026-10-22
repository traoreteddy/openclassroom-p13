"""Outil `evaluate_position` : évaluation rapide d'une position par Stockfish."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Protocol

import chess
import chess.engine
from pydantic import ConfigDict, Field

from app.tools.base import SideEffect, ToolInput, ToolOutput, ToolSpec
from app.tools.errors import ToolTransientError, ToolUnavailableError
from app.tools.fen import Fen
from app.tools.resilience import ResiliencePolicy

SICILIAN_FEN = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"
MAX_PV_PLIES = 8

DESCRIPTION = """\
Évalue une position avec le moteur Stockfish : score en centipions du point de vue des
Blancs (ou mat en N), appréciation en français (« léger avantage aux Blancs »…) et
meilleurs coups avec leur suite principale. Réponse en 1 à 2 secondes.

À utiliser : pour dire qui est mieux et de combien, conseiller un coup quand la position
est sortie de la théorie, ou juger un coup du joueur.
Exemple : « Qui est mieux ici ? » → evaluate_position(fen=...)
Exemple : « Donne-moi les 3 meilleurs coups » → evaluate_position(fen=..., lines=3)

Ne pas utiliser : pour nommer l'ouverture ou connaître les coups des maîtres
(→ get_opening_theory), ni pour une analyse très profonde (→ start_deep_analysis).
"""


class EvaluatePositionInput(ToolInput):
    model_config = ConfigDict(json_schema_extra={"examples": [{"fen": SICILIAN_FEN, "lines": 1}]})

    fen: Fen
    depth: int = Field(default=16, ge=8, le=22, description="Profondeur de recherche.")
    lines: int = Field(default=1, ge=1, le=3, description="Nombre de variantes (MultiPV).")


class EngineLine(ToolOutput):
    rank: int
    score_cp: int | None
    mate: int | None
    move_san: str
    move_uci: str
    pv_san: list[str]


class EvaluationOutput(ToolOutput):
    fen: str
    depth: int
    score_cp: int | None
    mate: int | None
    assessment: str
    lines: list[EngineLine]


class Analyzer(Protocol):
    """Ce dont l'outil a besoin d'un moteur. Permet de le remplacer par un stub."""

    async def analyse(
        self, board: chess.Board, *, depth: int, multipv: int
    ) -> list[chess.engine.InfoDict]: ...


class StockfishAnalyzer:
    """Moteur Stockfish réel, démarré à la première analyse et redémarré après un crash.

    Un moteur UCI ne traite qu'une analyse à la fois : un verrou sérialise les appels.
    """

    def __init__(self, path: str = "stockfish", *, threads: int = 1, hash_mb: int = 64) -> None:
        self._path = path
        self._options = {"Threads": threads, "Hash": hash_mb}
        self._engine: chess.engine.UciProtocol | None = None
        self._lock = asyncio.Lock()
        self._background: set[asyncio.Task[None]] = set()

    async def analyse(
        self, board: chess.Board, *, depth: int, multipv: int
    ) -> list[chess.engine.InfoDict]:
        async with self._lock:
            engine = await self._ensure_engine()
            try:
                return await engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
            except chess.engine.EngineError as exc:  # inclut EngineTerminatedError
                await self.close()
                raise ToolTransientError(
                    code="engine_crashed",
                    message="Le moteur Stockfish s'est arrêté pendant l'analyse.",
                ) from exc
            except asyncio.CancelledError:
                # Timeout : l'analyse interrompue laisse le moteur dans un état incertain,
                # on le jette (arrêt en arrière-plan) et on en relancera un au prochain appel.
                self._discard_engine()
                raise

    async def close(self) -> None:
        if self._engine is not None:
            engine, self._engine = self._engine, None
            await _quit_quietly(engine)

    def _discard_engine(self) -> None:
        if self._engine is not None:
            engine, self._engine = self._engine, None
            task = asyncio.get_running_loop().create_task(_quit_quietly(engine))
            self._background.add(task)  # garder une référence jusqu'à la fin de la tâche
            task.add_done_callback(self._background.discard)

    async def _ensure_engine(self) -> chess.engine.UciProtocol:
        if self._engine is None:
            try:
                _, engine = await chess.engine.popen_uci(self._path)
            except FileNotFoundError as exc:
                raise ToolUnavailableError(
                    code="engine_not_found",
                    message=f"Stockfish est introuvable ({self._path}).",
                    hint="Installer Stockfish ou renseigner STOCKFISH_PATH.",
                ) from exc
            await engine.configure(self._options)
            self._engine = engine
        return self._engine


async def _quit_quietly(engine: chess.engine.UciProtocol) -> None:
    with contextlib.suppress(chess.engine.EngineError, OSError):
        await engine.quit()


async def evaluate(analyzer: Analyzer, fen: str, *, depth: int, lines: int) -> EvaluationOutput:
    """Évalue `fen` ; partagé par l'outil rapide et l'analyse différée."""
    board = chess.Board(fen)
    if board.is_game_over():
        return EvaluationOutput(
            fen=fen,
            depth=0,
            score_cp=None,
            mate=None,
            assessment=_game_over_label(board),
            lines=[],
        )

    infos = await analyzer.analyse(board, depth=depth, multipv=lines)
    parsed = [
        _line(board, rank, info) for rank, info in enumerate(infos, start=1) if info.get("pv")
    ]
    best = parsed[0] if parsed else None
    return EvaluationOutput(
        fen=fen,
        depth=max((info.get("depth", 0) for info in infos), default=0),
        score_cp=best.score_cp if best else None,
        mate=best.mate if best else None,
        assessment=assess(best.score_cp, best.mate) if best else "Évaluation indisponible",
        lines=parsed,
    )


def assess(score_cp: int | None, mate: int | None) -> str:
    """Appréciation lisible par un jeune joueur, point de vue des Blancs."""
    if mate is not None:
        side = "les Blancs" if mate > 0 else "les Noirs"
        return f"Mat en {abs(mate)} pour {side}"
    cp = score_cp or 0
    side = "aux Blancs" if cp > 0 else "aux Noirs"
    magnitude = abs(cp)
    if magnitude <= 30:
        return "Position équilibrée"
    if magnitude <= 100:
        return f"Léger avantage {side}"
    if magnitude <= 300:
        return f"Net avantage {side}"
    return f"Avantage décisif {side}"


def _line(board: chess.Board, rank: int, info: chess.engine.InfoDict) -> EngineLine:
    pv = info["pv"][:MAX_PV_PLIES]
    score = info["score"].white()
    return EngineLine(
        rank=rank,
        score_cp=score.score(),
        mate=score.mate(),
        move_san=board.san(pv[0]),
        move_uci=pv[0].uci(),
        pv_san=_pv_to_san(board, pv),
    )


def _pv_to_san(board: chess.Board, pv: list[chess.Move]) -> list[str]:
    copy = board.copy(stack=False)
    sans: list[str] = []
    for move in pv:
        sans.append(copy.san(move))
        copy.push(move)
    return sans


def _game_over_label(board: chess.Board) -> str:
    if board.is_checkmate():
        return "Partie terminée : échec et mat"
    if board.is_stalemate():
        return "Partie terminée : pat"
    return "Partie terminée : nulle"


def build_evaluate_position_tool(analyzer: Analyzer) -> ToolSpec:
    async def handler(params: EvaluatePositionInput) -> EvaluationOutput:
        return await evaluate(analyzer, params.fen, depth=params.depth, lines=params.lines)

    return ToolSpec(
        name="evaluate_position",
        description=DESCRIPTION,
        input_model=EvaluatePositionInput,
        output_model=EvaluationOutput,
        handler=handler,
        side_effect=SideEffect.NONE,
        idempotent=True,
        policy=ResiliencePolicy(
            timeout_s=10.0, max_attempts=2, breaker_threshold=3, breaker_reset_s=30.0
        ),
    )
