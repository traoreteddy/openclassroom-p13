"""Outil `inspect_position` : identifie et décrit une position à partir de sa FEN."""

from __future__ import annotations

from typing import Literal

import chess
from pydantic import ConfigDict

from app.tools.base import SideEffect, ToolInput, ToolOutput, ToolSpec
from app.tools.fen import START_FEN, Fen
from app.tools.resilience import ResiliencePolicy

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
OPENING_MAX_FULLMOVE = 15

GameStatus = Literal["ongoing", "checkmate", "stalemate", "insufficient_material", "draw_by_rule"]

DESCRIPTION = """\
Identifie une position d'échecs à partir de sa FEN : FEN normalisée, camp au trait,
numéro du coup, coups légaux (notation SAN), échec, fin de partie, bilan matériel et
phase de jeu. Calcul local, instantané et sans appel réseau.

À utiliser : en premier, pour vérifier une FEN fournie par l'utilisateur ou savoir
quels coups sont légaux avant d'en proposer un.
Exemple : « Est-ce que Cf3 est légal ici ? » → inspect_position(fen=...)

Ne pas utiliser : pour savoir quel coup est le meilleur (→ evaluate_position) ou ce
que dit la théorie (→ get_opening_theory).
"""


class InspectPositionInput(ToolInput):
    model_config = ConfigDict(json_schema_extra={"examples": [{"fen": START_FEN}]})

    fen: Fen


class Material(ToolOutput):
    white: int
    black: int
    balance: int  # > 0 : avantage matériel blanc


class InspectPositionOutput(ToolOutput):
    fen: str
    side_to_move: Literal["white", "black"]
    fullmove_number: int
    legal_moves: list[str]
    is_check: bool
    status: GameStatus
    material: Material
    phase: Literal["opening", "middlegame_or_endgame"]


def inspect(fen: str) -> InspectPositionOutput:
    board = chess.Board(fen)
    white = _material(board, chess.WHITE)
    black = _material(board, chess.BLACK)
    return InspectPositionOutput(
        fen=board.fen(),
        side_to_move="white" if board.turn == chess.WHITE else "black",
        fullmove_number=board.fullmove_number,
        legal_moves=[board.san(move) for move in board.legal_moves],
        is_check=board.is_check(),
        status=_status(board),
        material=Material(white=white, black=black, balance=white - black),
        phase="opening"
        if board.fullmove_number <= OPENING_MAX_FULLMOVE
        else "middlegame_or_endgame",
    )


def _material(board: chess.Board, color: chess.Color) -> int:
    return sum(len(board.pieces(piece, color)) * value for piece, value in PIECE_VALUES.items())


def _status(board: chess.Board) -> GameStatus:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient_material"
    if board.can_claim_draw() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return "draw_by_rule"
    return "ongoing"


async def _handler(params: InspectPositionInput) -> InspectPositionOutput:
    return inspect(params.fen)


def build_inspect_position_tool() -> ToolSpec:
    return ToolSpec(
        name="inspect_position",
        description=DESCRIPTION,
        input_model=InspectPositionInput,
        output_model=InspectPositionOutput,
        handler=_handler,
        side_effect=SideEffect.NONE,
        idempotent=True,
        policy=ResiliencePolicy.none(timeout_s=1.0),
    )
