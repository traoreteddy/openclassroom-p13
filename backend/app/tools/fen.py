"""Type FEN partagé par tous les outils : validation + normalisation."""

from __future__ import annotations

from typing import Annotated

import chess
from pydantic import AfterValidator, Field

START_FEN = chess.STARTING_FEN

_STATUS_MESSAGES: dict[chess.Status, str] = {
    chess.STATUS_NO_WHITE_KING: "pas de roi blanc",
    chess.STATUS_NO_BLACK_KING: "pas de roi noir",
    chess.STATUS_TOO_MANY_KINGS: "trop de rois",
    chess.STATUS_TOO_MANY_WHITE_PAWNS: "plus de 8 pions blancs",
    chess.STATUS_TOO_MANY_BLACK_PAWNS: "plus de 8 pions noirs",
    chess.STATUS_PAWNS_ON_BACKRANK: "pion sur la 1re ou la 8e rangée",
    chess.STATUS_OPPOSITE_CHECK: "le camp qui n'a pas le trait est en échec",
    chess.STATUS_BAD_CASTLING_RIGHTS: "droits de roque incohérents",
    chess.STATUS_INVALID_EP_SQUARE: "case de prise en passant incohérente",
    chess.STATUS_TOO_MANY_CHECKERS: "trop de pièces donnent échec",
}


def validate_fen(value: str) -> str:
    """Valide une FEN et renvoie sa forme normalisée (6 champs, espaces simples).

    Lève ValueError avec un message en français, réutilisé tel quel dans l'erreur
    renvoyée au modèle.
    """
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError("FEN vide")
    try:
        board = chess.Board(normalized)
    except ValueError as exc:
        raise ValueError(
            f"FEN mal formée ({exc}). Format attendu : 6 champs séparés par des espaces, "
            f"par exemple « {START_FEN} »"
        ) from None

    status = board.status()
    if status != chess.STATUS_VALID:
        reasons = [msg for flag, msg in _STATUS_MESSAGES.items() if status & flag]
        raise ValueError(f"position illégale : {', '.join(reasons) or 'statut invalide'}")
    return board.fen()


Fen = Annotated[
    str,
    AfterValidator(validate_fen),
    Field(
        description="Position en notation FEN complète (6 champs).",
        examples=[START_FEN],
        min_length=1,
        max_length=100,
    ),
]
