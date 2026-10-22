"""Outil inspect_position et validation FEN commune (cas limites paramétrés)."""

from __future__ import annotations

import pytest

from app.tools.base import ToolRegistry
from app.tools.fen import validate_fen
from app.tools.position import build_inspect_position_tool
from tests.conftest import ITALIAN_FEN, START_FEN


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry([build_inspect_position_tool()])


@pytest.mark.parametrize(
    ("fen", "fragment"),
    [
        ("", "FEN vide"),
        ("   ", "FEN vide"),
        ("pas une fen", "FEN mal formée"),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1", "FEN mal formée"),
        ("8/8/8/8/8/8/8/K7 w - - 0 1", "pas de roi noir"),
        ("k7/8/8/8/8/8/8/8 w - - 0 1", "pas de roi blanc"),
        ("kK6/8/8/8/8/8/8/K7 w - - 0 1", "trop de rois"),
        ("k7/8/8/8/8/8/PPPPPPPP/KP6 w - - 0 1", "plus de 8 pions blancs"),
        ("kp6/pppppppp/8/8/8/8/8/K7 w - - 0 1", "plus de 8 pions noirs"),
        ("k6P/8/8/8/8/8/8/K7 w - - 0 1", "pion sur la 1re ou la 8e rangée"),
        ("k7/8/8/8/8/8/8/K6r b - - 0 1", "le camp qui n'a pas le trait est en échec"),
        ("k7/8/8/8/8/8/8/K7 w KQ - 0 1", "droits de roque incohérents"),
        ("k7/8/8/8/8/8/8/K7 w - e3 0 1", "case de prise en passant incohérente"),
        ("k7/8/8/8/8/1n6/1q6/K1r5 w - - 0 1", "trop de pièces donnent échec"),
    ],
)
def test_invalid_fens_are_rejected_with_reason(fen, fragment):
    with pytest.raises(ValueError, match=fragment):
        validate_fen(fen)


def test_fen_is_normalized():
    messy = "  rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR   w KQkq -   0 1 "

    assert validate_fen(messy) == START_FEN


def test_fen_without_move_counters_is_completed():
    assert validate_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -") == START_FEN


async def test_start_position(registry):
    result = await registry.execute("inspect_position", {"fen": START_FEN})

    data = result.data
    assert data["side_to_move"] == "white"
    assert data["fullmove_number"] == 1
    assert len(data["legal_moves"]) == 20
    assert "Nf3" in data["legal_moves"]
    assert data["status"] == "ongoing"
    assert data["material"] == {"white": 39, "black": 39, "balance": 0}
    assert data["phase"] == "opening"


async def test_black_to_move_in_italian(registry):
    data = (await registry.execute("inspect_position", {"fen": ITALIAN_FEN})).data

    assert data["side_to_move"] == "black"
    assert "Bc5" in data["legal_moves"]


@pytest.mark.parametrize(
    ("fen", "status", "is_check"),
    [
        ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", "checkmate", True),
        ("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1", "stalemate", False),
        ("k7/8/8/8/8/8/8/K7 w - - 0 1", "insufficient_material", False),
        ("k7/8/8/8/8/8/8/KQ6 w - - 100 80", "draw_by_rule", False),
        ("k7/8/8/8/8/8/8/K6q w - - 0 1", "ongoing", True),
    ],
    ids=["mat", "pat", "materiel-insuffisant", "regle-50-coups", "echec"],
)
async def test_game_status(registry, fen, status, is_check):
    data = (await registry.execute("inspect_position", {"fen": fen})).data

    assert data["status"] == status
    assert data["is_check"] is is_check


async def test_material_balance_and_phase(registry):
    fen = "k7/8/8/8/8/8/8/KQR5 w - - 0 40"

    data = (await registry.execute("inspect_position", {"fen": fen})).data

    assert data["material"] == {"white": 14, "black": 0, "balance": 14}
    assert data["phase"] == "middlegame_or_endgame"


async def test_illegal_fen_returns_actionable_error(registry):
    result = await registry.execute("inspect_position", {"fen": "8/8/8/8/8/8/8/K7 w - - 0 1"})

    assert result.error["code"] == "invalid_arguments"
    assert "pas de roi noir" in result.to_model_content()
    assert "Exemple valide" in result.to_model_content()
