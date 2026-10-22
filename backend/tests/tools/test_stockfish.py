"""Outil evaluate_position : moteur remplacé par un stub ; StockfishAnalyzer testé sans binaire."""

from __future__ import annotations

import chess
import chess.engine
import pytest

from app.tools.errors import ToolTransientError, ToolUnavailableError
from app.tools.stockfish import StockfishAnalyzer, assess
from tests.conftest import ITALIAN_FEN, START_FEN, StubAnalyzer


async def test_single_line_evaluation(make_registry):
    analyzer = StubAnalyzer([35], depth=16)

    data = (await make_registry(analyzer).execute("evaluate_position", {"fen": START_FEN})).data

    assert data["score_cp"] == 35
    assert data["mate"] is None
    assert data["depth"] == 16
    assert data["assessment"] == "Léger avantage aux Blancs"
    (line,) = data["lines"]
    assert line["rank"] == 1
    assert line["move_san"] == chess.Board().san(chess.Move.from_uci(line["move_uci"]))
    assert len(line["pv_san"]) == 2
    assert analyzer.calls == [(START_FEN, 16, 1)]


async def test_multipv_returns_ranked_lines(make_registry):
    analyzer = StubAnalyzer([40, 10, -20])

    data = (
        await make_registry(analyzer).execute(
            "evaluate_position", {"fen": ITALIAN_FEN, "lines": 3, "depth": 12}
        )
    ).data

    assert [line["rank"] for line in data["lines"]] == [1, 2, 3]
    assert [line["score_cp"] for line in data["lines"]] == [40, 10, -20]
    assert analyzer.calls[0][1:] == (12, 3)


async def test_mate_score(make_registry):
    data = (
        await make_registry(StubAnalyzer([("mate", -2)])).execute(
            "evaluate_position", {"fen": START_FEN}
        )
    ).data

    assert data["score_cp"] is None
    assert data["mate"] == -2
    assert data["assessment"] == "Mat en 2 pour les Noirs"


@pytest.mark.parametrize(
    ("fen", "label"),
    [
        ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", "échec et mat"),
        ("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1", "pat"),
        ("k7/8/8/8/8/8/8/K7 w - - 0 1", "nulle"),
    ],
)
async def test_finished_game_does_not_call_engine(make_registry, fen, label):
    analyzer = StubAnalyzer()

    data = (await make_registry(analyzer).execute("evaluate_position", {"fen": fen})).data

    assert data["lines"] == []
    assert label in data["assessment"]
    assert analyzer.calls == []


async def test_engine_without_pv_gives_no_line(make_registry):
    class NoPv(StubAnalyzer):
        async def analyse(self, board, *, depth, multipv):
            return [{"depth": 5}]

    data = (await make_registry(NoPv()).execute("evaluate_position", {"fen": START_FEN})).data

    assert data["lines"] == []
    assert data["assessment"] == "Évaluation indisponible"


@pytest.mark.parametrize(
    ("score_cp", "mate", "expected"),
    [
        (0, None, "Position équilibrée"),
        (-30, None, "Position équilibrée"),
        (31, None, "Léger avantage aux Blancs"),
        (-100, None, "Léger avantage aux Noirs"),
        (250, None, "Net avantage aux Blancs"),
        (-301, None, "Avantage décisif aux Noirs"),
        (None, 3, "Mat en 3 pour les Blancs"),
        (None, None, "Position équilibrée"),
    ],
)
def test_assessment_thresholds(score_cp, mate, expected):
    assert assess(score_cp, mate) == expected


async def test_engine_crash_is_retried(make_registry):
    class CrashOnce(StubAnalyzer):
        async def analyse(self, board, *, depth, multipv):
            if not self.calls:
                self.calls.append(("crash", depth, multipv))
                raise ToolTransientError(code="engine_crashed", message="arrêt")
            return await super().analyse(board, depth=depth, multipv=multipv)

    result = await make_registry(CrashOnce()).execute("evaluate_position", {"fen": START_FEN})

    assert result.ok


@pytest.mark.parametrize(
    "arguments",
    [
        {"fen": START_FEN, "depth": 7},
        {"fen": START_FEN, "depth": 23},
        {"fen": START_FEN, "lines": 4},
    ],
)
async def test_bounds(make_registry, arguments):
    result = await make_registry().execute("evaluate_position", arguments)

    assert result.error["code"] == "invalid_arguments"


# --- StockfishAnalyzer (sans binaire Stockfish) -----------------------------------------


async def test_missing_binary_is_unavailable():
    analyzer = StockfishAnalyzer("/chemin/inexistant/stockfish")

    with pytest.raises(ToolUnavailableError) as info:
        await analyzer.analyse(chess.Board(), depth=8, multipv=1)

    assert info.value.code == "engine_not_found"
    assert "STOCKFISH_PATH" in info.value.hint


class FakeEngine:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.configured: dict | None = None
        self.quit_called = False

    async def configure(self, options: dict) -> None:
        self.configured = options

    async def analyse(self, board, limit, multipv):
        if self.error:
            raise self.error
        return [{"depth": limit.depth, "pv": [chess.Move.from_uci("e2e4")]}]

    async def quit(self) -> None:
        self.quit_called = True


@pytest.fixture
def fake_popen(monkeypatch):
    engines: list[FakeEngine] = []

    def install(*errors: Exception | None) -> list[FakeEngine]:
        queue = list(errors)

        async def popen_uci(path):
            engine = FakeEngine(queue.pop(0) if queue else None)
            engines.append(engine)
            return object(), engine

        monkeypatch.setattr(chess.engine, "popen_uci", popen_uci)
        return engines

    return install


async def test_engine_is_started_once_and_configured(fake_popen):
    engines = fake_popen(None)
    analyzer = StockfishAnalyzer("stockfish", threads=2, hash_mb=32)

    await analyzer.analyse(chess.Board(), depth=8, multipv=1)
    await analyzer.analyse(chess.Board(), depth=8, multipv=1)
    await analyzer.close()

    assert len(engines) == 1
    assert engines[0].configured == {"Threads": 2, "Hash": 32}
    assert engines[0].quit_called


async def test_crashed_engine_is_replaced(fake_popen):
    engines = fake_popen(chess.engine.EngineTerminatedError("mort"), None)
    analyzer = StockfishAnalyzer()

    with pytest.raises(ToolTransientError) as info:
        await analyzer.analyse(chess.Board(), depth=8, multipv=1)
    await analyzer.analyse(chess.Board(), depth=8, multipv=1)

    assert info.value.code == "engine_crashed"
    assert len(engines) == 2
    assert engines[0].quit_called


async def test_cancelled_analysis_discards_engine(fake_popen):
    import asyncio

    class Hanging(FakeEngine):
        async def analyse(self, board, limit, multipv):
            await asyncio.sleep(10)

    engines: list[FakeEngine] = []

    async def popen_uci(path):
        engine = Hanging() if not engines else FakeEngine()
        engines.append(engine)
        return object(), engine

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(chess.engine, "popen_uci", popen_uci)
        analyzer = StockfishAnalyzer()
        with _pytest.raises(TimeoutError):
            async with asyncio.timeout(0.01):
                await analyzer.analyse(chess.Board(), depth=8, multipv=1)
        await asyncio.sleep(0)  # laisse l'arrêt en arrière-plan s'exécuter
        await analyzer.analyse(chess.Board(), depth=8, multipv=1)

    assert len(engines) == 2
    assert engines[0].quit_called


async def test_close_without_engine_is_noop():
    await StockfishAnalyzer().close()
