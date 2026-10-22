"""Outils start_deep_analysis / get_deep_analysis : soumission puis interrogation d'état."""

from __future__ import annotations

import asyncio

import pytest

from app.tools.deep_analysis import (
    DeepAnalysisJobs,
    GetDeepAnalysisInput,
    StartDeepAnalysisInput,
)
from app.tools.errors import ToolUnavailableError
from tests.conftest import ITALIAN_FEN, START_FEN, StubAnalyzer


class SlowAnalyzer(StubAnalyzer):
    """Analyse qui ne se termine que lorsque le test relâche `release`."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.release = asyncio.Event()

    async def analyse(self, board, *, depth, multipv):
        await self.release.wait()
        return await super().analyse(board, depth=depth, multipv=multipv)


async def test_submit_then_poll_until_done(clock):
    analyzer = SlowAnalyzer(scores=[80], depth=30)
    jobs = DeepAnalysisJobs(analyzer, clock=clock, poll_after_s=10)

    ticket = await jobs.start(StartDeepAnalysisInput(fen=ITALIAN_FEN, depth=30))
    assert ticket.status == "pending"
    assert ticket.poll_after_s == 10

    await asyncio.sleep(0)  # la tâche démarre
    clock.advance(42)
    state = await jobs.get(GetDeepAnalysisInput(job_id=ticket.job_id))
    assert state.status == "running"
    assert state.elapsed_s == 42
    assert state.result is None
    assert state.poll_after_s == 10

    analyzer.release.set()
    await jobs.wait(ticket.job_id)
    clock.advance(100)  # le temps écoulé est figé à la fin de la tâche
    state = await jobs.get(GetDeepAnalysisInput(job_id=ticket.job_id))
    assert state.status == "done"
    assert state.elapsed_s == 42
    assert state.result.score_cp == 80
    assert state.result.depth == 30
    assert state.poll_after_s is None


async def test_submit_is_idempotent_per_fen_and_depth(clock):
    jobs = DeepAnalysisJobs(SlowAnalyzer(), clock=clock)

    first = await jobs.start(StartDeepAnalysisInput(fen=START_FEN, depth=30))
    again = await jobs.start(StartDeepAnalysisInput(fen=START_FEN, depth=30))
    deeper = await jobs.start(StartDeepAnalysisInput(fen=START_FEN, depth=35))

    assert again.job_id == first.job_id
    assert deeper.job_id != first.job_id


async def test_failed_job_reports_error_and_can_be_resubmitted(clock):
    analyzer = StubAnalyzer(
        error=ToolUnavailableError(code="engine_not_found", message="Stockfish introuvable")
    )
    jobs = DeepAnalysisJobs(analyzer, clock=clock)

    ticket = await jobs.start(StartDeepAnalysisInput(fen=START_FEN))
    await jobs.wait(ticket.job_id)
    state = await jobs.get(GetDeepAnalysisInput(job_id=ticket.job_id))
    retry = await jobs.start(StartDeepAnalysisInput(fen=START_FEN))

    assert state.status == "failed"
    assert state.error == "[engine_not_found] Stockfish introuvable"
    assert retry.job_id != ticket.job_id  # un échec n'est pas réutilisé


async def test_unexpected_crash_is_captured(clock):
    jobs = DeepAnalysisJobs(StubAnalyzer(error=RuntimeError("boum")), clock=clock)

    ticket = await jobs.start(StartDeepAnalysisInput(fen=START_FEN))
    await jobs.wait(ticket.job_id)

    state = await jobs.get(GetDeepAnalysisInput(job_id=ticket.job_id))
    assert state.status == "failed"
    assert state.error == "Erreur interne : RuntimeError"


async def test_tools_through_registry(make_registry):
    registry = make_registry(StubAnalyzer(scores=[15], depth=30))

    ticket = await registry.execute("start_deep_analysis", {"fen": START_FEN})
    job_id = ticket.data["job_id"]
    for _ in range(5):
        state = await registry.execute("get_deep_analysis", {"job_id": job_id})
        if state.data["status"] == "done":
            break
        await asyncio.sleep(0)

    assert state.data["status"] == "done"
    assert state.data["result"]["assessment"] == "Position équilibrée"


async def test_unknown_job_is_an_input_error(make_registry):
    result = await make_registry().execute("get_deep_analysis", {"job_id": "inconnu"})

    assert result.error["kind"] == "input"
    assert result.error["code"] == "unknown_job"
    assert "start_deep_analysis" in result.error["hint"]


@pytest.mark.parametrize("depth", [22, 41])
async def test_depth_bounds(make_registry, depth):
    result = await make_registry().execute(
        "start_deep_analysis", {"fen": START_FEN, "depth": depth}
    )

    assert result.error["code"] == "invalid_arguments"
