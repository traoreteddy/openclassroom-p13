"""Timeout, retry exponentiel (tenacity) et disjoncteur."""

from __future__ import annotations

import asyncio

import pytest

from app.tools.errors import ToolInputError, ToolTransientError, ToolUnavailableError
from app.tools.resilience import (
    BreakerState,
    CircuitBreaker,
    ResiliencePolicy,
    call_with_resilience,
    make_breaker,
)


class Flaky:
    """Échoue `failures` fois avec `error`, puis réussit."""

    def __init__(self, failures: int, error: Exception | None = None) -> None:
        self.failures = failures
        self.error = error or ToolTransientError(code="upstream_error", message="HTTP 503")
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return "ok"


POLICY = ResiliencePolicy(timeout_s=1, max_attempts=3, breaker_threshold=None)


async def test_retry_until_success(sleep):
    fn = Flaky(failures=2)

    result = await call_with_resilience(fn, tool="t", policy=POLICY, sleep=sleep)

    assert result == "ok"
    assert fn.calls == 3
    assert len(sleep.calls) == 2  # deux attentes entre trois tentatives


async def test_backoff_grows_exponentially(sleep):
    policy = ResiliencePolicy(
        timeout_s=1, max_attempts=4, backoff_initial_s=1, backoff_jitter_s=0, breaker_threshold=None
    )

    await call_with_resilience(Flaky(failures=3), tool="t", policy=policy, sleep=sleep)

    assert sleep.calls == [1, 2, 4]


async def test_retries_exhausted_becomes_unavailable(sleep):
    fn = Flaky(failures=10)

    with pytest.raises(ToolUnavailableError) as info:
        await call_with_resilience(fn, tool="lichess", policy=POLICY, sleep=sleep)

    assert info.value.code == "retries_exhausted"
    assert "3 fois" in info.value.message
    assert info.value.details["last_error"]["code"] == "upstream_error"
    assert fn.calls == 3


@pytest.mark.parametrize(
    "error",
    [
        ToolInputError(code="invalid_arguments", message="FEN invalide"),
        ToolUnavailableError(code="rate_limited", message="429", retry_after_s=60),
    ],
    ids=["erreur-de-schema", "panne-durable"],
)
async def test_non_transient_errors_are_not_retried(sleep, error):
    fn = Flaky(failures=1, error=error)

    with pytest.raises(type(error)):
        await call_with_resilience(fn, tool="t", policy=POLICY, sleep=sleep)

    assert fn.calls == 1
    assert sleep.calls == []


async def test_timeout_is_transient_then_retried(sleep):
    calls = 0

    async def slow_then_fast() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(1)
        return "ok"

    policy = ResiliencePolicy(timeout_s=0.01, max_attempts=2, breaker_threshold=None)

    assert await call_with_resilience(slow_then_fast, tool="t", policy=policy, sleep=sleep) == "ok"
    assert calls == 2


async def test_timeout_message_names_the_tool(sleep):
    async def never() -> None:
        await asyncio.sleep(1)

    with pytest.raises(ToolUnavailableError) as info:
        await call_with_resilience(
            never, tool="stockfish", policy=ResiliencePolicy.none(timeout_s=0.01), sleep=sleep
        )

    assert info.value.details["last_error"]["code"] == "timeout"
    assert "stockfish" in info.value.details["last_error"]["message"]


# --- Disjoncteur ---------------------------------------------------------------------


def test_breaker_opens_after_threshold(clock):
    breaker = CircuitBreaker(threshold=3, reset_s=30, clock=clock)

    for _ in range(2):
        breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED

    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    assert not breaker.allow()
    assert breaker.remaining_s() == 30


def test_breaker_success_resets_failure_count(clock):
    breaker = CircuitBreaker(threshold=2, reset_s=30, clock=clock)

    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()

    assert breaker.state is BreakerState.CLOSED


def test_breaker_half_open_after_reset_delay(clock):
    breaker = CircuitBreaker(threshold=1, reset_s=30, clock=clock)
    breaker.record_failure()

    clock.advance(30)

    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.allow()


@pytest.mark.parametrize(
    ("trial_succeeds", "expected"),
    [(True, BreakerState.CLOSED), (False, BreakerState.OPEN)],
    ids=["essai-reussi", "essai-rate"],
)
def test_breaker_half_open_trial(clock, trial_succeeds, expected):
    breaker = CircuitBreaker(threshold=1, reset_s=30, clock=clock)
    breaker.record_failure()
    clock.advance(31)

    if trial_succeeds:
        breaker.record_success()
    else:
        breaker.record_failure()

    assert breaker.state is expected


def test_closed_breaker_has_no_remaining_delay(clock):
    assert CircuitBreaker(threshold=1, reset_s=30, clock=clock).remaining_s() == 0


def test_make_breaker_respects_policy(clock):
    assert make_breaker(ResiliencePolicy.none(), clock) is None
    assert make_breaker(ResiliencePolicy(breaker_threshold=2), clock) is not None


async def test_open_breaker_rejects_without_calling(sleep, clock):
    breaker = CircuitBreaker(threshold=1, reset_s=30, clock=clock)
    fn = Flaky(failures=10)
    policy = ResiliencePolicy(timeout_s=1, max_attempts=1)

    with pytest.raises(ToolUnavailableError):
        await call_with_resilience(fn, tool="t", policy=policy, breaker=breaker, sleep=sleep)
    with pytest.raises(ToolUnavailableError) as info:
        await call_with_resilience(fn, tool="t", policy=policy, breaker=breaker, sleep=sleep)

    assert info.value.code == "circuit_open"
    assert info.value.details["retry_after_s"] == 30
    assert fn.calls == 1  # le 2e appel n'a pas atteint le service


async def test_unavailable_error_counts_as_breaker_failure(sleep, clock):
    breaker = CircuitBreaker(threshold=1, reset_s=30, clock=clock)
    fn = Flaky(failures=1, error=ToolUnavailableError(code="auth_required", message="401"))

    with pytest.raises(ToolUnavailableError):
        await call_with_resilience(fn, tool="t", policy=POLICY, breaker=breaker, sleep=sleep)

    assert breaker.state is BreakerState.OPEN


async def test_input_error_does_not_trip_breaker(sleep, clock):
    breaker = CircuitBreaker(threshold=1, reset_s=30, clock=clock)
    fn = Flaky(failures=1, error=ToolInputError(code="invalid_arguments", message="x"))

    with pytest.raises(ToolInputError):
        await call_with_resilience(fn, tool="t", policy=POLICY, breaker=breaker, sleep=sleep)

    assert breaker.state is BreakerState.CLOSED


async def test_success_closes_breaker(sleep, clock):
    breaker = CircuitBreaker(threshold=2, reset_s=30, clock=clock)
    breaker.record_failure()

    await call_with_resilience(
        Flaky(failures=0), tool="t", policy=POLICY, breaker=breaker, sleep=sleep
    )

    assert breaker._failures == 0
