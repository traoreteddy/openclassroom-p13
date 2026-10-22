"""Politiques de résilience : timeout par outil, retry exponentiel, disjoncteur.

Ordre d'application pour un appel d'outil :

    disjoncteur ouvert ? ──oui──► ToolUnavailableError("circuit_open")  (abandon signalé)
          │ non
          ▼
    tentative ── timeout ──► ToolTransientError("timeout")
          │
          ├─ ToolTransientError ──► nouvel essai après attente exponentielle (+ gigue)
          │                         … puis ToolUnavailableError("retries_exhausted")
          ├─ ToolInputError ──────► remontée immédiate (le modèle doit corriger l'appel)
          └─ succès ──────────────► le disjoncteur se referme
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.tools.errors import ToolInputError, ToolTransientError, ToolUnavailableError

Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]


@dataclass(frozen=True)
class ResiliencePolicy:
    """Politique d'un outil. Les valeurs par défaut conviennent à une API HTTP externe."""

    timeout_s: float = 5.0
    max_attempts: int = 3
    backoff_initial_s: float = 0.5
    backoff_max_s: float = 8.0
    backoff_jitter_s: float = 0.5
    breaker_threshold: int | None = 5
    breaker_reset_s: float = 30.0

    @classmethod
    def none(cls, timeout_s: float = 5.0) -> ResiliencePolicy:
        """Aucune protection hormis le timeout (sert de référence « avant »)."""
        return cls(timeout_s=timeout_s, max_attempts=1, breaker_threshold=None)


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Disjoncteur à trois états.

    - CLOSED : les appels passent ; `threshold` échecs consécutifs ouvrent le circuit.
    - OPEN : les appels sont refusés sans contacter le service pendant `reset_s`.
    - HALF_OPEN : un appel d'essai passe ; succès → CLOSED, échec → OPEN.
    """

    def __init__(self, threshold: int, reset_s: float, clock: Clock = time.monotonic) -> None:
        self.threshold = threshold
        self.reset_s = reset_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> BreakerState:
        if self._opened_at is None:
            return BreakerState.CLOSED
        if self._clock() - self._opened_at >= self.reset_s:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    def allow(self) -> bool:
        return self.state is not BreakerState.OPEN

    def remaining_s(self) -> float:
        if self._opened_at is None:
            return 0.0
        return max(0.0, self.reset_s - (self._clock() - self._opened_at))

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        if self.state is BreakerState.HALF_OPEN:
            self._opened_at = self._clock()
            return
        self._failures += 1
        if self._failures >= self.threshold:
            self._opened_at = self._clock()


def make_breaker(policy: ResiliencePolicy, clock: Clock = time.monotonic) -> CircuitBreaker | None:
    if policy.breaker_threshold is None:
        return None
    return CircuitBreaker(policy.breaker_threshold, policy.breaker_reset_s, clock)


async def call_with_resilience[T](
    fn: Callable[[], Awaitable[T]],
    *,
    tool: str,
    policy: ResiliencePolicy,
    breaker: CircuitBreaker | None = None,
    sleep: Sleep = asyncio.sleep,
) -> T:
    """Exécute `fn` avec timeout, retry et disjoncteur selon `policy`."""
    if breaker is not None and not breaker.allow():
        raise ToolUnavailableError(
            code="circuit_open",
            message=f"L'outil {tool} est temporairement désactivé après plusieurs échecs.",
            hint="Ne pas réessayer maintenant : répondre avec les autres sources disponibles.",
            retry_after_s=round(breaker.remaining_s(), 1),
        )

    retrying = AsyncRetrying(
        stop=stop_after_attempt(policy.max_attempts),
        wait=wait_exponential_jitter(
            multiplier=policy.backoff_initial_s,
            max=policy.backoff_max_s,
            jitter=policy.backoff_jitter_s,
        ),
        retry=retry_if_exception_type(ToolTransientError),
        reraise=True,
        sleep=sleep,
    )

    try:
        async for attempt in retrying:
            with attempt:
                result = await _with_timeout(fn, tool=tool, timeout_s=policy.timeout_s)
    except ToolInputError:
        raise  # l'appel est faux : ce n'est pas une panne du service
    except ToolTransientError as exc:
        if breaker is not None:
            breaker.record_failure()
        attempts = retrying.statistics.get("attempt_number", policy.max_attempts)
        raise ToolUnavailableError(
            code="retries_exhausted",
            message=f"L'outil {tool} a échoué {attempts} fois de suite ({exc.code}).",
            hint="Continuer sans ce résultat et le signaler à l'utilisateur.",
            details={"last_error": exc.to_dict()},
        ) from exc
    except ToolUnavailableError:
        if breaker is not None:
            breaker.record_failure()
        raise

    if breaker is not None:
        breaker.record_success()
    return result


async def _with_timeout[T](fn: Callable[[], Awaitable[T]], *, tool: str, timeout_s: float) -> T:
    try:
        async with asyncio.timeout(timeout_s):
            return await fn()
    except TimeoutError as exc:
        raise ToolTransientError(
            code="timeout",
            message=f"L'outil {tool} n'a pas répondu en {timeout_s:g} s.",
        ) from exc
