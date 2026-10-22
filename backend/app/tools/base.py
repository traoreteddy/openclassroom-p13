"""Socle commun des outils : spécification, enveloppe de résultat et exécuteur.

Un outil = un contrat (`ToolSpec`) :
- un nom et une description destinés au modèle ;
- un schéma d'entrée et un schéma de sortie (Pydantic → JSON Schema) ;
- ses effets (lecture seule, appel externe, création de tâche) et son idempotence ;
- sa politique de résilience.

L'exécuteur (`ToolRegistry.execute`) ne lève jamais d'exception : il renvoie toujours un
`ToolResult`, réussi ou en erreur, que l'agent peut transmettre tel quel au modèle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from app.tools.errors import ToolError, ToolInputError, ToolUnavailableError
from app.tools.resilience import (
    CircuitBreaker,
    Clock,
    ResiliencePolicy,
    Sleep,
    call_with_resilience,
    make_breaker,
)

logger = logging.getLogger(__name__)


class SideEffect(StrEnum):
    NONE = "none"  # calcul local pur
    READ_EXTERNAL = "read_external"  # lecture d'un service externe, sans modification
    CREATES_JOB = "creates_job"  # crée une tâche de fond (non idempotent)


class ToolInput(BaseModel):
    """Base des schémas d'entrée : champs inconnus refusés (erreur de schéma explicite)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


Handler = Callable[[Any], Awaitable[ToolOutput]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[ToolInput]
    output_model: type[ToolOutput]
    handler: Handler
    side_effect: SideEffect
    idempotent: bool
    policy: ResiliencePolicy = field(default_factory=ResiliencePolicy)

    def definition(self) -> dict[str, Any]:
        """Définition neutre (nom, description, JSON Schema d'entrée) pour un LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }


class ToolResult(BaseModel):
    tool: str
    ok: bool
    data: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    duration_ms: float

    def to_model_content(self) -> str:
        """Contenu renvoyé au modèle après l'appel."""
        if self.ok:
            return json.dumps(self.data, ensure_ascii=False)
        assert self.error is not None
        text = f"[{self.error['code']}] {self.error['message']}"
        if self.error.get("hint"):
            text += f" Conseil : {self.error['hint']}"
        return text


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


class ToolRegistry:
    """Catalogue d'outils + exécution protégée."""

    def __init__(
        self,
        specs: Iterable[ToolSpec] = (),
        *,
        sleep: Sleep = asyncio.sleep,
        clock: Clock = time.monotonic,
    ) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._breakers: dict[str, CircuitBreaker | None] = {}
        self._sleep = sleep
        self._clock = clock
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Outil déjà enregistré : {spec.name}")
        self._specs[spec.name] = spec
        self._breakers[spec.name] = make_breaker(spec.policy, self._clock)

    @property
    def names(self) -> list[str]:
        return list(self._specs)

    def get(self, name: str) -> ToolSpec:
        return self._specs[name]

    def breaker(self, name: str) -> CircuitBreaker | None:
        return self._breakers[name]

    def definitions(self) -> list[dict[str, Any]]:
        return [spec.definition() for spec in self._specs.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        start = time.perf_counter()
        try:
            spec = self._spec_or_error(name)
            params = _validate(spec, arguments)
            output = await call_with_resilience(
                lambda: spec.handler(params),
                tool=spec.name,
                policy=spec.policy,
                breaker=self._breakers[spec.name],
                sleep=self._sleep,
            )
        except ToolError as exc:
            return _result(name, start, error=exc)
        except Exception as exc:  # bug inattendu : on le journalise, l'agent continue
            logger.exception("Erreur interne dans l'outil %s", name)
            error = ToolUnavailableError(
                code="internal_error",
                message=f"Erreur interne de l'outil {name} : {type(exc).__name__}.",
                hint="Continuer sans ce résultat.",
            )
            return _result(name, start, error=error)
        return _result(name, start, data=output.model_dump(mode="json"))

    async def execute_many(self, calls: Iterable[ToolCall]) -> list[ToolResult]:
        """Exécute plusieurs appels en parallèle (asyncio.gather), dans l'ordre fourni."""
        return list(await asyncio.gather(*(self.execute(c.name, c.arguments) for c in calls)))

    def _spec_or_error(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise ToolInputError(
                code="unknown_tool",
                message=f"Outil inconnu : {name}.",
                hint=f"Outils disponibles : {', '.join(self._specs)}.",
            ) from None


def _validate(spec: ToolSpec, arguments: dict[str, Any]) -> ToolInput:
    try:
        return spec.input_model.model_validate(arguments)
    except ValidationError as exc:
        problems = "; ".join(_format_error(err) for err in exc.errors())
        example = (spec.input_model.model_config.get("json_schema_extra") or {}).get("examples")
        hint = "Corriger les paramètres puis rappeler l'outil."
        if example:
            hint += f" Exemple valide : {json.dumps(example[0], ensure_ascii=False)}"
        raise ToolInputError(
            code="invalid_arguments",
            message=f"Paramètres invalides pour {spec.name} : {problems}.",
            hint=hint,
            details={"errors": [_format_error(err) for err in exc.errors()]},
        ) from None


def _format_error(err: Any) -> str:
    location = ".".join(str(part) for part in err["loc"]) or "arguments"
    message = str(err["msg"]).removeprefix("Value error, ")
    return f"{location} : {message}"


def _result(
    name: str,
    start: float,
    *,
    data: dict[str, Any] | None = None,
    error: ToolError | None = None,
) -> ToolResult:
    return ToolResult(
        tool=name,
        ok=error is None,
        data=data,
        error=error.to_dict() if error else None,
        duration_ms=round((time.perf_counter() - start) * 1000, 2),
    )
