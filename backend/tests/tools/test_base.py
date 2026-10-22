"""Exécuteur du catalogue : validation, enveloppe de résultat, erreurs, concurrence."""

from __future__ import annotations

import asyncio
import time

import pytest
from pydantic import ConfigDict

from app.tools.base import (
    SideEffect,
    ToolCall,
    ToolInput,
    ToolOutput,
    ToolRegistry,
    ToolSpec,
)
from app.tools.errors import (
    ToolError,
    ToolInputError,
    ToolTransientError,
    ToolUnavailableError,
)
from app.tools.resilience import ResiliencePolicy


class EchoInput(ToolInput):
    model_config = ConfigDict(json_schema_extra={"examples": [{"text": "bonjour"}]})

    text: str
    delay_s: float = 0.0


class EchoOutput(ToolOutput):
    text: str


async def echo(params: EchoInput) -> EchoOutput:
    await asyncio.sleep(params.delay_s)
    return EchoOutput(text=params.text)


def spec(name: str = "echo", handler=echo, policy: ResiliencePolicy | None = None) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Renvoie le texte reçu.",
        input_model=EchoInput,
        output_model=EchoOutput,
        handler=handler,
        side_effect=SideEffect.NONE,
        idempotent=True,
        policy=policy or ResiliencePolicy.none(),
    )


async def test_success_envelope():
    result = await ToolRegistry([spec()]).execute("echo", {"text": "salut"})

    assert result.ok
    assert result.data == {"text": "salut"}
    assert result.error is None
    assert result.duration_ms >= 0
    assert result.to_model_content() == '{"text": "salut"}'


async def test_unknown_tool_lists_available_tools():
    result = await ToolRegistry([spec()]).execute("ecoh", {})

    assert not result.ok
    assert result.error["code"] == "unknown_tool"
    assert "echo" in result.error["hint"]


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        ({}, "text : Field required"),
        ({"text": 3}, "text : Input should be a valid string"),
        ({"text": "x", "colour": "red"}, "colour : Extra inputs are not permitted"),
    ],
    ids=["champ-manquant", "mauvais-type", "champ-inconnu"],
)
async def test_schema_errors_are_explained_to_the_model(arguments, fragment):
    result = await ToolRegistry([spec()]).execute("echo", arguments)

    assert result.error["kind"] == "input"
    assert result.error["code"] == "invalid_arguments"
    assert fragment in result.error["message"]
    content = result.to_model_content()
    assert content.startswith("[invalid_arguments]")
    assert 'Exemple valide : {"text": "bonjour"}' in content


async def test_unexpected_exception_becomes_internal_error(caplog):
    async def broken(params: EchoInput) -> EchoOutput:
        raise ZeroDivisionError

    result = await ToolRegistry([spec(handler=broken)]).execute("echo", {"text": "x"})

    assert result.error["code"] == "internal_error"
    assert result.error["kind"] == "unavailable"
    assert "ZeroDivisionError" in result.error["message"]
    assert "Erreur interne dans l'outil echo" in caplog.text


async def test_transient_errors_are_retried_by_registry(sleep):
    calls = 0

    async def flaky(params: EchoInput) -> EchoOutput:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ToolTransientError(code="network_error", message="coupure")
        return EchoOutput(text=params.text)

    registry = ToolRegistry(
        [spec(handler=flaky, policy=ResiliencePolicy(timeout_s=1, max_attempts=3))], sleep=sleep
    )

    result = await registry.execute("echo", {"text": "x"})

    assert result.ok
    assert calls == 3


async def test_breaker_is_kept_per_tool(sleep):
    async def down(params: EchoInput) -> EchoOutput:
        raise ToolUnavailableError(code="auth_required", message="401")

    policy = ResiliencePolicy(timeout_s=1, max_attempts=1, breaker_threshold=1)
    registry = ToolRegistry([spec("a", down, policy), spec("b", echo, policy)], sleep=sleep)

    await registry.execute("a", {"text": "x"})

    assert (await registry.execute("a", {"text": "x"})).error["code"] == "circuit_open"
    assert (await registry.execute("b", {"text": "x"})).ok
    assert registry.breaker("b").allow()


async def test_execute_many_runs_concurrently_and_keeps_order():
    registry = ToolRegistry([spec(policy=ResiliencePolicy.none(timeout_s=2))])
    calls = [ToolCall("echo", {"text": str(i), "delay_s": 0.1}) for i in range(8)]

    start = time.perf_counter()
    results = await registry.execute_many(calls)
    elapsed = time.perf_counter() - start

    assert [r.data["text"] for r in results] == [str(i) for i in range(8)]
    assert elapsed < 0.4  # 8 x 0,1 s en série = 0,8 s


async def test_execute_many_isolates_failures():
    registry = ToolRegistry([spec()])

    results = await registry.execute_many([ToolCall("echo", {"text": "ok"}), ToolCall("nope", {})])

    assert [r.ok for r in results] == [True, False]


def test_duplicate_tool_name_is_refused():
    registry = ToolRegistry([spec()])

    with pytest.raises(ValueError, match="déjà enregistré"):
        registry.register(spec())


def test_definitions_expose_json_schema():
    registry = ToolRegistry([spec()])

    (definition,) = registry.definitions()

    assert registry.names == ["echo"]
    assert registry.get("echo").idempotent
    assert definition["name"] == "echo"
    assert definition["input_schema"]["required"] == ["text"]
    assert definition["input_schema"]["additionalProperties"] is False


@pytest.mark.parametrize(
    ("error", "kind", "retryable"),
    [
        (ToolInputError(code="c", message="m"), "input", False),
        (ToolTransientError(code="c", message="m"), "transient", True),
        (ToolUnavailableError(code="c", message="m"), "unavailable", False),
    ],
)
def test_error_families(error: ToolError, kind, retryable):
    assert error.kind == kind
    assert error.retryable is retryable
    assert error.to_model_message() == "[c] m"
    assert str(error) == "m"


def test_retry_after_is_exposed_in_details():
    error = ToolUnavailableError(
        code="rate_limited", message="429", hint="attendre", retry_after_s=60
    )

    assert error.to_dict()["details"] == {"retry_after_s": 60}
    assert error.to_model_message() == "[rate_limited] 429 Conseil : attendre"
