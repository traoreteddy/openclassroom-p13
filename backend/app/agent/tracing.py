"""Instrumentation OpenTelemetry de l'agent, exportée vers Langfuse.

- Un `TracerProvider` OpenTelemetry standard, avec la ressource `service.name=p13-agent`.
- Le **journal de décision** (`JournalSpanProcessor`) est toujours branché.
- **Langfuse** est branché seulement si ses clés sont configurées : son SDK (v5, bâti sur
  OpenTelemetry) ajoute au même provider un exporteur OTLP vers
  `<LANGFUSE_BASE_URL>/api/public/otel/v1/traces`. Sans clés, l'agent trace en local et
  rien ne sort de la machine.

Attributs lus par Langfuse pour regrouper les traces : `langfuse.session.id`,
`langfuse.user.id`, `langfuse.trace.name`.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Span, Status, StatusCode

from app.agent.journal import ATTR, JournalSpanProcessor, Pricing

SERVICE_NAME = "p13-agent"


@dataclass
class Tracing:
    provider: TracerProvider
    journal: JournalSpanProcessor
    langfuse: Any | None = None  # client Langfuse, ou None si non configuré

    @property
    def tracer(self) -> trace.Tracer:
        return self.provider.get_tracer("p13.agent")

    def langchain_callbacks(self) -> list[Any]:
        """Callbacks à passer aux appels LangChain : générations détaillées dans Langfuse."""
        if self.langfuse is None:
            return []
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]

    def flush(self) -> None:
        self.provider.force_flush()
        if self.langfuse is not None:
            self.langfuse.flush()

    def shutdown(self) -> None:
        self.flush()
        if self.langfuse is not None:
            self.langfuse.shutdown()
        self.provider.shutdown()


def configure_tracing(
    *,
    environment: str = "development",
    langfuse_base_url: str | None = None,
    langfuse_public_key: str | None = None,
    langfuse_secret_key: str | None = None,
    langfuse_timeout: int = 30,
    journal_path: Path | None = None,
    pricing: Pricing | None = None,
    exporter: SpanExporter | None = None,
) -> Tracing:
    """Construit le provider ; `exporter` sert aux tests (export en mémoire)."""
    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": SERVICE_NAME, "deployment.environment": environment}
        )
    )
    journal = JournalSpanProcessor(path=journal_path, pricing=pricing)
    provider.add_span_processor(journal)
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))

    langfuse = None
    if langfuse_public_key and langfuse_secret_key:
        from langfuse import Langfuse

        langfuse = Langfuse(
            public_key=langfuse_public_key,
            secret_key=langfuse_secret_key,
            base_url=langfuse_base_url,
            environment=environment,
            tracer_provider=provider,
            # Par défaut, le SDK n'exporte que les spans de son propre traceur et des
            # bibliothèques LLM connues : nos spans OpenTelemetry (`p13.agent`) seraient ignorés.
            should_export_span=lambda span: True,
            # Délai d'export OTLP : 5 s par défaut, trop court quand la machine est chargée
            # (modèle local + encodeur) ; les lots dépassés sont perdus, pas rejoués.
            timeout=langfuse_timeout,
        )
    return Tracing(provider=provider, journal=journal, langfuse=langfuse)


# --- Aide à l'instrumentation ---------------------------------------------------------------


@contextmanager
def turn_span(tracer: trace.Tracer, *, session_id: str, name: str = "agent_turn", **attrs: Any):
    """Span racine d'un tour de conversation, regroupé par session dans Langfuse."""
    attributes = {
        "langfuse.session.id": session_id,
        "session.id": session_id,
        "langfuse.trace.name": name,
        f"{ATTR}session_id": session_id,
        **{f"{ATTR}{k}": _attr(v) for k, v in attrs.items()},
    }
    with tracer.start_as_current_span(name, attributes=attributes) as span:
        yield span


@contextmanager
def step_span(
    tracer: trace.Tracer,
    *,
    kind: str,
    step: int,
    session_id: str,
    choice: str = "",
    arguments: dict[str, Any] | None = None,
):
    """Span d'une étape (`decision`, `tool_call`, `answer`) lue par le journal."""
    attributes = {
        f"{ATTR}kind": kind,
        f"{ATTR}step": step,
        f"{ATTR}session_id": session_id,
        f"{ATTR}choice": choice,
        f"{ATTR}arguments": json.dumps(arguments or {}, ensure_ascii=False),
        "langfuse.observation.type": "tool" if kind == "tool_call" else "span",
    }
    with tracer.start_as_current_span(f"{kind}:{choice or kind}", attributes=attributes) as span:
        yield span


def record_outcome(
    span: Span,
    *,
    status: str = "success",
    outcome: str = "",
    choice: str | None = None,
    arguments: dict[str, Any] | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> None:
    """Complète un span d'étape à la fin de l'étape."""
    span.set_attribute(f"{ATTR}status", status)
    span.set_attribute(f"{ATTR}outcome", outcome[:300])
    if choice is not None:
        span.set_attribute(f"{ATTR}choice", choice)
        span.update_name(f"{span.name.split(':', 1)[0]}:{choice}")
    if arguments is not None:
        span.set_attribute(f"{ATTR}arguments", json.dumps(arguments, ensure_ascii=False))
    if tokens_in or tokens_out:
        span.set_attribute(f"{ATTR}tokens_in", tokens_in)
        span.set_attribute(f"{ATTR}tokens_out", tokens_out)
    if status == "error":
        span.set_status(Status(StatusCode.ERROR, outcome[:120]))


def _attr(value: Any) -> Any:
    return value if isinstance(value, str | int | float | bool) else json.dumps(value)
