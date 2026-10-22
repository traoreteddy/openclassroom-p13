"""Journal de décision de l'agent, dérivé des spans OpenTelemetry.

Une seule instrumentation (les spans) alimente deux sorties : Langfuse (visualisation)
et ce journal (fichier JSON Lines + mémoire, pour les mesures et les tests).

Schéma d'une entrée :

| Champ        | Contenu                                                              |
|--------------|----------------------------------------------------------------------|
| timestamp    | début de l'étape, ISO 8601 UTC                                       |
| session_id   | identifiant de session (regroupe les tours d'une même conversation)  |
| trace_id     | identifiant OpenTelemetry du tour (même valeur dans Langfuse)        |
| step         | numéro de l'étape dans le tour (1 = première décision)               |
| kind         | `decision` (choix du LLM), `tool_call` (exécution), `answer` (réponse)|
| choice       | outil choisi, ou `answer` / `none`                                   |
| arguments    | arguments de l'appel (dict)                                          |
| status       | `success`, `error`, `skipped`                                        |
| duration_ms  | durée de l'étape                                                     |
| tokens_in    | tokens d'entrée du LLM (0 pour un appel d'outil)                     |
| tokens_out   | tokens de sortie du LLM                                              |
| cost_eur     | coût : 0 en local ; estimation « équivalent cloud » si un tarif est réglé |
| outcome      | issue courte : code d'erreur, nombre de passages, extrait de réponse |
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

Kind = Literal["decision", "tool_call", "answer"]
Status = Literal["success", "error", "skipped"]

# Préfixe des attributs de span lus par le journal
ATTR = "p13."


@dataclass(frozen=True)
class DecisionEntry:
    timestamp: str
    session_id: str
    trace_id: str
    step: int
    kind: Kind
    choice: str
    arguments: dict[str, Any]
    status: Status
    duration_ms: float
    tokens_in: int
    tokens_out: int
    cost_eur: float
    outcome: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass(frozen=True)
class Pricing:
    """Tarif en € par million de tokens. 0 pour un modèle local ; une valeur non nulle
    sert à estimer ce que coûterait le même trafic chez un fournisseur cloud."""

    input_per_million: float = 0.0
    output_per_million: float = 0.0

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return round(
            (tokens_in * self.input_per_million + tokens_out * self.output_per_million) / 1e6, 6
        )


def span_to_entry(span: ReadableSpan, pricing: Pricing) -> DecisionEntry | None:
    """Convertit un span terminé en entrée de journal, ou None s'il n'en est pas un."""
    attributes = span.attributes or {}
    kind = attributes.get(f"{ATTR}kind")
    if kind not in ("decision", "tool_call", "answer"):
        return None
    tokens_in = int(attributes.get(f"{ATTR}tokens_in", 0))
    tokens_out = int(attributes.get(f"{ATTR}tokens_out", 0))
    start = span.start_time or 0
    end = span.end_time or start
    return DecisionEntry(
        timestamp=datetime.fromtimestamp(start / 1e9, tz=UTC).isoformat(timespec="milliseconds"),
        session_id=str(attributes.get(f"{ATTR}session_id", "")),
        trace_id=format(span.context.trace_id, "032x") if span.context else "",
        step=int(attributes.get(f"{ATTR}step", 0)),
        kind=kind,  # type: ignore[arg-type]
        choice=str(attributes.get(f"{ATTR}choice", "")),
        arguments=json.loads(str(attributes.get(f"{ATTR}arguments", "{}"))),
        status=str(attributes.get(f"{ATTR}status", "success")),  # type: ignore[arg-type]
        duration_ms=round((end - start) / 1e6, 1),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_eur=pricing.cost(tokens_in, tokens_out),
        outcome=str(attributes.get(f"{ATTR}outcome", ""))[:300],
    )


class JournalSpanProcessor(SpanProcessor):
    """Transforme les spans `p13.*` terminés en entrées de journal.

    Les entrées sont gardées en mémoire (mesures, tests) et, si un chemin est donné,
    ajoutées à un fichier JSON Lines. `on_entry` permet de brancher un autre destinataire.
    """

    def __init__(
        self,
        *,
        path: Path | None = None,
        pricing: Pricing | None = None,
        on_entry: Callable[[DecisionEntry], None] | None = None,
    ) -> None:
        self.path = path
        self.pricing = pricing or Pricing()
        self.entries: list[DecisionEntry] = []
        self._on_entry = on_entry
        self._lock = threading.Lock()

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        entry = span_to_entry(span, self.pricing)
        if entry is None:
            return
        with self._lock:
            self.entries.append(entry)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as file:
                    file.write(entry.to_json() + "\n")
        if self._on_entry is not None:
            self._on_entry(entry)

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def for_session(self, session_id: str) -> list[DecisionEntry]:
        return [e for e in self.entries if e.session_id == session_id]

    def clear(self) -> None:
        with self._lock:
            self.entries.clear()
