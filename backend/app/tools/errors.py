"""Taxonomie des erreurs d'outil.

Trois familles, chacune avec une politique différente :

| Famille                | Exemple                          | Politique                    |
|------------------------|----------------------------------|------------------------------|
| ToolInputError         | FEN invalide, paramètre manquant | Pas de retry : le modèle     |
|                        |                                  | corrige l'appel              |
| ToolTransientError     | timeout, HTTP 5xx, connexion     | Retry exponentiel            |
|                        | coupée                           |                              |
| ToolUnavailableError   | disjoncteur ouvert, quota épuisé,| Repli ou abandon signalé :   |
|                        | jeton manquant, retries épuisés  | l'agent continue sans l'outil|

Chaque erreur se convertit en message exploitable par le modèle (`to_model_message`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ErrorKind = Literal["input", "transient", "unavailable"]


@dataclass(eq=False)
class ToolError(Exception):
    """Erreur levée par un outil. `code` est stable et documenté dans le contrat."""

    code: str
    message: str
    hint: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    kind: ErrorKind = field(init=False, default="transient")
    retryable: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
            "retryable": self.retryable,
            "details": self.details,
        }

    def to_model_message(self) -> str:
        """Message destiné au LLM : quoi, pourquoi, et quoi faire ensuite."""
        parts = [f"[{self.code}] {self.message}"]
        if self.hint:
            parts.append(f"Conseil : {self.hint}")
        return " ".join(parts)


@dataclass(eq=False)
class ToolInputError(ToolError):
    """Erreur de schéma ou de valeur : l'appel est faux, le réessayer tel quel est inutile."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.kind = "input"
        self.retryable = False


@dataclass(eq=False)
class ToolTransientError(ToolError):
    """Erreur récupérable : réessayer le même appel peut réussir."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.kind = "transient"
        self.retryable = True


@dataclass(eq=False)
class ToolUnavailableError(ToolError):
    """Panne durable : ne pas réessayer maintenant, continuer sans l'outil."""

    retry_after_s: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.kind = "unavailable"
        self.retryable = False
        if self.retry_after_s is not None:
            self.details.setdefault("retry_after_s", self.retry_after_s)
