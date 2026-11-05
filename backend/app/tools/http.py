"""Appel HTTP commun aux outils, avec traduction des échecs dans la taxonomie d'erreurs.

Le 429 n'est pas réessayé automatiquement : Lichess demande d'attendre une minute
entière, et réessayer aggraverait le blocage.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import httpx

from app.tools.errors import (
    ToolError,
    ToolInputError,
    ToolTransientError,
    ToolUnavailableError,
)

# Permet à un service de traduire lui-même certaines réponses (ex. quota YouTube).
ErrorOverride = Callable[[httpx.Response], ToolError | None]

DEFAULT_RETRY_AFTER_S = 60.0


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    service: str,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    override: ErrorOverride | None = None,
) -> Any:
    try:
        response = await client.get(url, params=params, headers=headers)
    except httpx.TimeoutException as exc:
        raise ToolTransientError(
            code="timeout", message=f"{service} n'a pas répondu à temps."
        ) from exc
    except httpx.TransportError as exc:
        raise ToolTransientError(
            code="network_error",
            message=f"Connexion à {service} impossible ({type(exc).__name__}).",
        ) from exc

    if response.is_success:
        try:
            return response.json()
        except ValueError as exc:
            raise ToolUnavailableError(
                code="bad_response",
                message=f"{service} a renvoyé une réponse illisible.",
            ) from exc

    if override is not None and (error := override(response)) is not None:
        raise error
    raise _error_for_status(response, service)


def _error_for_status(response: httpx.Response, service: str) -> ToolError:
    status = response.status_code
    if status == 429:
        retry_after = _retry_after(response)
        return ToolUnavailableError(
            code="rate_limited",
            message=f"{service} limite le nombre de requêtes.",
            hint=f"Attendre {retry_after:g} s avant de rappeler ce service.",
            retry_after_s=retry_after,
        )
    if status in (401, 403):
        return ToolUnavailableError(
            code="auth_required",
            message=f"{service} refuse l'accès (HTTP {status}).",
            hint="Vérifier la clé ou le jeton d'API dans la configuration (.env).",
        )
    if status >= 500:
        return ToolTransientError(
            code="upstream_error", message=f"{service} est en erreur (HTTP {status})."
        )
    return ToolInputError(
        code="rejected_by_service",
        message=f"{service} a refusé la requête (HTTP {status}) : {response.text[:200]}",
        hint="Vérifier les paramètres de l'appel.",
    )


def _retry_after(response: httpx.Response) -> float:
    try:
        return float(response.headers["Retry-After"])
    except (KeyError, ValueError):
        return DEFAULT_RETRY_AFTER_S
