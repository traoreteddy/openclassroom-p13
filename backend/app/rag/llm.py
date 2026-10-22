"""Client minimal du LLM local (Ollama sur la machine hôte), en sortie structurée."""

from __future__ import annotations

import json
from typing import Any

import httpx


class OllamaChat:
    """Appel minimal à l'API chat d'Ollama avec sortie structurée."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str,
        model: str,
        timeout_s: float = 120,
        max_tokens: int | None = None,
    ) -> None:
        self._client = client
        self._url = base_url.rstrip("/") + "/api/chat"
        self.model = model
        self._timeout_s = timeout_s
        # Borne la génération : sans elle, un petit modèle peut boucler en sortie structurée.
        self._max_tokens = max_tokens

    async def structured(
        self, system: str, user: str, schema: dict[str, Any], *, temperature: float = 0.0
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": temperature}
        if self._max_tokens is not None:
            options["num_predict"] = self._max_tokens
        response = await self._client.post(
            self._url,
            json={
                "model": self.model,
                "stream": False,
                "format": schema,
                "options": options,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        return json.loads(response.json()["message"]["content"])
