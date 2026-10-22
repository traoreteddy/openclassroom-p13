"""Fabrique du modèle de langage de l'agent (Ollama sur la machine hôte, via LangChain)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel

# Fabrique : `structured_schema` impose une sortie JSON conforme au schéma (mode structuré).
LLMFactory = Callable[..., BaseChatModel]


def ollama_factory(
    *, base_url: str, model: str, temperature: float = 0.0, max_tokens: int = 1024
) -> LLMFactory:
    def make(structured_schema: dict[str, Any] | None = None) -> BaseChatModel:
        from langchain_ollama import ChatOllama

        options: dict[str, Any] = {
            "model": model,
            "base_url": base_url,
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if model.startswith("qwen3"):
            options["reasoning"] = False  # le mode « réflexion » de Qwen3 double la latence
        if structured_schema is not None:
            options["format"] = structured_schema
        return ChatOllama(**options)

    return make


def usage(message: Any) -> tuple[int, int]:
    """(tokens d'entrée, tokens de sortie) d'une réponse LangChain, 0 si inconnu."""
    meta = getattr(message, "usage_metadata", None) or {}
    return int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))
