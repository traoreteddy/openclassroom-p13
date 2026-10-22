"""Dépendances FastAPI partagées par les routes."""

from fastapi import Request

from app.tools.base import ToolRegistry


def get_registry(request: Request) -> ToolRegistry:
    """Catalogue d'outils créé au démarrage de l'application (voir app.main.lifespan)."""
    return request.app.state.tool_registry
