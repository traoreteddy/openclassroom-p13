"""Adaptateur ToolSpec → outil LangChain (option C du contrat, § 10.5).

Les `ToolSpec` restent la source de vérité (nom, description, schéma Pydantic).
Cet adaptateur les expose à LangGraph sous forme de `StructuredTool` standards
(`llm.bind_tools(...)`, `ToolNode(...)`), en faisant passer chaque appel par le
`ToolRegistry` : validation, timeout, retry, disjoncteur.

Ce que reçoit le modèle (contenu du `ToolMessage`) :

| Situation                         | Contenu                                   | status    |
|-----------------------------------|-------------------------------------------|-----------|
| succès                            | données JSON de l'outil                   | success   |
| arguments invalides               | [invalid_arguments] … Exemple valide : {…}| error     |
| panne / erreur de l'outil         | [code] message Conseil : …                | error     |

L'application reçoit en plus, dans `ToolMessage.artifact`, le `ToolResult` complet
(données structurées pour le front, durée, erreur détaillée).
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool, ToolException
from pydantic import ValidationError

from app.tools.base import ToolRegistry, ToolSpec, invalid_arguments_error


def to_langchain_tool(registry: ToolRegistry, spec: ToolSpec) -> StructuredTool:
    """Crée l'outil LangChain correspondant à une fiche `ToolSpec`."""

    async def run(**arguments: Any) -> tuple[str, dict[str, Any]]:
        result = await registry.execute(spec.name, arguments)
        if not result.ok:
            # Message déjà formaté pour le modèle ; LangChain le renvoie avec status="error".
            raise ToolException(result.to_model_content())
        return result.to_model_content(), result.model_dump(mode="json")

    def on_validation_error(exc: ValidationError) -> str:
        return invalid_arguments_error(spec, exc).to_model_message()

    return StructuredTool.from_function(
        coroutine=run,
        name=spec.name,
        description=spec.description,
        args_schema=spec.input_model,
        response_format="content_and_artifact",
        handle_validation_error=on_validation_error,
        handle_tool_error=True,
    )


def build_langchain_tools(registry: ToolRegistry) -> list[StructuredTool]:
    """Tous les outils du catalogue, prêts pour `llm.bind_tools` et `ToolNode`."""
    return [to_langchain_tool(registry, registry.get(name)) for name in registry.names]
