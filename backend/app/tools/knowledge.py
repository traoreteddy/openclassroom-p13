"""Outil `search_opening_knowledge` : recherche vectorielle dans le corpus Wikipédia FR (Milvus).

L'outil dépend de deux interfaces :
- `Embedder` : transforme la question en vecteur (sentence-transformers, étape 2) ;
- `VectorStore` : cherche les passages les plus proches (Milvus, étape 2).
Les tests les remplacent par des stubs ; aucun modèle ni serveur n'est nécessaire.
"""

from __future__ import annotations

import re
from typing import Protocol

from pydantic import ConfigDict, Field, field_validator

from app.rag.text import normalize_query
from app.tools.base import SideEffect, ToolInput, ToolOutput, ToolSpec
from app.tools.resilience import ResiliencePolicy

DEFAULT_MIN_SCORE = 0.5
# Placement de pièces d'une FEN : 8 rangées séparées par des « / ».
_FEN_LIKE = re.compile(r"^([1-8pnbrqkPNBRQK]+/){7}[1-8pnbrqkPNBRQK]+(\s|$)")

DESCRIPTION = """\
Recherche dans la base de connaissances sur les ouvertures (articles Wikipédia indexés
dans Milvus) les passages les plus proches d'une question : idées, plans typiques,
pièges, histoire. Renvoie des extraits avec leur source.

À utiliser : pour expliquer pourquoi on joue un coup ou une ouverture.
Exemple : « Quelle est l'idée de la Sicilienne ? » →
search_opening_knowledge(query="idées principales de la défense sicilienne",
opening="Sicilian Defense")

Ne pas utiliser : pour obtenir des coups ou des statistiques (→ get_opening_theory), ni
une évaluation (→ evaluate_position). Ne pas passer une FEN comme query.
"""


class SearchKnowledgeInput(ToolInput):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"query": "idées principales de la défense sicilienne", "opening": "Sicilian"}
            ]
        }
    )

    query: str = Field(min_length=3, max_length=300, description="Question en langage naturel.")
    opening: str | None = Field(
        default=None,
        max_length=100,
        description="Filtre sur le nom d'ouverture (celui de get_opening_theory).",
    )
    limit: int = Field(default=4, ge=1, le=8)

    @field_validator("query")
    @classmethod
    def _not_a_fen(cls, value: str) -> str:
        if _FEN_LIKE.match(value):
            raise ValueError(
                "query contient une FEN ; poser une question en mots, par exemple "
                "« idées de la partie italienne », et passer le nom de l'ouverture dans opening"
            )
        return value


class Passage(ToolOutput):
    chunk_id: str
    article_id: str
    title: str
    opening: str
    opening_family: str
    section: str
    text: str
    source_url: str
    score: float


class SearchKnowledgeOutput(ToolOutput):
    found: bool
    passages: list[Passage]


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    async def search(
        self, vector: list[float], *, limit: int, opening: str | None
    ) -> list[Passage]: ...


class KnowledgeSearch:
    def __init__(
        self, embedder: Embedder, store: VectorStore, *, min_score: float = DEFAULT_MIN_SCORE
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._min_score = min_score

    async def search(self, params: SearchKnowledgeInput) -> SearchKnowledgeOutput:
        vector = await self._embedder.embed(normalize_query(params.query))
        hits = await self._store.search(vector, limit=params.limit, opening=params.opening)
        relevant = sorted(
            (hit for hit in hits if hit.score >= self._min_score),
            key=lambda hit: hit.score,
            reverse=True,
        )[: params.limit]
        return SearchKnowledgeOutput(found=bool(relevant), passages=relevant)


def build_search_knowledge_tool(search: KnowledgeSearch) -> ToolSpec:
    return ToolSpec(
        name="search_opening_knowledge",
        description=DESCRIPTION,
        input_model=SearchKnowledgeInput,
        output_model=SearchKnowledgeOutput,
        handler=search.search,
        side_effect=SideEffect.READ_EXTERNAL,
        idempotent=True,
        policy=ResiliencePolicy(timeout_s=5.0, max_attempts=2, breaker_threshold=5),
    )
