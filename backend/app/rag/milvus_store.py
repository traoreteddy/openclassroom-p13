"""Adaptateur Milvus : implémente l'interface `VectorStore` de l'outil de recherche.

- Index HNSW, distance cosinus (vecteurs normalisés) : le score est une similarité.
- Identifiants stables (`stratégie:article:numéro`) : réingérer remplace, sans doublon.
- Métadonnées filtrables en champs scalaires (date, version, service émetteur,
  confidentialité, type de document, ouverture).
- Client créé au premier appel : l'API démarre même si Milvus est arrêté.
- Erreurs traduites dans la taxonomie des outils (Milvus injoignable → erreur
  passagère, réessayée ; collection absente → `not_indexed`).

pymilvus est synchrone : les appels s'exécutent dans un fil séparé (`asyncio.to_thread`).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence
from typing import Any

from app.rag.chunking import Chunk
from app.tools.errors import ToolTransientError, ToolUnavailableError
from app.tools.knowledge import Passage

TEXT_MAX_LENGTH = 16_384
OUTPUT_FIELDS = [
    "text",
    "article_id",
    "title",
    "opening",
    "opening_family",
    "section",
    "source_url",
]
HNSW = {"M": 16, "efConstruction": 200}
SEARCH_EF = 64

ClientFactory = Callable[[], Any]


def opening_filter(opening: str | None) -> str | None:
    """Filtre Milvus pour un nom d'ouverture Lichess.

    « Italian Game: Evans Gambit » trouve l'article du gambit Evans (nom exact) et ceux de
    la famille « Italian Game » (partie avant les deux-points).
    """
    if not opening:
        return None
    family = opening.split(":", 1)[0].strip()
    exact = opening.replace("\\", "\\\\").replace('"', '\\"')
    family = family.replace("\\", "\\\\").replace('"', '\\"')
    return f'opening == "{exact}" or opening_family == "{family}"'


class MilvusVectorStore:
    def __init__(
        self,
        uri: str,
        collection: str,
        *,
        dimension: int = 1024,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.uri = uri
        self.collection = collection
        self.dimension = dimension
        self._factory = client_factory or self._default_factory
        self._client: Any = None
        self._lock = threading.Lock()

    # --- Recherche (utilisée par l'outil) -------------------------------------------------

    async def search(
        self, vector: list[float], *, limit: int, opening: str | None
    ) -> list[Passage]:
        return await self._run(self._search, vector, limit, opening_filter(opening))

    def _search(self, vector: list[float], limit: int, expr: str | None) -> list[Passage]:
        client = self._get_client()
        if not client.has_collection(self.collection):
            raise ToolUnavailableError(
                code="not_indexed",
                message=f"La collection Milvus « {self.collection} » n'existe pas.",
                hint="Lancer l'ingestion : python -m scripts.ingest_corpus",
            )
        results = client.search(
            collection_name=self.collection,
            data=[vector],
            limit=limit,
            filter=expr or "",
            output_fields=OUTPUT_FIELDS,
            search_params={"metric_type": "COSINE", "params": {"ef": max(SEARCH_EF, limit)}},
        )
        return [
            Passage(
                chunk_id=str(hit["id"]),
                score=round(float(hit["distance"]), 4),
                **{key: hit["entity"][key] for key in OUTPUT_FIELDS},
            )
            for hit in results[0]
        ]

    # --- Indexation (utilisée par l'ingestion) --------------------------------------------

    async def recreate(self) -> None:
        await self._run(self._recreate)

    async def ensure_collection(self) -> None:
        await self._run(self._ensure_collection)

    async def replace_article(
        self,
        article_id: str,
        strategy: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[list[float]],
    ) -> None:
        await self._run(self._replace_article, article_id, strategy, chunks, vectors)

    async def count(self) -> int:
        return await self._run(self._count)

    def _recreate(self) -> None:
        client = self._get_client()
        if client.has_collection(self.collection):
            client.drop_collection(self.collection)
        self._create(client)

    def _ensure_collection(self) -> None:
        client = self._get_client()
        if not client.has_collection(self.collection):
            self._create(client)

    def _create(self, client: Any) -> None:
        from pymilvus import DataType, MilvusClient

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dimension)
        schema.add_field("text", DataType.VARCHAR, max_length=TEXT_MAX_LENGTH)
        for name, length in [
            ("article_id", 64),
            ("title", 256),
            ("opening", 256),
            ("opening_family", 128),
            ("section", 512),
            ("strategy", 64),
            ("source_url", 512),
            ("date", 32),
            ("service_emetteur", 64),
            ("confidentialite", 32),
            ("type_document", 32),
        ]:
            schema.add_field(name, DataType.VARCHAR, max_length=length)
        schema.add_field("version", DataType.INT64)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("token_count", DataType.INT64)

        index = client.prepare_index_params()
        index.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE", params=HNSW)
        client.create_collection(collection_name=self.collection, schema=schema, index_params=index)

    def _replace_article(
        self,
        article_id: str,
        strategy: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[list[float]],
    ) -> None:
        client = self._get_client()
        # Supprimer d'abord : si le nouveau découpage produit moins de chunks, les anciens
        # ne doivent pas rester dans l'index.
        client.delete(
            collection_name=self.collection,
            filter=f'article_id == "{article_id}" and strategy == "{strategy}"',
        )
        rows = [row(chunk, vector) for chunk, vector in zip(chunks, vectors, strict=True)]
        if rows:
            client.upsert(collection_name=self.collection, data=rows)

    def _count(self) -> int:
        # `get_collection_stats` compte aussi les lignes supprimées non compactées (un upsert
        # = suppression + insertion) : seule une requête count(*) donne le vrai nombre.
        client = self._get_client()
        result = client.query(
            collection_name=self.collection,
            filter="",
            output_fields=["count(*)"],
            consistency_level="Strong",
        )
        return int(result[0]["count(*)"])

    # --- Plomberie ------------------------------------------------------------------------

    async def _run(self, fn: Callable[..., Any], *args: Any) -> Any:
        try:
            return await asyncio.to_thread(fn, *args)
        except (ToolUnavailableError, ToolTransientError):
            raise
        except Exception as exc:
            if _is_milvus_error(exc):
                raise ToolTransientError(
                    code="vector_store_error",
                    message=f"Milvus a renvoyé une erreur : {exc}"[:300],
                ) from exc
            raise

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is None:
                self._client = self._factory()
            return self._client

    def _default_factory(self) -> Any:
        from pymilvus import MilvusClient

        try:
            return MilvusClient(uri=self.uri, timeout=5)
        except Exception as exc:
            raise ToolTransientError(
                code="network_error", message=f"Milvus injoignable ({self.uri})."
            ) from exc


def row(chunk: Chunk, vector: list[float]) -> dict[str, Any]:
    meta = chunk.metadata
    return {
        "id": chunk.id,
        "vector": vector,
        "text": chunk.text,
        "article_id": chunk.article_id,
        "title": chunk.title,
        "opening": chunk.opening,
        "opening_family": chunk.opening_family,
        "section": chunk.section[:512],
        "strategy": chunk.strategy,
        "source_url": meta.source_url,
        "date": meta.date,
        "service_emetteur": meta.service_emetteur,
        "confidentialite": meta.confidentialite,
        "type_document": meta.type_document,
        "version": meta.version,
        "chunk_index": chunk.chunk_index,
        "token_count": chunk.token_count,
    }


def _is_milvus_error(exc: Exception) -> bool:
    from pymilvus.exceptions import MilvusException

    return isinstance(exc, MilvusException | ConnectionError | TimeoutError)
