"""Banc de comparaison des découpages (phase 3).

Méthode, identique pour toutes les stratégies :
1. découper les 12 articles ;
2. encoder les chunks (cache disque : un texte n'est jamais encodé deux fois) ;
3. recherche **exacte** en mémoire (produit scalaire de vecteurs normalisés) : on compare
   des découpages, pas des index ; avec quelques centaines de chunks, c'est instantané et
   sans l'approximation de HNSW ;
4. mesurer sur 50 requêtes (Réf-30 + les 20 sous-questions de Multi-10).
"""

from __future__ import annotations

import hashlib
import itertools
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.rag.chunking import Chunk, TokenCounter, prepare_document
from app.rag.corpus import Article
from app.rag.eval.datasets import (
    EVAL_DIR,
    MultiHopQuestion,
    RefQuestion,
    load_jsonl,
)
from app.rag.eval.retrieval import RetrievalReport, evaluate
from app.rag.eval.strategies import HierarchicalChunk, sentences
from app.rag.milvus_store import opening_filter
from app.rag.text import contains_excerpt

CACHE_DIR = EVAL_DIR / "cache"


class CachedEmbedder:
    """Encodeur synchrone avec cache disque, utilisable aussi comme `Embeddings` LangChain."""

    def __init__(self, model: Any, cache_dir: Path | None = CACHE_DIR) -> None:
        self._model = model
        self._cache_dir = cache_dir
        self._memory: dict[str, np.ndarray] = {}
        self.encoded = 0  # textes réellement encodés (hors cache)

    def encode(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray:
        keys = [self._key(text, query) for text in texts]
        missing = [i for i, key in enumerate(keys) if self._lookup(key) is None]
        if missing:
            vectors = self._model.encode(
                [texts[i] for i in missing],
                prompt_name="query" if query else None,
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=16,
                show_progress_bar=False,
            )
            for i, vector in zip(missing, vectors, strict=True):
                self._store(keys[i], np.asarray(vector, dtype=np.float32))
            self.encoded += len(missing)
        return np.stack([self._lookup(key) for key in keys])

    # Interface `Embeddings` de LangChain (utilisée par SemanticChunker)
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.encode([text], query=True)[0].tolist()

    def _key(self, text: str, query: bool) -> str:
        return hashlib.sha1(f"{'q' if query else 'd'}\x00{text}".encode()).hexdigest()

    def _lookup(self, key: str) -> np.ndarray | None:
        if key in self._memory:
            return self._memory[key]
        if self._cache_dir is None:  # cache en mémoire seulement
            return None
        path = self._cache_dir / key[:2] / f"{key}.npy"
        if path.exists():
            self._memory[key] = np.load(path)
            return self._memory[key]
        return None

    def _store(self, key: str, vector: np.ndarray) -> None:
        self._memory[key] = vector
        if self._cache_dir is None:
            return
        path = self._cache_dir / key[:2] / f"{key}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vector)


@dataclass
class MemoryHit:
    chunk_id: str
    article_id: str
    text: str
    score: float


class MemoryIndex:
    """Recherche exacte. Avec des parents (hiérarchique), renvoie les parents distincts."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        parents: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.vectors = vectors
        self.parents = parents or {}  # id enfant → (id parent, texte parent)

    def search(self, query_vector: np.ndarray, k: int, opening: str | None) -> list[MemoryHit]:
        scores = self.vectors @ query_vector
        allowed = _filter(self.chunks, opening)
        order = [i for i in np.argsort(-scores) if allowed[i]]
        hits: list[MemoryHit] = []
        seen: set[str] = set()
        for i in order:
            chunk = self.chunks[i]
            if chunk.id in self.parents:
                parent_id, parent_text = self.parents[chunk.id]
                if parent_id in seen:
                    continue
                seen.add(parent_id)
                hits.append(MemoryHit(parent_id, chunk.article_id, parent_text, float(scores[i])))
            else:
                hits.append(MemoryHit(chunk.id, chunk.article_id, chunk.text, float(scores[i])))
            if len(hits) == k:
                break
        return hits


def _filter(chunks: Sequence[Chunk], opening: str | None) -> list[bool]:
    if opening_filter(opening) is None:
        return [True] * len(chunks)
    family = opening.split(":", 1)[0].strip()  # type: ignore[union-attr]
    return [c.opening == opening or c.opening_family == family for c in chunks]


def benchmark_queries() -> list[RefQuestion]:
    """Réf-30 + les sous-questions de Multi-10 (chacune avec son extrait) : 50 requêtes."""
    questions = load_jsonl(RefQuestion, EVAL_DIR / "ref30.jsonl")
    for multi in load_jsonl(MultiHopQuestion, EVAL_DIR / "multi10.jsonl"):
        for position, hop in enumerate(multi.hops, start=1):
            questions.append(
                RefQuestion(
                    id=f"{multi.id}-saut{position}",
                    question=hop.sub_question,
                    opening_family=None,
                    expected=[hop.expected],
                    answer="",
                    provenance=multi.provenance,
                )
            )
    return questions


@dataclass
class StrategyResult:
    name: str
    chunks: int
    tokens_mean: float
    tokens_median: float
    tokens_stdev: float
    tokens_min: int
    tokens_max: int
    redundancy: float
    reachable: float  # part des requêtes dont l'extrait est entier dans au moins un chunk
    recall_ref30: float
    recall_all: float
    mrr_all: float
    context_tokens: float  # tokens renvoyés à l'agent pour un top-5 (moyenne)
    recall_budget: float  # recall à budget égal : chunks pris dans l'ordre jusqu'à N tokens
    missed: list[str]
    cohesion: float | None = None  # similarité moyenne phrase ↔ centre de son chunk
    cut_similarity: float | None = (
        None  # similarité entre les phrases de part et d'autre d'une coupure
    )


async def run_strategy(
    name: str,
    items: Sequence[Chunk] | Sequence[HierarchicalChunk],
    articles: Sequence[Article],
    embedder: CachedEmbedder,
    count: TokenCounter,
    queries: Sequence[RefQuestion],
    *,
    k: int = 5,
    coherence: bool = False,
    budget_tokens: int = 2000,
) -> StrategyResult:
    hierarchical = bool(items) and isinstance(items[0], HierarchicalChunk)
    chunks = [i.child for i in items] if hierarchical else list(items)  # type: ignore[union-attr]
    parents = (
        {i.child.id: (i.parent_id, i.parent) for i in items}  # type: ignore[union-attr]
        if hierarchical
        else None
    )
    index = MemoryIndex(chunks, embedder.encode([c.text for c in chunks]), parents)

    async def search(query: str, top: int, opening: str | None):
        return index.search(embedder.encode([query], query=True)[0], top, opening)

    returned_tokens: list[int] = []

    async def counted_search(query: str, top: int, opening: str | None):
        hits = await search(query, top, opening)
        returned_tokens.append(sum(count(hit.text) for hit in hits[:k]))
        return hits

    report: RetrievalReport = await evaluate(queries, counted_search, k=k)
    recall_budget = statistics.fmean(
        [found_within_budget(index, embedder, count, q, budget_tokens) for q in queries]
    )
    texts = [i.parent for i in items] if hierarchical else [c.text for c in chunks]  # type: ignore[union-attr]
    reachable = statistics.fmean(
        any(contains_excerpt(text, passage.excerpt) for text in texts for passage in q.expected)
        for q in queries
    )
    ref_ids = {q.id for q in queries if q.id.startswith("ref-")}
    ref_results = [r for r in report.results if r.id in ref_ids]
    tokens = [c.token_count for c in chunks]
    document_tokens = sum(count(prepare_document(a)) for a in articles)
    return StrategyResult(
        name=name,
        chunks=len(chunks),
        tokens_mean=round(statistics.fmean(tokens), 1),
        tokens_median=statistics.median(tokens),
        tokens_stdev=round(statistics.pstdev(tokens), 1),
        tokens_min=min(tokens),
        tokens_max=max(tokens),
        redundancy=round(max(0.0, 1 - document_tokens / sum(tokens)), 3),
        reachable=round(reachable, 3),
        recall_ref30=round(
            statistics.fmean(r.rank is not None and r.rank <= k for r in ref_results), 3
        ),
        recall_all=round(report.recall, 3),
        mrr_all=round(report.mrr, 3),
        context_tokens=round(statistics.fmean(returned_tokens), 0),
        recall_budget=round(recall_budget, 3),
        missed=report.missed,
        cohesion=round(chunk_cohesion(chunks, embedder), 3) if coherence else None,
        cut_similarity=round(cut_similarity(chunks, embedder), 3) if coherence else None,
    )


def found_within_budget(
    index: MemoryIndex,
    embedder: CachedEmbedder,
    count: TokenCounter,
    question: RefQuestion,
    budget: int,
) -> bool:
    """Prend les meilleurs résultats dans l'ordre tant que le total reste sous `budget` tokens.

    Compare les stratégies à quantité de texte égale envoyée à l'agent : un gros chunk
    « coûte » plus qu'un petit.
    """
    from app.rag.eval.retrieval import is_relevant

    used = 0
    for hit in index.search(embedder.encode([question.question], query=True)[0], 60, None):
        used += count(hit.text)
        if used > budget:
            return False
        if is_relevant(hit, question):
            return True
    return False


def chunk_cohesion(chunks: Sequence[Chunk], embedder: CachedEmbedder) -> float:
    """Similarité moyenne entre chaque phrase et le centre (moyenne) des phrases de son chunk.

    Élevée = le chunk parle d'un seul sujet. Les chunks d'une seule phrase sont ignorés.
    """
    values: list[float] = []
    for chunk in chunks:
        parts = sentences(chunk.text)
        if len(parts) < 2:
            continue
        vectors = embedder.encode(parts)
        centre = vectors.mean(axis=0)
        centre /= np.linalg.norm(centre)
        values.extend(float(v @ centre) for v in vectors)
    return statistics.fmean(values) if values else 0.0


def cut_similarity(chunks: Sequence[Chunk], embedder: CachedEmbedder) -> float:
    """Similarité entre la dernière phrase d'un chunk et la première du suivant (même article).

    Basse = les coupures tombent là où le sujet change, ce que vise le découpage sémantique.
    """
    values: list[float] = []
    for before, after in itertools.pairwise(chunks):
        if before.article_id != after.article_id:
            continue
        last, first = sentences(before.text)[-1:], sentences(after.text)[:1]
        if last and first:
            a, b = embedder.encode([last[0], first[0]])
            values.append(float(a @ b))
    return statistics.fmean(values) if values else 0.0
