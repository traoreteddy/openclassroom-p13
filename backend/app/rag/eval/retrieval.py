"""Évaluation de la recherche (recall@k, MRR, precision@k) et calibration du seuil.

Réutilisé par toutes les phases : seule la fonction de recherche change (découpage,
enrichissement, agent…). Un chunk est pertinent s'il contient l'extrait attendu
(`contains_excerpt`), ce qui rend la mesure indépendante du découpage.
"""

from __future__ import annotations

import statistics
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from app.rag.eval.datasets import RefQuestion
from app.rag.text import contains_excerpt


class Hit(Protocol):
    text: str
    article_id: str
    score: float


Search = Callable[[str, int, str | None], Awaitable[Sequence[Hit]]]


@dataclass
class QuestionResult:
    id: str
    rank: int | None  # rang (1 = premier) du premier chunk pertinent, None si absent
    relevant_in_top_k: int
    top_score: float


@dataclass
class RetrievalReport:
    k: int
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def recall(self) -> float:
        """recall@k : une seule source attendue par question → part des questions trouvées."""
        return _mean(r.rank is not None and r.rank <= self.k for r in self.results)

    @property
    def mrr(self) -> float:
        return _mean(1 / r.rank if r.rank else 0.0 for r in self.results)

    @property
    def precision(self) -> float:
        return _mean(r.relevant_in_top_k / self.k for r in self.results)

    def recall_at(self, k: int) -> float:
        return _mean(r.rank is not None and r.rank <= k for r in self.results)

    @property
    def missed(self) -> list[str]:
        return [r.id for r in self.results if r.rank is None or r.rank > self.k]


def is_relevant(hit: Hit, question: RefQuestion) -> bool:
    return any(
        hit.article_id == passage.article_id and contains_excerpt(hit.text, passage.excerpt)
        for passage in question.expected
    )


async def evaluate(
    questions: Sequence[RefQuestion], search: Search, *, k: int = 5, use_filter: bool = False
) -> RetrievalReport:
    report = RetrievalReport(k=k)
    for question in questions:
        opening = question.opening_family if use_filter else None
        hits = await search(question.question, k, opening)
        relevant = [is_relevant(hit, question) for hit in hits]
        rank = relevant.index(True) + 1 if True in relevant else None
        report.results.append(
            QuestionResult(
                id=question.id,
                rank=rank,
                relevant_in_top_k=sum(relevant[:k]),
                top_score=hits[0].score if hits else 0.0,
            )
        )
    return report


@dataclass
class Threshold:
    value: float
    kept_in_corpus: float  # part des questions du corpus dont le meilleur score passe
    rejected_out_of_corpus: float  # part des questions hors corpus correctement écartées

    @property
    def balanced_accuracy(self) -> float:
        return (self.kept_in_corpus + self.rejected_out_of_corpus) / 2


def calibrate(in_corpus: Sequence[float], out_of_corpus: Sequence[float]) -> list[Threshold]:
    """Évalue des seuils de 0,30 à 0,70 sur les meilleurs scores des deux populations."""
    thresholds = []
    for step in range(30, 71, 5):
        value = step / 100
        thresholds.append(
            Threshold(
                value=value,
                kept_in_corpus=_mean(score >= value for score in in_corpus),
                rejected_out_of_corpus=_mean(score < value for score in out_of_corpus),
            )
        )
    return thresholds


def _mean(values) -> float:
    values = [float(v) for v in values]
    return statistics.fmean(values) if values else 0.0
