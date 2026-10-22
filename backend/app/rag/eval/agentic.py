"""Métriques de la recherche agentique (phase 7) : rappel, précision, sauts moyens.

Jeu de référence : Réf-30 (30 questions, un extrait attendu chacune). Pour une question,
la boucle multi-sauts renvoie les passages retenus et le nombre de recherches (sauts).

| Métrique            | Définition                                                            |
|---------------------|-----------------------------------------------------------------------|
| rappel              | part des questions dont l'extrait attendu figure dans un passage retenu|
| précision (stricte) | moyenne, par question, de (passages retenus contenant l'extrait) /    |
|                     | (passages retenus) ; plafonnée par 1/n car un seul extrait est attendu|
| précision (article) | idem, en comptant pertinent tout passage du bon article               |
| sauts moyens        | nombre moyen de recherches par question                               |

Objectifs de la consigne : rappel > 0,8 et moins de 3 sauts par question.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.rag.eval.datasets import RefQuestion
from app.rag.text import contains_excerpt

RECALL_TARGET = 0.8
HOPS_TARGET = 3


@dataclass
class QuestionOutcome:
    id: str
    found: bool
    retained: int
    relevant: int  # passages contenant l'extrait attendu
    same_article: int  # passages du bon article
    hops: int
    context_tokens: int
    stop_reason: str
    sub_questions: list[str] = field(default_factory=list)
    reformulations: int = 0

    @property
    def precision(self) -> float:
        return self.relevant / self.retained if self.retained else 0.0

    @property
    def article_precision(self) -> float:
        return self.same_article / self.retained if self.retained else 0.0


def assess(
    question: RefQuestion,
    passages: Sequence[dict[str, Any]],
    *,
    hops: int,
    context_tokens: int,
    stop_reason: str,
    sub_questions: Sequence[str] = (),
    reformulations: int = 0,
) -> QuestionOutcome:
    expected = question.expected
    relevant = sum(
        any(
            p.get("article_id") == e.article_id
            and contains_excerpt(str(p.get("text", "")), e.excerpt)
            for e in expected
        )
        for p in passages
    )
    same_article = sum(p.get("article_id") in {e.article_id for e in expected} for p in passages)
    return QuestionOutcome(
        id=question.id,
        found=relevant > 0,
        retained=len(passages),
        relevant=relevant,
        same_article=same_article,
        hops=hops,
        context_tokens=context_tokens,
        stop_reason=stop_reason,
        sub_questions=list(sub_questions),
        reformulations=reformulations,
    )


@dataclass
class AgenticReport:
    outcomes: list[QuestionOutcome]

    @property
    def recall(self) -> float:
        return _mean(o.found for o in self.outcomes)

    @property
    def precision(self) -> float:
        return _mean(o.precision for o in self.outcomes)

    @property
    def article_precision(self) -> float:
        return _mean(o.article_precision for o in self.outcomes)

    @property
    def mean_hops(self) -> float:
        return _mean(o.hops for o in self.outcomes)

    @property
    def max_hops(self) -> int:
        return max((o.hops for o in self.outcomes), default=0)

    @property
    def mean_context_tokens(self) -> float:
        return _mean(o.context_tokens for o in self.outcomes)

    @property
    def missed(self) -> list[str]:
        return [o.id for o in self.outcomes if not o.found]

    @property
    def stop_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for o in self.outcomes:
            counts[o.stop_reason] = counts.get(o.stop_reason, 0) + 1
        return counts

    @property
    def meets_targets(self) -> dict[str, bool]:
        return {
            f"recall > {RECALL_TARGET}": self.recall > RECALL_TARGET,
            f"mean hops < {HOPS_TARGET}": self.mean_hops < HOPS_TARGET,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "recall": round(self.recall, 3),
            "precision": round(self.precision, 3),
            "article_precision": round(self.article_precision, 3),
            "mean_hops": round(self.mean_hops, 2),
            "max_hops": self.max_hops,
            "mean_context_tokens": round(self.mean_context_tokens),
            "stop_reasons": self.stop_reasons,
            "missed": self.missed,
            "targets": self.meets_targets,
            "questions": [o.__dict__ for o in self.outcomes],
        }


def _mean(values) -> float:
    values = [float(v) for v in values]
    return statistics.fmean(values) if values else 0.0
