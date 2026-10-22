"""Validation des jeux de référence (phase 1, étape 1.8).

Vérifie la cohérence des 4 jeux avec le corpus et le catalogue d'outils. Utilisé par
`scripts/validate_eval.py` et par les tests (le dépôt ne peut pas contenir un jeu
incohérent sans faire échouer la suite).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import chess

from app.rag.corpus import Article
from app.rag.eval.datasets import (
    DecisionQuestion,
    ExpectedPassage,
    MultiHopQuestion,
    RefQuestion,
    ToolTask,
    load_jsonl,
)
from app.rag.text import contains_excerpt

EXPECTED_SIZES = {"ref30": 30, "decision50": 50, "tasks30": 30, "multi10": 10}
EXPECTED_DECISION = {"dans_corpus": 30, "echecs_hors_corpus": 8, "autre_outil": 6, "hors_sujet": 6}


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class EvalSets:
    ref30: list[RefQuestion]
    decision50: list[DecisionQuestion]
    tasks30: list[ToolTask]
    multi10: list[MultiHopQuestion]

    @classmethod
    def load(cls, directory: Path) -> EvalSets:
        return cls(
            ref30=load_jsonl(RefQuestion, directory / "ref30.jsonl"),
            decision50=load_jsonl(DecisionQuestion, directory / "decision50.jsonl"),
            tasks30=load_jsonl(ToolTask, directory / "tasks30.jsonl"),
            multi10=load_jsonl(MultiHopQuestion, directory / "multi10.jsonl"),
        )


def validate(sets: EvalSets, articles: list[Article], tool_names: set[str]) -> Report:
    report = Report()
    by_id = {article.id: article for article in articles}

    def check_passage(where: str, passage: ExpectedPassage) -> None:
        article = by_id.get(passage.article_id)
        if article is None:
            report.errors.append(f"{where} : article inconnu {passage.article_id}")
        elif not contains_excerpt(article.text, passage.excerpt):
            report.errors.append(f"{where} : extrait absent de {passage.article_id}")

    for name, size in EXPECTED_SIZES.items():
        items = getattr(sets, name)
        if len(items) != size:
            report.errors.append(f"{name} : {len(items)} éléments au lieu de {size}")
        duplicates = [i for i, n in Counter(item.id for item in items).items() if n > 1]
        if duplicates:
            report.errors.append(f"{name} : identifiants en double {duplicates}")

    for question in sets.ref30:
        for passage in question.expected:
            check_passage(question.id, passage)

    categories = Counter(item.category for item in sets.decision50)
    if dict(categories) != EXPECTED_DECISION:
        report.errors.append(f"decision50 : répartition {dict(categories)}")
    for item in sets.decision50:
        if item.search_useful != (item.category == "dans_corpus"):
            report.errors.append(f"{item.id} : search_useful incohérent avec {item.category}")

    for task in sets.tasks30:
        if task.expected_tool is not None and task.expected_tool not in tool_names:
            report.errors.append(f"{task.id} : outil inconnu {task.expected_tool}")
        fen = task.expected_arguments.get("fen")
        if fen is not None and not chess.Board(str(fen)).is_valid():
            report.errors.append(f"{task.id} : FEN invalide")

    for question in sets.multi10:
        for position, hop in enumerate(question.hops):
            check_passage(f"{question.id} saut {position + 1}", hop.expected)
            expected_dependency = position - 1 if position else None
            if hop.depends_on != expected_dependency:
                report.errors.append(f"{question.id} saut {position + 1} : dépendance incohérente")

    report.stats = {
        "ref30_par_article": dict(Counter(q.expected[0].article_id for q in sets.ref30)),
        "decision50": dict(categories),
        "tasks30": dict(Counter(t.expected_tool or "aucun" for t in sets.tasks30)),
        "multi10_entre_articles": sum(
            len({hop.expected.article_id for hop in q.hops}) > 1 for q in sets.multi10
        ),
    }
    return report
