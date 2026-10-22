"""Métriques de la recherche agentique : rappel, précisions, sauts, objectifs."""

from __future__ import annotations

from app.rag.eval.agentic import AgenticReport, assess
from app.rag.eval.datasets import ExpectedPassage, RefQuestion


def question(qid: str = "ref-01", excerpt: str = "la bonne réponse se trouve ici"):
    return RefQuestion(
        id=qid,
        question="Q ?",
        opening_family=None,
        expected=[ExpectedPassage(article_id="a", excerpt=excerpt)],
        answer="",
        provenance="redige+verifie",
    )


RIGHT = {"article_id": "a", "text": "Début. La bonne réponse se trouve ici. Fin."}
SAME_ARTICLE = {"article_id": "a", "text": "Autre passage du même article."}
OTHER = {"article_id": "b", "text": "Sans rapport."}


def test_assess_counts_relevant_and_same_article_passages():
    outcome = assess(
        question(),
        [RIGHT, SAME_ARTICLE, OTHER],
        hops=2,
        context_tokens=900,
        stop_reason="enough",
        sub_questions=["q1", "q2"],
    )

    assert outcome.found is True
    assert outcome.retained == 3 and outcome.relevant == 1 and outcome.same_article == 2
    assert outcome.precision == 1 / 3
    assert outcome.article_precision == 2 / 3
    assert outcome.hops == 2


def test_assess_without_passages():
    outcome = assess(question(), [], hops=3, context_tokens=0, stop_reason="max_searches")

    assert outcome.found is False
    assert outcome.precision == 0.0 and outcome.article_precision == 0.0


def test_report_metrics_and_targets():
    outcomes = [
        assess(question("r1"), [RIGHT], hops=1, context_tokens=300, stop_reason="enough"),
        assess(question("r2"), [RIGHT, OTHER], hops=2, context_tokens=600, stop_reason="enough"),
        assess(question("r3"), [OTHER], hops=3, context_tokens=200, stop_reason="max_searches"),
    ]

    report = AgenticReport(outcomes)

    assert report.recall == 2 / 3
    assert report.precision == (1 + 0.5 + 0) / 3
    assert report.mean_hops == 2.0 and report.max_hops == 3
    assert report.missed == ["r3"]
    assert report.stop_reasons == {"enough": 2, "max_searches": 1}
    assert report.meets_targets == {"recall > 0.8": False, "mean hops < 3": True}
    assert report.to_dict()["mean_context_tokens"] == 367


def test_empty_report():
    report = AgenticReport([])

    assert report.recall == 0.0 and report.mean_hops == 0.0 and report.max_hops == 0
