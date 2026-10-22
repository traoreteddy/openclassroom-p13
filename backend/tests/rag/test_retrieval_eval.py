"""Métriques de recherche (recall@k, MRR, precision@k) et calibration du seuil."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.rag.eval.datasets import ExpectedPassage, RefQuestion
from app.rag.eval.retrieval import calibrate, evaluate, is_relevant


@dataclass
class FakeHit:
    text: str
    article_id: str
    score: float


def question(
    qid: str, excerpt: str, article_id: str = "a", family: str | None = "Fam"
) -> RefQuestion:
    return RefQuestion(
        id=qid,
        question=f"Question {qid} ?",
        opening_family=family,
        expected=[ExpectedPassage(article_id=article_id, excerpt=excerpt)],
        answer="réponse",
        provenance="redige+verifie",
    )


GOOD = "la bonne réponse se trouve ici, mot pour mot"
QUESTIONS = [question("q1", GOOD), question("q2", GOOD), question("q3", GOOD), question("q4", GOOD)]
NOISE = FakeHit("texte sans rapport avec la question posée", "a", 0.4)
RIGHT = FakeHit(f"Début. {GOOD.capitalize()}. Fin.", "a", 0.9)

# Rang du bon chunk par question : 1, 3, 6 (hors top-5), absent
RANKINGS = {
    "Question q1 ?": [RIGHT, NOISE, NOISE, NOISE, NOISE],
    "Question q2 ?": [NOISE, NOISE, RIGHT, NOISE, NOISE],
    "Question q3 ?": [NOISE] * 5 + [RIGHT],
    "Question q4 ?": [NOISE] * 6,
}


async def fake_search(query: str, k: int, opening: str | None):
    calls.append(opening)
    return RANKINGS[query][: max(k, 6)]


calls: list[str | None] = []


async def test_metrics():
    report = await evaluate(QUESTIONS, fake_search, k=5)

    assert [r.rank for r in report.results] == [1, 3, 6, None]
    assert report.recall == 0.5  # q1 et q2 dans le top-5
    assert report.recall_at(1) == 0.25
    assert report.recall_at(10) == 0.75
    assert report.mrr == pytest.approx((1 + 1 / 3 + 1 / 6 + 0) / 4)
    assert report.precision == pytest.approx((1 + 1 + 0 + 0) / 5 / 4)
    assert report.missed == ["q3", "q4"]
    assert report.results[0].top_score == 0.9


async def test_filter_uses_the_question_opening_family():
    calls.clear()

    await evaluate(QUESTIONS[:1], fake_search, use_filter=True)
    await evaluate(QUESTIONS[:1], fake_search)

    assert calls == ["Fam", None]


async def test_no_hits():
    async def empty(query, k, opening):
        return []

    report = await evaluate(QUESTIONS[:1], empty)

    assert report.results[0].rank is None
    assert report.results[0].top_score == 0.0


def test_relevance_requires_the_right_article():
    q = question("q", GOOD, article_id="a")

    assert is_relevant(RIGHT, q)
    assert not is_relevant(FakeHit(RIGHT.text, "autre-article", 0.9), q)


def test_calibrate():
    thresholds = calibrate(in_corpus=[0.8, 0.7, 0.52], out_of_corpus=[0.2, 0.6])
    by_value = {t.value: t for t in thresholds}

    assert by_value[0.5].kept_in_corpus == 1.0
    assert by_value[0.5].rejected_out_of_corpus == 0.5
    assert by_value[0.65].kept_in_corpus == pytest.approx(2 / 3)
    assert by_value[0.65].rejected_out_of_corpus == 1.0
    assert by_value[0.5].balanced_accuracy == 0.75
    assert [t.value for t in thresholds][:2] == [0.3, 0.35]


def test_empty_report_is_zero():
    from app.rag.eval.retrieval import RetrievalReport

    assert RetrievalReport(k=5).recall == 0.0
