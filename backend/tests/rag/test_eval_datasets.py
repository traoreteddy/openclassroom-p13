"""Jeux de référence : les fichiers versionnés sont cohérents, et le validateur détecte
les incohérences."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.corpus import load_corpus
from app.rag.eval.datasets import (
    EVAL_DIR,
    DecisionQuestion,
    ExpectedPassage,
    RefQuestion,
    load_jsonl,
    save_jsonl,
)
from app.rag.eval.validate import EvalSets, validate
from scripts.validate_eval import TOOL_NAMES

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(CORPUS_DIR)


@pytest.fixture
def sets():
    return EvalSets.load(EVAL_DIR)


def test_versioned_datasets_are_consistent(sets, corpus):
    report = validate(sets, corpus, TOOL_NAMES)

    assert report.errors == []
    assert report.stats["multi10_entre_articles"] == 3


def test_tool_names_match_the_real_catalog(make_registry):
    assert set(make_registry().names) == TOOL_NAMES


def test_every_article_has_a_reference_question(sets, corpus):
    covered = {q.expected[0].article_id for q in sets.ref30}

    assert covered == {article.id for article in corpus}


def test_validator_detects_a_missing_excerpt(sets, corpus):
    sets.ref30[0].expected[0] = ExpectedPassage(
        article_id=sets.ref30[0].expected[0].article_id,
        excerpt="Cette phrase n'existe dans aucun article du corpus.",
    )

    report = validate(sets, corpus, TOOL_NAMES)

    assert any("ref-01 : extrait absent" in error for error in report.errors)


def test_validator_detects_size_duplicates_and_unknown_article(sets, corpus):
    sets.ref30.append(sets.ref30[0])
    sets.multi10[0].hops[0].expected.article_id = "article-inconnu"

    report = validate(sets, corpus, TOOL_NAMES)

    assert any("31 éléments au lieu de 30" in e for e in report.errors)
    assert any("identifiants en double ['ref-01']" in e for e in report.errors)
    assert any("article inconnu article-inconnu" in e for e in report.errors)


def test_validator_detects_inconsistent_decisions_and_tasks(sets, corpus):
    sets.decision50[0].search_useful = False
    sets.decision50[-1].category = "dans_corpus"
    sets.tasks30[0].expected_tool = "outil_imaginaire"
    sets.tasks30[1].expected_arguments["fen"] = "8/8/8/8/8/8/8/K7 w - - 0 1"
    sets.multi10[0].hops[1].depends_on = None

    report = validate(sets, corpus, TOOL_NAMES)
    text = "\n".join(report.errors)

    assert "dec-01 : search_useful incohérent" in text
    assert "decision50 : répartition" in text
    assert "outil inconnu outil_imaginaire" in text
    assert "task-02 : FEN invalide" in text
    assert "multi-01 saut 2 : dépendance incohérente" in text


def test_jsonl_roundtrip(tmp_path):
    items = [
        DecisionQuestion(id="d1", question="Bonjour ?", category="hors_sujet", search_useful=False)
    ]

    save_jsonl(items, tmp_path / "x.jsonl")

    assert load_jsonl(DecisionQuestion, tmp_path / "x.jsonl") == items


def test_unknown_fields_are_refused():
    with pytest.raises(ValueError):
        RefQuestion.model_validate(
            {
                "id": "r",
                "question": "q ?",
                "opening_family": None,
                "expected": [],
                "answer": "a",
                "provenance": "inventee",
            }
        )
