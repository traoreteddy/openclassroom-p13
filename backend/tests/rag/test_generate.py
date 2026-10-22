"""Génération des questions candidates : Ollama simulé avec respx."""

from __future__ import annotations

import json

import httpx
import pytest

from app.rag.eval.generate import (
    GENERATION_SCHEMA,
    GeneratedQuestion,
    OllamaChat,
    check,
    generate_candidate,
)
from app.rag.eval.passages import CandidatePassage

OLLAMA = "http://ollama.test"
PASSAGE = CandidatePassage(
    article_id="gambit-evans",
    section_path=("Gambit Evans", "Histoire"),
    text=(
        "Le gambit porte le nom du capitaine de navire gallois William Davies Evans, le "
        "premier joueur à l'avoir pratiqué. Le gambit devint populaire peu après."
    ),
)
GOOD = {
    "question": "D'où vient le nom du gambit Evans ?",
    "excerpt": "Le gambit porte le nom du capitaine de navire gallois William Davies Evans",
    "answer": "Du capitaine William Davies Evans.",
}
PARAPHRASED = {**GOOD, "excerpt": "Le gambit doit son nom à un capitaine gallois appelé Evans"}


def ollama_reply(content: dict) -> httpx.Response:
    return httpx.Response(200, json={"message": {"content": json.dumps(content)}})


@pytest.fixture
def llm(http_client):
    return OllamaChat(http_client, base_url=OLLAMA, model="phi4-mini")


async def test_valid_candidate_on_first_attempt(llm, respx_mock):
    route = respx_mock.post(f"{OLLAMA}/api/chat").mock(return_value=ollama_reply(GOOD))

    candidate = await generate_candidate(llm, PASSAGE)

    assert candidate.status == "valide"
    assert candidate.attempts == 1
    assert candidate.breadcrumb == "Gambit Evans > Histoire"
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "phi4-mini"
    assert body["format"] == GENERATION_SCHEMA  # sortie structurée imposée
    assert body["options"]["temperature"] == 0.0
    assert "Gambit Evans > Histoire" in body["messages"][1]["content"]


async def test_paraphrased_excerpt_is_retried_with_feedback(llm, respx_mock):
    route = respx_mock.post(f"{OLLAMA}/api/chat").mock(
        side_effect=[ollama_reply(PARAPHRASED), ollama_reply(GOOD)]
    )

    candidate = await generate_candidate(llm, PASSAGE)

    assert candidate.status == "valide"
    assert candidate.attempts == 2
    retry = json.loads(route.calls.last.request.content)
    assert "refusée" in retry["messages"][1]["content"]
    assert retry["options"]["temperature"] == 0.4


async def test_candidate_rejected_after_two_failures(llm, respx_mock):
    respx_mock.post(f"{OLLAMA}/api/chat").mock(return_value=ollama_reply(PARAPHRASED))

    candidate = await generate_candidate(llm, PASSAGE)

    assert candidate.status == "rejete"
    assert "mot pour mot" in candidate.rejection
    assert candidate.excerpt == PARAPHRASED["excerpt"]  # gardé pour la relecture


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"excerpt": "Evans"}, "trop court"),
        ({"excerpt": PARAPHRASED["excerpt"]}, "mot pour mot"),
        ({"question": "Le nom ?"}, "vraie question"),
        ({"question": "Donne le nom du gambit Evans."}, "vraie question"),
    ],
)
def test_check_reasons(override, reason):
    assert reason in check(GeneratedQuestion(**{**GOOD, **override}), PASSAGE)


def test_check_accepts_good_candidate():
    assert check(GeneratedQuestion(**GOOD), PASSAGE) is None


async def test_ollama_error_is_raised(llm, respx_mock):
    respx_mock.post(f"{OLLAMA}/api/chat").respond(500)

    with pytest.raises(httpx.HTTPStatusError):
        await generate_candidate(llm, PASSAGE)
