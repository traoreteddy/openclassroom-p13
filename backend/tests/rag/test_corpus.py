"""Corpus Wikipédia : récupération (API simulée), conversions, sauvegarde."""

from __future__ import annotations

import httpx
import pytest

from app.rag.corpus import (
    ARTICLES,
    ArticleRef,
    WikipediaClient,
    clean_html,
    load_corpus,
    save_article,
    slugify,
    wiki_to_markdown,
)
from app.tools.errors import ToolUnavailableError

API = "https://wiki.test/w/api.php"
REF = ArticleRef(
    title="Gambit Evans", opening="Italian Game: Evans Gambit", opening_family="Italian Game"
)

QUERY = {
    "query": {
        "pages": [
            {
                "title": "Gambit Evans",
                "extract": (
                    "Le gambit Evans est une ouverture.\n\n== Histoire ==\nInventé par Evans.\n"
                ),
                "revisions": [{"revid": 123, "timestamp": "2026-09-12T10:00:00Z"}],
            }
        ]
    }
}
PARSE = {
    "parse": {
        "text": '<div><style>.x{}</style><h2>Histoire</h2><p>Inventé<sup class="reference">'
        "[1]</sup> par Evans.</p><!-- cache --></div>"
    }
}


def route_wikipedia(respx_mock, *responses):
    return respx_mock.get(API).mock(side_effect=list(responses))


@pytest.fixture
def wikipedia(http_client, sleep):
    return WikipediaClient(http_client, api_url=API, pause_s=0.5, sleep=sleep, contact="test")


async def test_fetch_builds_article_with_metadata(wikipedia, respx_mock):
    route_wikipedia(respx_mock, httpx.Response(200, json=QUERY), httpx.Response(200, json=PARSE))

    article = await wikipedia.fetch(REF)

    assert article.id == "gambit-evans"
    assert article.opening == "Italian Game: Evans Gambit"
    assert article.opening_family == "Italian Game"
    meta = article.metadata
    assert meta.version == 123
    assert meta.date == "2026-09-12T10:00:00Z"
    assert meta.service_emetteur == "wikipedia-fr"
    assert meta.confidentialite == "public"
    assert meta.type_document == "article"
    assert meta.source_url.endswith("oldid=123")
    assert article.markdown == (
        "# Gambit Evans\n\nLe gambit Evans est une ouverture.\n\n## Histoire\nInventé par Evans."
    )
    assert "<style>" not in article.html
    assert "[1]" not in article.html


async def test_requests_follow_wikimedia_policy(wikipedia, respx_mock, sleep):
    route = route_wikipedia(
        respx_mock, httpx.Response(200, json=QUERY), httpx.Response(200, json=PARSE)
    )

    await wikipedia.fetch(REF)

    first, second = (call.request for call in route.calls)
    assert first.headers["User-Agent"] == "P13-FFE-POC/0.1 (test)"
    assert first.url.params["maxlag"] == "5"
    assert first.url.params["titles"] == "Gambit Evans"
    assert second.url.params["oldid"] == "123"  # même révision pour le texte et le HTML
    assert sleep.calls == [0.5, 0.5]  # une pause après chaque requête


async def test_rate_limit_is_retried(wikipedia, respx_mock):
    route = route_wikipedia(
        respx_mock,
        httpx.Response(503),
        httpx.Response(200, json=QUERY),
        httpx.Response(200, json=PARSE),
    )

    article = await wikipedia.fetch(REF)

    assert article.metadata.version == 123
    assert route.call_count == 3


async def test_missing_user_agent_contact_is_reported(wikipedia, respx_mock):
    route_wikipedia(respx_mock, httpx.Response(403))

    with pytest.raises(ToolUnavailableError) as info:
        await wikipedia.fetch(REF)

    assert info.value.code == "auth_required"


async def test_missing_article(wikipedia, respx_mock):
    route_wikipedia(respx_mock, httpx.Response(200, json={"query": {"pages": [{"missing": True}]}}))

    with pytest.raises(LookupError, match="Gambit Evans"):
        await wikipedia.fetch(REF)


@pytest.mark.parametrize(
    ("wiki", "markdown"),
    [
        ("== Histoire ==", "## Histoire"),
        ("=== Variante Najdorf ===", "### Variante Najdorf"),
        ("==== Ligne 6.Fg5 ====", "#### Ligne 6.Fg5"),
        ("Texte avec == au milieu", "Texte avec == au milieu"),
        ("1.e4 e5 2.Cf3", "1.e4 e5 2.Cf3"),
    ],
)
def test_wiki_headings_become_markdown(wiki, markdown):
    assert wiki_to_markdown(wiki) == markdown


@pytest.mark.parametrize(
    ("title", "slug"),
    [
        ("Défense sicilienne", "defense-sicilienne"),
        ("Défense Caro-Kann", "defense-caro-kann"),
        ("Partie espagnole", "partie-espagnole"),
    ],
)
def test_slugify(title, slug):
    assert slugify(title) == slug


def test_clean_html_keeps_content():
    assert (
        clean_html(PARSE["parse"]["text"])
        == "<div><h2>Histoire</h2><p>Inventé par Evans.</p></div>"
    )


def test_article_list_is_consistent():
    slugs = [ref.slug for ref in ARTICLES]
    assert len(ARTICLES) == 12
    assert len(set(slugs)) == len(slugs)
    assert all(ref.opening.startswith(ref.opening_family) or ":" in ref.opening for ref in ARTICLES)


async def test_save_and_load_roundtrip(tmp_path, wikipedia, respx_mock):
    route_wikipedia(respx_mock, httpx.Response(200, json=QUERY), httpx.Response(200, json=PARSE))
    article = await wikipedia.fetch(REF)

    path = save_article(article, tmp_path / "corpus")

    assert path.name == "gambit-evans.json"
    assert load_corpus(tmp_path / "corpus") == [article]


def test_versioned_corpus_is_complete():
    """Le corpus figé dans le dépôt correspond à la liste ARTICLES."""
    from pathlib import Path

    corpus = load_corpus(Path(__file__).resolve().parents[2] / "data" / "corpus")

    assert {a.id for a in corpus} == {ref.slug for ref in ARTICLES}
    assert all(len(a.text) > 1000 for a in corpus)
    assert all(a.metadata.version > 0 for a in corpus)
