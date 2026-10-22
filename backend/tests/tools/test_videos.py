"""Outil find_opening_videos : requêtes combinées, filtre qualité, cache, quota.

API YouTube simulée avec respx et de vraies réponses enregistrées (`tests/fixtures/youtube/`).
"""

from __future__ import annotations

import httpx
import pytest

from app.tools.videos import (
    FindVideosInput,
    build_queries,
    parse_duration,
    seconds_until_reset,
)
from tests.conftest import YOUTUBE_URL, load_fixture

SEARCH = f"{YOUTUBE_URL}/search"
VIDEOS = f"{YOUTUBE_URL}/videos"
ITALIAN = {"opening": "Italian Game"}
FREE = {"query": "partie italienne explication"}


@pytest.fixture
def registry(make_registry):
    return make_registry()


@pytest.fixture
def youtube(respx_mock):
    """Les deux recherches (FR puis EN) et la lecture des détails, réponses réelles."""
    search = respx_mock.get(SEARCH).mock(
        side_effect=[
            httpx.Response(200, json=load_fixture("youtube/search_italian_fr.json")),
            httpx.Response(200, json=load_fixture("youtube/search_italian_en.json")),
        ]
    )
    details = respx_mock.get(VIDEOS).respond(json=load_fixture("youtube/videos_italian.json"))
    return search, details


# --- Requêtes ---------------------------------------------------------------------------------


def test_queries_combine_french_and_english_names_with_keywords():
    assert build_queries(FindVideosInput(opening="Italian Game")) == [
        "partie italienne échecs ouverture explication",
        "Italian Game chess opening tutorial",
    ]
    # variante connue : nom français de la variante ; inconnue : nom de la famille
    assert build_queries(FindVideosInput(opening="Italian Game: Evans Gambit"))[0].startswith(
        "gambit Evans"
    )
    assert build_queries(FindVideosInput(opening="Sicilian Defense: Najdorf"))[0].startswith(
        "défense sicilienne"
    )
    # ouverture hors dictionnaire : nom anglais avec les mots-clés français
    assert build_queries(FindVideosInput(opening="Bird Opening"))[0] == (
        "Bird Opening échecs ouverture explication"
    )
    # anglais demandé : une seule requête
    assert build_queries(FindVideosInput(opening="Italian Game", language="en")) == [
        "Italian Game chess opening tutorial"
    ]
    assert build_queries(FindVideosInput(query="gambit Evans idées")) == ["gambit Evans idées"]


def test_opening_or_query_is_required(registry):
    import asyncio

    result = asyncio.get_event_loop().run_until_complete(
        registry.execute("find_opening_videos", {"max_results": 2})
    )

    assert result.error["code"] == "invalid_arguments"


# --- Recherche, filtre, classement ------------------------------------------------------------


async def test_opening_search_merges_queries_and_ranks_french_quality_videos(registry, youtube):
    search, details = youtube

    result = await registry.execute("find_opening_videos", {**ITALIAN, "max_results": 5})

    assert result.ok
    data = result.data
    assert data["queries"] == [
        "partie italienne échecs ouverture explication",
        "Italian Game chess opening tutorial",
    ]
    assert search.call_count == 2
    assert details.call_count == 1
    assert (
        details.calls.last.request.url.params["part"] == "snippet,contentDetails,statistics,status"
    )
    ids = [v["video_id"] for v in data["videos"]]
    # les 5 françaises d'abord, classées par vues (les anglaises viennent ensuite)
    assert ids == ["P1Pb7HTKjSY", "gfxAZIX8hC4", "wpIcyDhdGKs", "_tltitUIyX8", "5Ny8I1Bj-ek"]
    first = data["videos"][0]
    assert first["title"] == "LE PLAN à connaitre dans l'ITALIENNE"
    assert first["embed_url"] == "https://www.youtube-nocookie.com/embed/P1Pb7HTKjSY"
    assert first["duration_s"] == 21 * 60 + 9 and first["views"] == 433024
    assert first["language"] == "fr"
    assert data["from_cache"] is False
    assert data["quota_used_today"] == 201


async def test_quality_filter_drops_too_long_and_foreign_language_videos(registry, respx_mock):
    respx_mock.get(SEARCH).respond(json=load_fixture("youtube/search_italian_en.json"))
    respx_mock.get(VIDEOS).respond(json=load_fixture("youtube/videos_italian.json"))
    await registry.execute("find_opening_videos", {**ITALIAN, "language": "en", "max_results": 5})

    # cache alimenté par la recherche : on relit la liste complète retenue
    cached = (await registry.execute("find_opening_videos", {**ITALIAN, "language": "en"})).data
    ids = {v["video_id"] for v in cached["videos"]}
    assert "QBXk_AUmL2s" not in ids  # 1 h 46 : trop long
    assert "2-iD3_M7104" not in ids  # audio en hindi
    assert cached["videos"][0]["video_id"] == "MhNs8GLo894"  # anglaise la plus vue


async def test_request_is_safe_for_young_players(registry, youtube):
    search, _ = youtube

    await registry.execute("find_opening_videos", {**FREE, "duration": "short", "max_results": 2})

    params = search.calls.last.request.url.params
    assert params["safeSearch"] == "strict"
    assert params["type"] == "video"
    assert params["videoEmbeddable"] == "true"
    assert params["videoDuration"] == "short"
    assert params["relevanceLanguage"] == "fr"
    assert params["key"] == "test-key"


async def test_duration_any_does_not_constrain_the_search(registry, youtube):
    search, _ = youtube

    await registry.execute("find_opening_videos", ITALIAN)

    assert "videoDuration" not in search.calls.last.request.url.params


async def test_max_results_is_enforced(registry, youtube):
    data = (await registry.execute("find_opening_videos", {**ITALIAN, "max_results": 1})).data

    assert len(data["videos"]) == 1


async def test_no_candidate_gives_an_empty_list_without_details_call(registry, respx_mock):
    respx_mock.get(SEARCH).respond(json={"items": []})
    details = respx_mock.get(VIDEOS)

    data = (await registry.execute("find_opening_videos", ITALIAN)).data

    assert data["videos"] == [] and not details.called
    assert data["quota_used_today"] == 200


# --- Cache et quota ---------------------------------------------------------------------------


async def test_second_search_is_served_from_cache(registry, youtube):
    search, _ = youtube

    first = await registry.execute("find_opening_videos", ITALIAN)
    second = await registry.execute("find_opening_videos", {**ITALIAN, "max_results": 2})

    assert first.data["from_cache"] is False and second.data["from_cache"] is True
    assert search.call_count == 2  # aucune nouvelle recherche
    assert [v["video_id"] for v in second.data["videos"]] == ["P1Pb7HTKjSY", "gfxAZIX8hC4"]
    assert second.data["quota_used_today"] == 201


async def test_cache_expires(make_registry, clock, youtube, respx_mock):
    registry = make_registry(youtube_cache_ttl_s=3600)
    await registry.execute("find_opening_videos", ITALIAN)
    respx_mock.get(SEARCH).respond(json={"items": []})

    clock.now += 3601
    data = (await registry.execute("find_opening_videos", ITALIAN)).data

    assert data["from_cache"] is False and data["videos"] == []


async def test_quota_reserve_blocks_new_searches_but_not_cached_ones(
    make_registry, youtube, respx_mock
):
    # 800 - 500 de réserve = 300 unités utilisables : une recherche (201) passe, pas deux
    registry = make_registry(youtube_quota_daily=800, youtube_quota_reserve=500)
    await registry.execute("find_opening_videos", ITALIAN)  # 201 unités

    respx_mock.get(SEARCH).respond(json={"items": []})
    blocked = await registry.execute("find_opening_videos", {"opening": "French Defense"})
    cached = await registry.execute("find_opening_videos", ITALIAN)

    assert blocked.error["code"] == "quota_exceeded" and blocked.error["kind"] == "unavailable"
    assert blocked.error["details"]["retry_after_s"] >= 1
    assert cached.ok and cached.data["from_cache"] is True


async def test_missing_api_key_fails_fast_without_calling(make_registry, respx_mock):
    route = respx_mock.get(SEARCH)

    result = await make_registry(youtube_api_key=None).execute("find_opening_videos", ITALIAN)

    assert result.error["code"] == "not_configured"
    assert not route.called


async def test_google_quota_error_is_not_retried(registry, respx_mock):
    route = respx_mock.get(SEARCH).respond(403, json=load_fixture("youtube/error_quota.json"))

    result = await registry.execute("find_opening_videos", FREE)

    assert result.error["code"] == "quota_exceeded"
    assert "aujourd'hui" in result.error["hint"]
    assert route.call_count == 1


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(403, json=load_fixture("youtube/error_key_invalid.json")),
        httpx.Response(403, text="forbidden"),
        httpx.Response(401),
    ],
    ids=["cle-invalide", "403-sans-json", "401"],
)
async def test_other_auth_errors(registry, respx_mock, response):
    respx_mock.get(SEARCH).mock(return_value=response)

    result = await registry.execute("find_opening_videos", FREE)

    assert result.error["code"] == "auth_required"


async def test_server_error_is_retried_once(registry, respx_mock):
    route = respx_mock.get(SEARCH).mock(
        side_effect=[httpx.Response(500), httpx.Response(200, json={"items": []})]
    )

    result = await registry.execute("find_opening_videos", FREE)

    assert result.ok and result.data["videos"] == []
    assert route.call_count == 2


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "ab"},
        {"query": "x" * 121},
        {**FREE, "max_results": 6},
        {**FREE, "duration": "long"},
        {**FREE, "language": "french"},
    ],
    ids=["requete-trop-courte", "requete-trop-longue", "trop-de-resultats", "duree-long", "langue"],
)
async def test_invalid_arguments(registry, respx_mock, arguments):
    route = respx_mock.get(SEARCH)

    result = await registry.execute("find_opening_videos", arguments)

    assert result.error["code"] == "invalid_arguments"
    assert not route.called


# --- Petites fonctions ------------------------------------------------------------------------


def test_parse_duration():
    assert parse_duration("PT21M9S") == 1269
    assert parse_duration("PT1H46M41S") == 6401
    assert parse_duration("PT45S") == 45
    assert parse_duration("P1DT2H") == 93600
    assert parse_duration(None) is None and parse_duration("n'importe quoi") is None


def test_seconds_until_reset_is_within_a_day():
    assert 1 <= seconds_until_reset(1_700_000_000.0) <= 86_400
