"""Outil find_opening_videos : API YouTube simulée avec respx + fixtures versionnées."""

from __future__ import annotations

import httpx
import pytest

from tests.conftest import YOUTUBE_URL, load_fixture

SEARCH = f"{YOUTUBE_URL}/search"
QUERY = {"query": "partie italienne explication"}


@pytest.fixture
def registry(make_registry):
    return make_registry()


async def test_parses_videos_and_skips_channels(registry, respx_mock):
    respx_mock.get(SEARCH).respond(json=load_fixture("youtube/search_italian.json"))

    data = (await registry.execute("find_opening_videos", QUERY)).data

    assert [v["video_id"] for v in data["videos"]] == ["dQw4w9WgXc1", "aBcDeFgHiJ2"]
    first, second = data["videos"]
    assert first["url"] == "https://www.youtube.com/watch?v=dQw4w9WgXc1"
    assert first["channel"] == "Échecs & Stratégie"  # entité HTML décodée
    assert first["thumbnail_url"].endswith("mqdefault.jpg")
    assert second["thumbnail_url"].endswith("default.jpg")  # pas de miniature medium
    assert first["published_at"] == "2024-03-12T17:00:01Z"


async def test_request_is_safe_for_young_players(registry, respx_mock):
    route = respx_mock.get(SEARCH).respond(json={"items": []})

    await registry.execute("find_opening_videos", {**QUERY, "duration": "short", "max_results": 2})

    params = route.calls.last.request.url.params
    assert params["safeSearch"] == "strict"
    assert params["type"] == "video"
    assert params["videoDuration"] == "short"
    assert params["relevanceLanguage"] == "fr"
    assert params["maxResults"] == "2"
    assert params["key"] == "test-key"


async def test_max_results_is_enforced_even_if_api_returns_more(registry, respx_mock):
    respx_mock.get(SEARCH).respond(json=load_fixture("youtube/search_italian.json"))

    data = (await registry.execute("find_opening_videos", {**QUERY, "max_results": 1})).data

    assert len(data["videos"]) == 1


async def test_missing_api_key_fails_fast_without_calling(make_registry, respx_mock):
    route = respx_mock.get(SEARCH)

    result = await make_registry(youtube_api_key=None).execute("find_opening_videos", QUERY)

    assert result.error["code"] == "not_configured"
    assert not route.called


async def test_quota_exceeded_is_not_retried(registry, respx_mock):
    route = respx_mock.get(SEARCH).respond(403, json=load_fixture("youtube/error_quota.json"))

    result = await registry.execute("find_opening_videos", QUERY)

    assert result.error["code"] == "quota_exceeded"
    assert result.error["kind"] == "unavailable"
    assert "aujourd'hui" in result.error["hint"]
    assert route.call_count == 1


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(403, json=load_fixture("youtube/error_key_invalid.json")),
        httpx.Response(403, text="forbidden"),  # corps illisible
        httpx.Response(401),
    ],
    ids=["cle-invalide", "403-sans-json", "401"],
)
async def test_other_auth_errors(registry, respx_mock, response):
    respx_mock.get(SEARCH).mock(return_value=response)

    result = await registry.execute("find_opening_videos", QUERY)

    assert result.error["code"] == "auth_required"


async def test_server_error_is_retried_once(registry, respx_mock):
    route = respx_mock.get(SEARCH).mock(
        side_effect=[httpx.Response(500), httpx.Response(200, json={"items": []})]
    )

    result = await registry.execute("find_opening_videos", QUERY)

    assert result.ok
    assert result.data == {"videos": []}
    assert route.call_count == 2


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "ab"},
        {"query": "x" * 121},
        {**QUERY, "max_results": 6},
        {**QUERY, "duration": "long"},
        {**QUERY, "language": "french"},
    ],
    ids=["requete-trop-courte", "requete-trop-longue", "trop-de-resultats", "duree-long", "langue"],
)
async def test_invalid_arguments(registry, respx_mock, arguments):
    route = respx_mock.get(SEARCH)

    result = await registry.execute("find_opening_videos", arguments)

    assert result.error["code"] == "invalid_arguments"
    assert not route.called
