"""Outil `find_opening_videos` : vidéos explicatives via l'API YouTube Data v3."""

from __future__ import annotations

import html
from typing import Any, Literal

import httpx
from pydantic import ConfigDict, Field

from app.tools.base import SideEffect, ToolInput, ToolOutput, ToolSpec
from app.tools.errors import ToolError, ToolUnavailableError
from app.tools.http import get_json
from app.tools.resilience import ResiliencePolicy

SERVICE = "YouTube"
QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"}

DESCRIPTION = """\
Cherche sur YouTube des vidéos explicatives sur une ouverture ou une idée d'échecs.
Renvoie titre, chaîne, lien et miniature. Préférer des requêtes précises (nom de
l'ouverture + variante + « explication ») et des vidéos courtes ou moyennes.

À utiliser : quand le joueur veut voir une explication, ou pour compléter une réponse
sur une ouverture identifiée.
Exemple : « Tu as une vidéo sur l'Italienne ? » →
find_opening_videos(query="partie italienne explication ouverture", duration="medium")

Ne pas utiliser : pour une position hors théorie sans nom d'ouverture (résultats hors
sujet), ni pour obtenir des coups (→ get_opening_theory).
Chaque appel consomme du quota : ne pas répéter la même recherche.
"""


class FindVideosInput(ToolInput):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"query": "partie italienne explication ouverture", "duration": "medium"}]
        }
    )

    query: str = Field(min_length=3, max_length=120)
    max_results: int = Field(default=3, ge=1, le=5)
    duration: Literal["short", "medium", "any"] = Field(
        default="medium", description="short : < 4 min ; medium : 4 à 20 min."
    )
    language: str = Field(default="fr", pattern=r"^[a-z]{2}$", description="Code ISO 639-1.")


class Video(ToolOutput):
    video_id: str
    title: str
    channel: str
    url: str
    thumbnail_url: str | None
    published_at: str | None


class FindVideosOutput(ToolOutput):
    videos: list[Video]


class YouTubeSearch:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        api_key: str | None,
        base_url: str = "https://www.googleapis.com/youtube/v3",
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def search(self, params: FindVideosInput) -> FindVideosOutput:
        if not self._api_key:
            raise ToolUnavailableError(
                code="not_configured",
                message="La recherche YouTube n'est pas configurée.",
                hint="Renseigner YOUTUBE_API_KEY ; en attendant, répondre sans vidéo.",
            )
        payload = await get_json(
            self._client,
            f"{self._base_url}/search",
            service=SERVICE,
            params={
                "part": "snippet",
                "type": "video",
                "q": params.query,
                "maxResults": params.max_results,
                "videoDuration": params.duration,
                "relevanceLanguage": params.language,
                "safeSearch": "strict",
                "videoEmbeddable": "true",
                "key": self._api_key,
            },
            override=_quota_error,
        )
        videos = [_video(item) for item in payload.get("items", []) if _video_id(item)]
        return FindVideosOutput(videos=videos[: params.max_results])


def _quota_error(response: httpx.Response) -> ToolError | None:
    """YouTube signale le quota épuisé par un 403 : à distinguer d'une clé invalide."""
    if response.status_code != 403:
        return None
    try:
        reasons = {e.get("reason") for e in response.json()["error"]["errors"]}
    except (ValueError, KeyError, TypeError):
        return None
    if reasons & QUOTA_REASONS:
        return ToolUnavailableError(
            code="quota_exceeded",
            message="Le quota quotidien de l'API YouTube est épuisé.",
            hint="Ne plus appeler find_opening_videos aujourd'hui ; répondre sans vidéo.",
        )
    return None


def _video_id(item: dict[str, Any]) -> str | None:
    return item.get("id", {}).get("videoId")


def _video(item: dict[str, Any]) -> Video:
    snippet = item.get("snippet", {})
    video_id = _video_id(item) or ""
    thumbnails = snippet.get("thumbnails", {})
    thumbnail = thumbnails.get("medium") or thumbnails.get("default") or {}
    return Video(
        video_id=video_id,
        title=html.unescape(snippet.get("title", "")),
        channel=html.unescape(snippet.get("channelTitle", "")),
        url=f"https://www.youtube.com/watch?v={video_id}",
        thumbnail_url=thumbnail.get("url"),
        published_at=snippet.get("publishedAt"),
    )


def build_find_videos_tool(youtube: YouTubeSearch) -> ToolSpec:
    return ToolSpec(
        name="find_opening_videos",
        description=DESCRIPTION,
        input_model=FindVideosInput,
        output_model=FindVideosOutput,
        handler=youtube.search,
        side_effect=SideEffect.READ_EXTERNAL,
        idempotent=True,
        policy=ResiliencePolicy(
            timeout_s=5.0, max_attempts=2, breaker_threshold=3, breaker_reset_s=60.0
        ),
    )
