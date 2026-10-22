"""Outil `find_opening_videos` : vidéos explicatives via l'API YouTube Data v3.

    opening ou query ──► requêtes « intelligentes » (nom FR + nom EN + mots-clés)
                              │  search.list (100 unités chacune), fusion, dédoublonnage
                              ▼
                         videos.list (1 unité) : durée, vues, likes, langue, intégrable
                              │
                              ▼
                   filtre qualité + classement ──► cache (7 jours) ──► vidéos

Le cache et le compteur de quota (`app/tools/video_store.py`) sont consultés avant tout
appel réseau : une recherche déjà faite ne coûte rien, et une journée presque épuisée
refuse la recherche avant que Google ne le fasse.
"""

from __future__ import annotations

import html
import re
import time
from typing import Any, Literal

import httpx
from pydantic import ConfigDict, Field, model_validator

from app.tools.base import SideEffect, ToolInput, ToolOutput, ToolSpec
from app.tools.errors import ToolError, ToolUnavailableError
from app.tools.http import get_json
from app.tools.resilience import ResiliencePolicy
from app.tools.video_store import (
    DETAILS_COST,
    SEARCH_COST,
    MemoryVideoStore,
    VideoStore,
    quota_day,
)

SERVICE = "YouTube"
QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"}
RESULTS_PER_QUERY = 5

DESCRIPTION = """\
Cherche sur YouTube des vidéos explicatives sur une ouverture ou une idée d'échecs.
Renvoie titre, chaîne, lien, lien de lecteur intégré, durée, vues et miniature, après un
filtre de qualité (vidéos intégrables de 3 à 45 minutes, suffisamment vues, dans la langue
demandée).

À utiliser : quand le joueur veut voir une explication, ou pour compléter une réponse
sur une ouverture identifiée. Donner `opening` (nom Lichess, ex. « Italian Game ») dès
qu'il est connu : les requêtes sont alors construites en français et en anglais.
Exemple : « Tu as une vidéo sur l'Italienne ? » → find_opening_videos(opening="Italian Game")
Sinon, `query` : une recherche libre (« gambit Evans explication »).

Ne pas utiliser : pour une position hors théorie sans nom d'ouverture (résultats hors
sujet), ni pour obtenir des coups (→ get_opening_theory).
Chaque recherche nouvelle consomme du quota (résultats gardés 7 jours) : ne pas répéter.
"""

# Noms français des ouvertures (clé : nom Lichess, famille ou variante) pour les requêtes.
OPENING_NAMES_FR: dict[str, str] = {
    "Italian Game": "partie italienne",
    "Italian Game: Two Knights Defense": "défense des deux cavaliers",
    "Italian Game: Evans Gambit": "gambit Evans",
    "Ruy Lopez": "partie espagnole",
    "Sicilian Defense": "défense sicilienne",
    "French Defense": "défense française",
    "Caro-Kann Defense": "défense Caro-Kann",
    "Scandinavian Defense": "défense scandinave",
    "Queen's Gambit": "gambit dame",
    "Queen's Gambit Declined": "gambit dame refusé",
    "Queen's Gambit Accepted": "gambit dame accepté",
    "Slav Defense": "défense slave",
    "Queen's Gambit Declined: Slav Defense": "défense slave",
    "King's Indian Defense": "défense est-indienne",
    "English Opening": "ouverture anglaise",
    "King's Pawn Game": "ouverture du pion roi",
    "Queen's Pawn Game": "ouverture du pion dame",
    "Scotch Game": "partie écossaise",
    "Vienna Game": "partie viennoise",
    "King's Gambit": "gambit du roi",
    "Pirc Defense": "défense Pirc",
    "Alekhine Defense": "défense Alekhine",
    "Nimzo-Indian Defense": "défense nimzo-indienne",
    "Grunfeld Defense": "défense Grünfeld",
    "Dutch Defense": "défense hollandaise",
    "London System": "système de Londres",
    "Petrov's Defense": "défense Petrov",
    "Philidor Defense": "défense Philidor",
}
CHESS_WORDS = ("échec", "echec", "chess", "ouverture", "opening", "gambit", "défense", "defense")


class FindVideosInput(ToolInput):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"opening": "Italian Game"},
                {"query": "gambit Evans explication", "duration": "medium"},
            ]
        }
    )

    opening: str | None = Field(
        default=None,
        min_length=3,
        max_length=100,
        description="Nom d'ouverture Lichess (ex. « Sicilian Defense ») : requêtes FR + EN.",
    )
    query: str | None = Field(default=None, min_length=3, max_length=120)
    max_results: int = Field(default=3, ge=1, le=5)
    duration: Literal["short", "medium", "any"] = Field(
        default="any",
        description="short : < 4 min ; medium : 4 à 20 min ; any : filtre qualité seul.",
    )
    language: str = Field(default="fr", pattern=r"^[a-z]{2}$", description="Code ISO 639-1.")

    @model_validator(mode="after")
    def _opening_or_query(self) -> FindVideosInput:
        if not self.opening and not self.query:
            raise ValueError("donner `opening` (nom d'ouverture) ou `query` (recherche libre)")
        return self


class Video(ToolOutput):
    video_id: str
    title: str
    channel: str
    url: str
    embed_url: str
    thumbnail_url: str | None
    published_at: str | None
    duration_s: int | None = None
    views: int | None = None
    likes: int | None = None
    language: str | None = None


class FindVideosOutput(ToolOutput):
    videos: list[Video]
    queries: list[str]
    from_cache: bool = False
    quota_used_today: int = 0


class YouTubeSearch:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        api_key: str | None,
        base_url: str = "https://www.googleapis.com/youtube/v3",
        store: VideoStore | None = None,
        quota_daily: int = 10_000,
        quota_reserve: int = 500,
        cache_ttl_s: int = 7 * 24 * 3600,
        clock=time.time,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._store: VideoStore = store or MemoryVideoStore()
        self._quota_daily = quota_daily
        self._quota_reserve = quota_reserve
        self._cache_ttl_s = cache_ttl_s
        self._clock = clock

    async def search(self, params: FindVideosInput) -> FindVideosOutput:
        queries = build_queries(params)
        key = cache_key(params, queries)
        day = quota_day(self._clock())

        cached = await self._store.get(key)
        if cached is not None:
            used = await self._store.used_units(day)
            return FindVideosOutput(
                videos=[Video(**v) for v in cached][: params.max_results],
                queries=queries,
                from_cache=True,
                quota_used_today=used,
            )

        if not self._api_key:
            raise ToolUnavailableError(
                code="not_configured",
                message="La recherche YouTube n'est pas configurée.",
                hint="Renseigner YOUTUBE_API_KEY ; en attendant, répondre sans vidéo.",
            )

        cost = SEARCH_COST * len(queries) + DETAILS_COST
        used = await self._store.used_units(day)
        if used + cost > self._quota_daily - self._quota_reserve:
            raise ToolUnavailableError(
                code="quota_exceeded",
                message=f"Quota YouTube du jour presque épuisé ({used}/{self._quota_daily}).",
                hint="Ne plus lancer de nouvelle recherche aujourd'hui ; répondre sans vidéo.",
                retry_after_s=seconds_until_reset(self._clock()),
            )

        candidates: dict[str, dict[str, Any]] = {}
        for query in queries:
            for item in await self._search(query, params):
                video_id = _video_id(item)
                if video_id and video_id not in candidates:
                    candidates[video_id] = item
            await self._store.add_units(day, SEARCH_COST)

        details = await self._details(list(candidates)) if candidates else {}
        if candidates:
            await self._store.add_units(day, DETAILS_COST)

        videos = [_video(item, details.get(_video_id(item) or "")) for item in candidates.values()]
        selected = select_quality(videos, language=params.language, terms=search_terms(queries))
        await self._store.put(key, [v.model_dump() for v in selected], self._cache_ttl_s)
        return FindVideosOutput(
            videos=selected[: params.max_results],
            queries=queries,
            from_cache=False,
            quota_used_today=await self._store.used_units(day),
        )

    async def _search(self, query: str, params: FindVideosInput) -> list[dict[str, Any]]:
        request: dict[str, Any] = {
            "part": "snippet",
            "type": "video",
            "q": query,
            "maxResults": RESULTS_PER_QUERY,
            "relevanceLanguage": params.language,
            "safeSearch": "strict",
            "videoEmbeddable": "true",
            "key": self._api_key,
        }
        if params.duration != "any":
            request["videoDuration"] = params.duration
        payload = await get_json(
            self._client,
            f"{self._base_url}/search",
            service=SERVICE,
            params=request,
            override=_quota_error,
        )
        return list(payload.get("items", []))

    async def _details(self, video_ids: list[str]) -> dict[str, dict[str, Any]]:
        payload = await get_json(
            self._client,
            f"{self._base_url}/videos",
            service=SERVICE,
            params={
                "part": "snippet,contentDetails,statistics,status",
                "id": ",".join(video_ids),
                "key": self._api_key,
            },
            override=_quota_error,
        )
        return {item["id"]: item for item in payload.get("items", []) if item.get("id")}


# --- Requêtes, filtre et classement ---------------------------------------------------------


def build_queries(params: FindVideosInput) -> list[str]:
    """Requêtes combinant le nom de l'ouverture (FR et EN) et des mots-clés pédagogiques."""
    if not params.opening:
        return [str(params.query).strip()]
    opening = params.opening.strip()
    french = OPENING_NAMES_FR.get(opening) or OPENING_NAMES_FR.get(opening.split(":")[0].strip())
    queries = []
    if params.language == "fr":
        queries.append(f"{french or opening} échecs ouverture explication")
    queries.append(f"{opening} chess opening tutorial")
    return queries


def cache_key(params: FindVideosInput, queries: list[str]) -> str:
    return "|".join([*queries, params.language, params.duration]).lower()


def search_terms(queries: list[str]) -> set[str]:
    """Mots significatifs des requêtes (4 lettres et plus), pour juger un titre pertinent."""
    return {w for q in queries for w in re.findall(r"[^\W\d_]{4,}", q.lower())} | set(CHESS_WORDS)


def select_quality(videos: list[Video], *, language: str, terms: set[str]) -> list[Video]:
    """Garde les vidéos de 3 à 45 min, assez vues, dont le titre reprend un terme de la
    recherche, dans la langue demandée quand elle est connue ; classe par langue puis vues."""
    kept = [v for v in videos if _is_quality(v, language, terms)]
    kept.sort(key=lambda v: (v.language != language, -(v.views or 0)))
    return kept


MIN_DURATION_S = 3 * 60
MAX_DURATION_S = 45 * 60
MIN_VIEWS = 1_000


def _is_quality(video: Video, language: str, terms: set[str]) -> bool:
    if video.duration_s is not None and not MIN_DURATION_S <= video.duration_s <= MAX_DURATION_S:
        return False
    if video.views is not None and video.views < MIN_VIEWS:
        return False
    title = video.title.lower()
    if not any(term in title for term in terms):
        return False
    return video.language is None or video.language == language


def seconds_until_reset(now: float) -> int:
    """Secondes jusqu'à minuit, heure du Pacifique (remise à zéro du quota)."""
    from datetime import UTC, datetime, timedelta

    from app.tools.video_store import QUOTA_TIMEZONE

    local = datetime.fromtimestamp(now, tz=UTC).astimezone(QUOTA_TIMEZONE)
    midnight = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((midnight - local).total_seconds()))


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
    identifier = item.get("id")
    if isinstance(identifier, dict):
        return identifier.get("videoId")
    return identifier if isinstance(identifier, str) else None


def _video(item: dict[str, Any], detail: dict[str, Any] | None) -> Video:
    snippet = item.get("snippet", {})
    video_id = _video_id(item) or ""
    thumbnails = snippet.get("thumbnails", {})
    thumbnail = thumbnails.get("medium") or thumbnails.get("default") or {}
    detail = detail or {}
    stats = detail.get("statistics", {})
    detail_snippet = detail.get("snippet", {})
    language = detail_snippet.get("defaultAudioLanguage") or detail_snippet.get("defaultLanguage")
    return Video(
        video_id=video_id,
        title=html.unescape(snippet.get("title", "")),
        channel=html.unescape(snippet.get("channelTitle", "")),
        url=f"https://www.youtube.com/watch?v={video_id}",
        embed_url=f"https://www.youtube-nocookie.com/embed/{video_id}",
        thumbnail_url=thumbnail.get("url"),
        published_at=snippet.get("publishedAt"),
        duration_s=parse_duration(detail.get("contentDetails", {}).get("duration")),
        views=_int(stats.get("viewCount")),
        likes=_int(stats.get("likeCount")),
        language=language.split("-")[0].lower() if language else None,
    )


def parse_duration(value: str | None) -> int | None:
    """Durée ISO 8601 de YouTube (« PT21M9S », « PT1H46M41S ») en secondes."""
    if not value:
        return None
    match = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value)
    if not match:
        return None
    days, hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
            timeout_s=8.0, max_attempts=2, breaker_threshold=3, breaker_reset_s=60.0
        ),
    )
