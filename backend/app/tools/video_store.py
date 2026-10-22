"""Cache des recherches de vidéos et compteur de quota YouTube.

L'API YouTube Data v3 accorde 10 000 unités par jour (remise à zéro à minuit, heure du
Pacifique) ; une recherche coûte 100 unités, la lecture des détails de vidéos 1 unité.
Deux protections, derrière un même magasin (`VideoStore`) :

- **cache** des résultats par clé (ouverture ou requête, langue, nombre), avec expiration ;
- **registre de quota** : unités consommées par jour, pour refuser une recherche avant
  que Google ne le fasse, en gardant une réserve.

`MongoVideoStore` persiste dans MongoDB (collections `video_cache`, index TTL, et
`youtube_quota`). `MemoryVideoStore` sert aux tests et de repli si MongoDB est absent :
une panne du magasin ne doit jamais empêcher une recherche.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any, Protocol
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

QUOTA_TIMEZONE = ZoneInfo("America/Los_Angeles")
SEARCH_COST = 100  # search.list
DETAILS_COST = 1  # videos.list


def quota_day(now: float | None = None) -> str:
    """Jour de quota YouTube (heure du Pacifique), clé du compteur."""
    stamp = datetime.fromtimestamp(now if now is not None else time.time(), tz=UTC)
    return stamp.astimezone(QUOTA_TIMEZONE).strftime("%Y-%m-%d")


class VideoStore(Protocol):
    async def get(self, key: str) -> list[dict[str, Any]] | None: ...

    async def put(self, key: str, videos: list[dict[str, Any]], ttl_s: int) -> None: ...

    async def used_units(self, day: str) -> int: ...

    async def add_units(self, day: str, units: int) -> None: ...


class MemoryVideoStore:
    def __init__(self, clock=time.time) -> None:
        self._clock = clock
        self._cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._units: dict[str, int] = {}

    async def get(self, key: str) -> list[dict[str, Any]] | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        expires_at, videos = entry
        if expires_at <= self._clock():
            del self._cache[key]
            return None
        return [dict(v) for v in videos]

    async def put(self, key: str, videos: list[dict[str, Any]], ttl_s: int) -> None:
        self._cache[key] = (self._clock() + ttl_s, [dict(v) for v in videos])

    async def used_units(self, day: str) -> int:
        return self._units.get(day, 0)

    async def add_units(self, day: str, units: int) -> None:
        self._units[day] = self._units.get(day, 0) + units


class MongoVideoStore:
    """Magasin MongoDB. Toute erreur Mongo est journalisée et traitée comme un cache vide."""

    def __init__(self, database: Any) -> None:
        self._cache = database["video_cache"]
        self._quota = database["youtube_quota"]
        self._indexed = False

    async def _ensure_index(self) -> None:
        if self._indexed:
            return
        await self._cache.create_index("expires_at", expireAfterSeconds=0)
        self._indexed = True

    async def get(self, key: str) -> list[dict[str, Any]] | None:
        try:
            document = await self._cache.find_one(
                {"_id": key, "expires_at": {"$gt": datetime.now(UTC)}}
            )
        except Exception as exc:  # pymongo.errors.PyMongoError et connexions refusées
            log.warning("cache vidéos indisponible (%s) : recherche sans cache", exc)
            return None
        return list(document["videos"]) if document else None

    async def put(self, key: str, videos: list[dict[str, Any]], ttl_s: int) -> None:
        try:
            await self._ensure_index()
            now = datetime.now(UTC)
            await self._cache.replace_one(
                {"_id": key},
                {
                    "_id": key,
                    "videos": videos,
                    "fetched_at": now,
                    "expires_at": datetime.fromtimestamp(now.timestamp() + ttl_s, tz=UTC),
                },
                upsert=True,
            )
        except Exception as exc:
            log.warning("cache vidéos non écrit (%s)", exc)

    async def used_units(self, day: str) -> int:
        try:
            document = await self._quota.find_one({"_id": day})
        except Exception as exc:
            log.warning("compteur de quota illisible (%s) : supposé à zéro", exc)
            return 0
        return int(document["units"]) if document else 0

    async def add_units(self, day: str, units: int) -> None:
        try:
            await self._quota.update_one({"_id": day}, {"$inc": {"units": units}}, upsert=True)
        except Exception as exc:
            log.warning("compteur de quota non mis à jour (%s)", exc)
