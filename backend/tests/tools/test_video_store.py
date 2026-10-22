"""Magasins du cache vidéos et du compteur de quota : mémoire et MongoDB (doublé)."""

from __future__ import annotations

from datetime import UTC, datetime

from app.tools.video_store import MemoryVideoStore, MongoVideoStore, quota_day

VIDEOS = [{"video_id": "a", "title": "A"}]


def test_quota_day_follows_pacific_time():
    # 2024-03-10 06:00 UTC = 2024-03-09 22:00 à Los Angeles (encore la veille)
    assert quota_day(datetime(2024, 3, 10, 6, 0, tzinfo=UTC).timestamp()) == "2024-03-09"
    assert quota_day(datetime(2024, 3, 10, 9, 0, tzinfo=UTC).timestamp()) == "2024-03-10"


async def test_memory_store_expires_entries_and_counts_units():
    now = [1000.0]
    store = MemoryVideoStore(clock=lambda: now[0])

    await store.put("k", VIDEOS, ttl_s=60)
    assert await store.get("k") == VIDEOS
    now[0] += 61
    assert await store.get("k") is None

    await store.add_units("2024-03-10", 100)
    await store.add_units("2024-03-10", 1)
    assert await store.used_units("2024-03-10") == 101
    assert await store.used_units("2024-03-11") == 0


class FakeCollection:
    def __init__(self, fail: bool = False) -> None:
        self.docs: dict[str, dict] = {}
        self.indexes: list[tuple] = []
        self.fail = fail

    async def create_index(self, *args, **kwargs):
        self.indexes.append((args, kwargs))

    async def find_one(self, query):
        if self.fail:
            raise ConnectionError("mongo down")
        doc = self.docs.get(query["_id"])
        if doc and "expires_at" in query and not doc["expires_at"] > query["expires_at"]["$gt"]:
            return None
        return doc

    async def replace_one(self, query, document, upsert=False):
        if self.fail:
            raise ConnectionError("mongo down")
        self.docs[query["_id"]] = document

    async def update_one(self, query, update, upsert=False):
        if self.fail:
            raise ConnectionError("mongo down")
        doc = self.docs.setdefault(query["_id"], {"_id": query["_id"], "units": 0})
        doc["units"] += update["$inc"]["units"]


class FakeDatabase(dict):
    def __init__(self, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail

    def __getitem__(self, name):
        return self.setdefault(name, FakeCollection(self.fail))


async def test_mongo_store_caches_with_ttl_index_and_counts_units():
    db = FakeDatabase()
    store = MongoVideoStore(db)

    await store.put("k", VIDEOS, ttl_s=3600)
    await store.add_units("2024-03-10", 100)

    assert await store.get("k") == VIDEOS
    assert db["video_cache"].indexes[0] == (("expires_at",), {"expireAfterSeconds": 0})
    assert await store.used_units("2024-03-10") == 100
    assert await store.get("absent") is None


async def test_mongo_store_failures_degrade_to_no_cache():
    store = MongoVideoStore(FakeDatabase(fail=True))

    await store.put("k", VIDEOS, ttl_s=10)  # ne lève pas
    await store.add_units("2024-03-10", 100)  # ne lève pas

    assert await store.get("k") is None
    assert await store.used_units("2024-03-10") == 0
