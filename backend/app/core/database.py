from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import settings

# Le client ne se connecte qu'à la première requête envoyée à MongoDB
client: AsyncMongoClient = AsyncMongoClient(
    settings.mongodb_url,
    serverSelectionTimeoutMS=3000,
)


def get_db() -> AsyncDatabase:
    return client[settings.mongodb_db_name]


async def ping_db() -> None:
    await client.admin.command("ping")


async def close_db() -> None:
    await client.close()
