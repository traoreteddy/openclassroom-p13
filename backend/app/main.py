from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, HTTPException, status
from pymongo.errors import PyMongoError

from app.api.routes import tools as tool_routes
from app.core.config import settings
from app.core.database import close_db, ping_db
from app.tools.catalog import build_registry, default_dependencies


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    deps = default_dependencies(settings)
    app.state.tool_registry = build_registry(settings, deps)
    deps.start_warm_up()  # chargement du modèle d'embedding sans retarder le démarrage
    yield
    await deps.aclose()
    await close_db()


app = FastAPI(
    title=settings.app_name,
    docs_url="/docs" if settings.app_env != "prod" else None,
    lifespan=lifespan,
)
router = APIRouter(prefix=settings.api_v1_prefix)


@router.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}


@router.get("/healthcheck/database")
async def healthcheck_database() -> dict[str, str]:
    try:
        await ping_db()
    except PyMongoError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MongoDB unavailable",
        ) from exc
    return {"status": "ok", "database": settings.mongodb_db_name}


router.include_router(tool_routes.router)
app.include_router(router)
