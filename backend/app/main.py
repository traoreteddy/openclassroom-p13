from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pymongo.errors import PyMongoError

from app.agent.graph import AgentConfig, ChessAgent
from app.agent.llm import ollama_factory
from app.agent.tracing import configure_tracing
from app.api.routes import agent as agent_routes
from app.api.routes import tools as tool_routes
from app.core.config import settings
from app.core.database import close_db, ping_db
from app.tools.catalog import build_registry, default_dependencies


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    deps = default_dependencies(settings)
    registry = build_registry(settings, deps)
    app.state.tool_registry = registry
    deps.start_warm_up()  # chargement du modèle d'embedding sans retarder le démarrage
    tracing = configure_tracing(
        environment=settings.app_env,
        langfuse_base_url=settings.langfuse_base_url,
        langfuse_public_key=settings.langfuse_public_key,
        langfuse_secret_key=settings.langfuse_secret_key,
        journal_path=Path(settings.decision_journal_path)
        if settings.decision_journal_path
        else None,
    )
    app.state.agent = ChessAgent(
        registry,
        tracing,
        ollama_factory(base_url=settings.ollama_base_url, model=settings.agent_model),
        AgentConfig(selection=settings.agent_selection),  # type: ignore[arg-type]
    )
    yield
    tracing.shutdown()
    await deps.aclose()
    await close_db()


app = FastAPI(
    title=settings.app_name,
    docs_url="/docs" if settings.app_env != "prod" else None,
    lifespan=lifespan,
)
# Le front Angular (autre origine en développement) appelle l'API depuis le navigateur.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
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
router.include_router(agent_routes.router)
app.include_router(router)
