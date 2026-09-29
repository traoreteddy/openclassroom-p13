from fastapi import APIRouter, FastAPI

from app.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name)
router = APIRouter(prefix=settings.api_v1_prefix)


@router.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}


app.include_router(router)
