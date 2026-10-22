FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.20 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Stockfish (moteur UCI) : paquet Debian, exécutable /usr/games/stockfish
RUN apt-get update \
    && apt-get install -y --no-install-recommends stockfish \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
# Scripts (ingestion, mesures) et corpus figé : l'ingestion s'exécute dans le conteneur
COPY scripts ./scripts
COPY data ./data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
