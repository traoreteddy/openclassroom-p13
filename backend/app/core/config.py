from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ../.env : fichier de la racine du projet quand l'API est lancée depuis backend/
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    app_name: str = "P13 Chess Openings API"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"
    # Origines autorisées (CORS) pour le front Angular, séparées par des virgules
    cors_origins: str = "http://localhost:4200"

    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "p13"

    # Outils de l'agent
    lichess_explorer_url: str = "https://explorer.lichess.ovh"
    lichess_token: str | None = None
    theory_min_games: int = 20

    stockfish_path: str = "stockfish"
    stockfish_threads: int = 1
    stockfish_hash_mb: int = 64

    youtube_api_url: str = "https://www.googleapis.com/youtube/v3"
    youtube_api_key: str | None = None

    http_timeout_s: float = 5.0

    # RAG
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "openings"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    # Périphérique de l'encodeur (None = choix automatique : GPU si disponible). « cpu »
    # évite la concurrence avec Ollama sur le GPU du Mac pendant les évaluations.
    embedding_device: str | None = None
    # Similarité cosinus minimale pour qu'un passage soit jugé pertinent (calibrée en phase 4)
    rag_min_score: float = 0.6
    # Niveau maximal visible par l'agent (public < interne < confidentiel) : réglé par le
    # serveur, jamais par le LLM
    rag_max_confidentiality: str = "public"

    # LLM local (Ollama sur la machine hôte, hors Docker)
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "phi4-mini"
    # Modèle et mode de sélection de l'agent (choisis par la mesure en phase 5)
    agent_model: str = "qwen3:14b"
    agent_selection: str = "native"  # native (appel d'outils) ou structured (schéma JSON)

    # Traces : Langfuse auto-hébergé (profil docker compose « tracing »). Sans clés, les
    # spans OpenTelemetry restent locaux (journal de décision seulement).
    langfuse_base_url: str = "http://localhost:3000"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    decision_journal_path: str | None = "data/journal/decisions.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
