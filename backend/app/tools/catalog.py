"""Assemblage du catalogue d'outils exposé à l'agent.

Les dépendances (client HTTP, moteur, base vectorielle) sont injectées : l'application
fournit les vraies, les tests et les scripts de mesure fournissent des doubles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from app.core.config import Settings
from app.core.database import get_db
from app.rag.embedding import SentenceTransformerEmbedder
from app.rag.milvus_store import MetadataFilter, MilvusVectorStore
from app.tools.base import ToolRegistry
from app.tools.deep_analysis import DeepAnalysisJobs, build_deep_analysis_tools
from app.tools.knowledge import KnowledgeSearch, build_search_knowledge_tool
from app.tools.lichess import LichessExplorer, build_opening_theory_tool
from app.tools.position import build_inspect_position_tool
from app.tools.resilience import Clock, Sleep
from app.tools.stockfish import Analyzer, StockfishAnalyzer, build_evaluate_position_tool
from app.tools.video_store import MemoryVideoStore, MongoVideoStore, VideoStore
from app.tools.videos import YouTubeSearch, build_find_videos_tool


@dataclass
class ToolDependencies:
    http: httpx.AsyncClient
    analyzer: Analyzer
    knowledge: KnowledgeSearch
    # Cache des vidéos et compteur de quota YouTube (mémoire par défaut, MongoDB en prod)
    video_store: VideoStore = field(default_factory=MemoryVideoStore)

    def start_warm_up(self) -> None:
        """Préchauffe en arrière-plan ce qui est lent à charger (modèle d'embedding)."""
        embedder = getattr(self.knowledge, "_embedder", None)
        warm_up = getattr(embedder, "start_warm_up", None)
        if warm_up is not None:
            warm_up()

    async def aclose(self) -> None:
        """Libère les ressources : connexions HTTP et processus Stockfish."""
        await self.http.aclose()
        close = getattr(self.analyzer, "close", None)
        if close is not None:
            await close()


def default_dependencies(settings: Settings) -> ToolDependencies:
    """Dépendances réelles. Rien n'est contacté ni chargé avant le premier appel d'outil."""
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model, device=settings.embedding_device, dtype=settings.embedding_dtype
    )
    store = MilvusVectorStore(
        settings.milvus_uri,
        settings.milvus_collection,
        default_filter=MetadataFilter(max_confidentiality=settings.rag_max_confidentiality),
    )
    return ToolDependencies(
        http=httpx.AsyncClient(timeout=settings.http_timeout_s),
        analyzer=StockfishAnalyzer(
            settings.stockfish_path,
            threads=settings.stockfish_threads,
            hash_mb=settings.stockfish_hash_mb,
        ),
        knowledge=KnowledgeSearch(embedder, store, min_score=settings.rag_min_score),
        video_store=MongoVideoStore(get_db()),
    )


def build_registry(
    settings: Settings,
    deps: ToolDependencies,
    *,
    sleep: Sleep | None = None,
    clock: Clock | None = None,
) -> ToolRegistry:
    explorer = LichessExplorer(
        deps.http,
        base_url=settings.lichess_explorer_url,
        token=settings.lichess_token,
        min_games=settings.theory_min_games,
    )
    youtube = YouTubeSearch(
        deps.http,
        api_key=settings.youtube_api_key,
        base_url=settings.youtube_api_url,
        store=deps.video_store,
        quota_daily=settings.youtube_quota_daily,
        quota_reserve=settings.youtube_quota_reserve,
        cache_ttl_s=settings.youtube_cache_ttl_s,
        **({"clock": clock} if clock else {}),
    )
    jobs = DeepAnalysisJobs(deps.analyzer, **({"clock": clock} if clock else {}))

    specs = [
        build_inspect_position_tool(),
        build_opening_theory_tool(explorer),
        build_evaluate_position_tool(deps.analyzer),
        build_search_knowledge_tool(deps.knowledge),
        build_find_videos_tool(youtube),
        *build_deep_analysis_tools(jobs),
    ]
    options = {k: v for k, v in {"sleep": sleep, "clock": clock}.items() if v is not None}
    return ToolRegistry(specs, **options)
