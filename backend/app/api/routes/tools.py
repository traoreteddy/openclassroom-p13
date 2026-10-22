"""Routes HTTP adossées aux outils de l'agent.

Les routes ne contiennent aucune logique métier : elles appellent l'outil et traduisent
son `ToolResult` en réponse HTTP.

| Résultat de l'outil      | Réponse HTTP                        |
|--------------------------|-------------------------------------|
| ok                       | 200 + données                       |
| erreur `input`           | 422 + erreur                        |
| erreur `unavailable`     | 503 + erreur (+ en-tête Retry-After)|
| analyse lancée           | 202 + ticket (job_id)               |
| job inconnu              | 404 + erreur                        |
"""

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Path, Query
from fastapi.responses import JSONResponse

from app.api.deps import get_registry
from app.tools.base import ToolRegistry, ToolResult

router = APIRouter(tags=["outils"])

Registry = Annotated[ToolRegistry, Depends(get_registry)]
FenPath = Annotated[
    str,
    Path(
        description="FEN encodée dans l'URL (espaces → %20).",
        examples=["rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"],
    ),
]


@router.get("/moves/{fen:path}", summary="Coups théoriques (Lichess)")
async def get_moves(
    fen: FenPath,
    registry: Registry,
    database: Literal["masters", "lichess"] = "masters",
    max_moves: Annotated[int, Query(ge=1, le=12)] = 5,
    include_games: bool = False,
) -> Any:
    result = await registry.execute(
        "get_opening_theory",
        {
            "fen": fen,
            "database": database,
            "max_moves": max_moves,
            "include_games": include_games,
        },
    )
    return to_response(result)


@router.get("/evaluate/{fen:path}", summary="Évaluation Stockfish")
async def evaluate(
    fen: FenPath,
    registry: Registry,
    depth: Annotated[int, Query(ge=8, le=22)] = 16,
    lines: Annotated[int, Query(ge=1, le=3)] = 1,
) -> Any:
    result = await registry.execute(
        "evaluate_position", {"fen": fen, "depth": depth, "lines": lines}
    )
    return to_response(result)


@router.get("/vector-search", summary="Recherche vectorielle dans la base de connaissances")
async def vector_search(
    registry: Registry,
    query: Annotated[str, Query(min_length=3, max_length=300, description="Question.")],
    opening: Annotated[
        str | None, Query(max_length=100, description="Nom d'ouverture Lichess (filtre).")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=8)] = 4,
) -> Any:
    arguments: dict[str, Any] = {"query": query, "limit": limit}
    if opening:
        arguments["opening"] = opening
    return to_response(await registry.execute("search_opening_knowledge", arguments))


@router.get("/videos/{opening}", summary="Vidéos explicatives YouTube sur une ouverture")
async def videos(
    opening: Annotated[
        str,
        Path(min_length=3, max_length=100, description="Nom d'ouverture (Lichess, en anglais)."),
    ],
    registry: Registry,
    max_results: Annotated[int, Query(ge=1, le=5)] = 3,
    language: Annotated[str, Query(pattern=r"^[a-z]{2}$")] = "fr",
) -> Any:
    """200 avec la liste (vide + `message` si rien ne passe le filtre) ; 503 si YouTube est
    non configuré ou le quota épuisé (`Retry-After` : secondes avant la remise à zéro)."""
    result = await registry.execute(
        "find_opening_videos",
        {"opening": opening, "max_results": max_results, "language": language},
    )
    if result.ok and not (result.data or {}).get("videos"):
        return {
            **(result.data or {}),
            "message": f"Aucune vidéo de qualité trouvée pour « {opening} » en « {language} ».",
        }
    return to_response(result)


@router.post(
    "/analysis/{fen:path}",
    status_code=202,
    summary="Lance une analyse Stockfish profonde en tâche de fond",
)
async def start_analysis(
    fen: FenPath,
    registry: Registry,
    depth: Annotated[int, Query(ge=23, le=40)] = 28,
) -> Any:
    """Renvoie un `job_id` à interroger avec `GET /analysis/{job_id}` (202 Accepted)."""
    result = await registry.execute("start_deep_analysis", {"fen": fen, "depth": depth})
    return to_response(result)


@router.get("/analysis/{job_id}", summary="État et résultat d'une analyse profonde")
async def get_analysis(
    job_id: Annotated[str, Path(min_length=1, max_length=64)], registry: Registry
) -> Any:
    result = await registry.execute("get_deep_analysis", {"job_id": job_id})
    if not result.ok and result.error and result.error["code"] == "unknown_job":
        return JSONResponse(status_code=404, content={"error": result.error})
    return to_response(result)


@router.get("/tools", summary="Catalogue des outils exposés à l'agent")
async def list_tools(registry: Registry) -> list[dict[str, Any]]:
    return registry.definitions()


def to_response(result: ToolResult) -> Any:
    if result.ok:
        return result.data
    assert result.error is not None
    status = 422 if result.error["kind"] == "input" else 503
    headers = {}
    retry_after = result.error.get("details", {}).get("retry_after_s")
    if retry_after is not None:
        headers["Retry-After"] = str(int(retry_after))
    return JSONResponse(status_code=status, content={"error": result.error}, headers=headers)
