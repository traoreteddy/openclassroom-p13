"""Routes HTTP adossées aux outils de l'agent.

Les routes ne contiennent aucune logique métier : elles appellent l'outil et traduisent
son `ToolResult` en réponse HTTP.

| Résultat de l'outil      | Réponse HTTP                        |
|--------------------------|-------------------------------------|
| ok                       | 200 + données                       |
| erreur `input`           | 422 + erreur                        |
| erreur `unavailable`     | 503 + erreur (+ en-tête Retry-After)|
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
