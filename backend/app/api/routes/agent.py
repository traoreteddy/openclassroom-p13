"""Routes de l'agent.

| Route                          | Parcours                                                       |
|--------------------------------|----------------------------------------------------------------|
| POST /api/v1/agent/analyze     | position : Lichess → (Stockfish) → contexte → vidéos → réponse |
| POST /api/v1/agent/ask         | question libre : retrieval décidé par l'agent                  |

Les deux renvoient la même structure : réponse rédigée, données brutes par outil,
appels d'outils effectués (statut, issue), identifiant de trace (Langfuse).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.agent.graph import AgentResult, ChessAgent
from app.tools.fen import Fen

router = APIRouter(prefix="/agent", tags=["agent"])


def get_agent(request: Request) -> ChessAgent:
    return request.app.state.agent


Agent = Annotated[ChessAgent, Depends(get_agent)]


class AnalyzeRequest(BaseModel):
    fen: Fen
    question: str | None = Field(default=None, max_length=500)
    session_id: str | None = Field(default=None, max_length=64)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, max_length=64)


class ToolCallOut(BaseModel):
    name: str
    arguments: dict[str, Any]
    status: str
    outcome: str


class AgentResponse(BaseModel):
    answer: str
    in_theory: bool | None
    opening: str | None
    data: dict[str, Any]
    tool_calls: list[ToolCallOut]
    tokens_in: int
    tokens_out: int
    trace_id: str

    @classmethod
    def from_result(cls, result: AgentResult) -> AgentResponse:
        return cls(
            answer=result.answer,
            in_theory=result.in_theory,
            opening=result.opening,
            data=result.data,
            tool_calls=[
                ToolCallOut(name=c.name, arguments=c.arguments, status=c.status, outcome=c.outcome)
                for c in result.tool_calls
            ],
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            trace_id=result.trace_id,
        )


@router.post(
    "/analyze", summary="Conseil sur une position (Lichess, puis Stockfish si hors théorie)"
)
async def analyze(body: AnalyzeRequest, agent: Agent) -> AgentResponse:
    result = await agent.analyze(body.fen, question=body.question, session_id=body.session_id)
    return AgentResponse.from_result(result)


@router.post("/ask", summary="Question libre à l'agent")
async def ask(body: AskRequest, agent: Agent) -> AgentResponse:
    result = await agent.ask(body.question, session_id=body.session_id)
    return AgentResponse.from_result(result)
