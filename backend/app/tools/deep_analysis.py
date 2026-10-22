"""Outils `start_deep_analysis` / `get_deep_analysis` : analyse longue, exécution différée.

Une analyse Stockfish à profondeur 30+ dure de une à plusieurs minutes : trop long pour
un tour de conversation. On la découpe en deux appels courts :

    start_deep_analysis(fen, depth) ──► {job_id, status: "pending", poll_after_s}
                 │  (la tâche tourne en arrière-plan)
                 ▼
    get_deep_analysis(job_id) ──► {status: "running", …}  … puis {status: "done", result}

La soumission est idempotente par clé (fen, depth) : rappeler start_deep_analysis avec
les mêmes paramètres renvoie le même job_id au lieu de relancer le moteur.
Les tâches sont gardées en mémoire (suffisant pour le POC, perdues au redémarrage).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import ConfigDict, Field

from app.tools.base import SideEffect, ToolInput, ToolOutput, ToolSpec
from app.tools.errors import ToolError, ToolInputError
from app.tools.fen import START_FEN, Fen
from app.tools.resilience import ResiliencePolicy
from app.tools.stockfish import Analyzer, EvaluationOutput, evaluate

JobStatus = Literal["pending", "running", "done", "failed"]
POLL_AFTER_S = 15

START_DESCRIPTION = """\
Lance une analyse Stockfish profonde (profondeur 23 à 40, une à plusieurs minutes) en
tâche de fond et renvoie immédiatement un job_id. Le même appel (même FEN, même
profondeur) renvoie le même job_id.

À utiliser : seulement si le joueur demande explicitement une analyse approfondie.
Exemple : « Analyse cette position à fond » → start_deep_analysis(fen=..., depth=30)

Ne pas utiliser : pour une évaluation courante (→ evaluate_position, réponse en 2 s).
Ensuite : appeler get_deep_analysis(job_id) après poll_after_s secondes.
"""

GET_DESCRIPTION = """\
Renvoie l'état d'une analyse lancée par start_deep_analysis : pending, running, done ou
failed, et son résultat (même format que evaluate_position) quand elle est terminée.

À utiliser : pour suivre une analyse dont on a le job_id.
Exemple : get_deep_analysis(job_id="3f2c…")
Ne pas utiliser : sans job_id (lancer d'abord start_deep_analysis).
Rappeler après poll_after_s secondes tant que le statut n'est pas done ou failed.
"""


class StartDeepAnalysisInput(ToolInput):
    model_config = ConfigDict(json_schema_extra={"examples": [{"fen": START_FEN, "depth": 30}]})

    fen: Fen
    depth: int = Field(default=30, ge=23, le=40, description="Profondeur de recherche.")


class GetDeepAnalysisInput(ToolInput):
    model_config = ConfigDict(json_schema_extra={"examples": [{"job_id": "3f2c9a1e"}]})

    job_id: str = Field(min_length=1, max_length=64)


class JobTicket(ToolOutput):
    job_id: str
    status: JobStatus
    poll_after_s: int


class JobState(ToolOutput):
    job_id: str
    status: JobStatus
    elapsed_s: float
    result: EvaluationOutput | None
    error: str | None
    poll_after_s: int | None


@dataclass
class _Job:
    id: str
    fen: str
    depth: int
    created_at: float
    status: JobStatus = "pending"
    result: EvaluationOutput | None = None
    error: str | None = None
    finished_at: float | None = None
    task: asyncio.Task[None] | None = field(default=None, repr=False)


class DeepAnalysisJobs:
    """Gestionnaire de tâches d'analyse en mémoire."""

    def __init__(
        self,
        analyzer: Analyzer,
        *,
        clock: Callable[[], float] = time.monotonic,
        poll_after_s: int = POLL_AFTER_S,
    ) -> None:
        self._analyzer = analyzer
        self._clock = clock
        self._poll_after_s = poll_after_s
        self._jobs: dict[str, _Job] = {}
        self._by_key: dict[tuple[str, int], str] = {}

    async def start(self, params: StartDeepAnalysisInput) -> JobTicket:
        key = (params.fen, params.depth)
        existing = self._jobs.get(self._by_key.get(key, ""))
        if existing is not None and existing.status != "failed":
            return self._ticket(existing)

        job = _Job(
            id=uuid.uuid4().hex, fen=params.fen, depth=params.depth, created_at=self._clock()
        )
        job.task = asyncio.create_task(self._run(job))
        self._jobs[job.id] = job
        self._by_key[key] = job.id
        return self._ticket(job)

    async def get(self, params: GetDeepAnalysisInput) -> JobState:
        job = self._jobs.get(params.job_id)
        if job is None:
            raise ToolInputError(
                code="unknown_job",
                message=f"Aucune analyse avec le job_id {params.job_id}.",
                hint="Utiliser le job_id renvoyé par start_deep_analysis.",
            )
        end = job.finished_at if job.finished_at is not None else self._clock()
        finished = job.status in ("done", "failed")
        return JobState(
            job_id=job.id,
            status=job.status,
            elapsed_s=round(end - job.created_at, 1),
            result=job.result,
            error=job.error,
            poll_after_s=None if finished else self._poll_after_s,
        )

    async def wait(self, job_id: str) -> None:
        """Attend la fin d'une tâche (tests et arrêt propre)."""
        job = self._jobs[job_id]
        if job.task is not None:
            await job.task

    async def _run(self, job: _Job) -> None:
        job.status = "running"
        try:
            job.result = await evaluate(self._analyzer, job.fen, depth=job.depth, lines=1)
            job.status = "done"
        except ToolError as exc:
            job.status, job.error = "failed", exc.to_model_message()
        except Exception as exc:  # la tâche de fond ne doit jamais planter en silence
            job.status, job.error = "failed", f"Erreur interne : {type(exc).__name__}"
        finally:
            job.finished_at = self._clock()

    def _ticket(self, job: _Job) -> JobTicket:
        return JobTicket(job_id=job.id, status=job.status, poll_after_s=self._poll_after_s)


def build_deep_analysis_tools(jobs: DeepAnalysisJobs) -> list[ToolSpec]:
    quick = ResiliencePolicy.none(timeout_s=1.0)
    return [
        ToolSpec(
            name="start_deep_analysis",
            description=START_DESCRIPTION,
            input_model=StartDeepAnalysisInput,
            output_model=JobTicket,
            handler=jobs.start,
            side_effect=SideEffect.CREATES_JOB,
            idempotent=True,  # par clé (fen, depth)
            policy=quick,
        ),
        ToolSpec(
            name="get_deep_analysis",
            description=GET_DESCRIPTION,
            input_model=GetDeepAnalysisInput,
            output_model=JobState,
            handler=jobs.get,
            side_effect=SideEffect.NONE,
            idempotent=True,
            policy=quick,
        ),
    ]
