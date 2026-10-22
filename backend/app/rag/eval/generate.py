"""Génération des questions de référence avec le LLM local (phase 1, étape 1.3).

Le LLM (phi4-mini via Ollama) reçoit un passage et produit, en sortie structurée
(schéma JSON imposé), une question, l'extrait qui contient la réponse et la réponse.

Garde-fou : l'extrait doit exister **mot pour mot** dans le passage. Sinon le LLM a
paraphrasé ou inventé : on refait une tentative en lui donnant le motif du rejet, puis
on écarte le candidat. Chaque candidat garde son statut pour la relecture (étape 1.4).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from app.rag.eval.passages import CandidatePassage
from app.rag.llm import OllamaChat
from app.rag.text import contains_excerpt

GENERATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "excerpt": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["question", "excerpt", "answer"],
}

SYSTEM_PROMPT = """\
Tu prépares un jeu de test pour un moteur de recherche sur les ouvertures d'échecs.
À partir du passage fourni, écris UNE question qu'un jeune joueur d'échecs pourrait poser,
dont la réponse se trouve dans le passage.

Règles :
- La question est en français, naturelle et autonome : elle nomme l'ouverture concernée.
- Ne mentionne jamais « le passage », « le texte » ou « l'article ».
- La question porte sur une idée, un plan, une histoire ou un fait, pas sur une suite de coups
  à réciter.
- "excerpt" est UNE phrase du passage, recopiée EXACTEMENT, caractère pour caractère, qui
  contient la réponse. Ne la reformule pas.
- "answer" est la réponse, en une phrase courte.
Réponds uniquement en JSON."""


class GeneratedQuestion(BaseModel):
    question: str
    excerpt: str
    answer: str


class Candidate(BaseModel):
    """Résultat de la génération pour un passage, avant relecture."""

    article_id: str
    breadcrumb: str
    passage: str
    status: Literal["valide", "rejete"]
    attempts: int
    question: str | None = None
    excerpt: str | None = None
    answer: str | None = None
    rejection: str | None = None


def user_prompt(passage: CandidatePassage, feedback: str | None = None) -> str:
    prompt = f"Section : {passage.breadcrumb}\n\nPassage :\n{passage.text}"
    if feedback:
        prompt += f"\n\nTa réponse précédente a été refusée : {feedback}"
    return prompt


def check(generated: GeneratedQuestion, passage: CandidatePassage) -> str | None:
    """Motif de rejet, ou None si le candidat est acceptable."""
    if len(generated.excerpt.strip()) < 20:
        return "l'extrait est trop court (moins de 20 caractères)"
    if not contains_excerpt(passage.text, generated.excerpt):
        return "l'extrait n'apparaît pas mot pour mot dans le passage ; recopie une phrase exacte"
    if len(generated.question.strip()) < 15 or not generated.question.strip().endswith("?"):
        return "la question doit être une vraie question, terminée par « ? »"
    return None


async def generate_candidate(
    llm: OllamaChat, passage: CandidatePassage, *, max_attempts: int = 2
) -> Candidate:
    feedback: str | None = None
    generated: GeneratedQuestion | None = None
    for attempt in range(1, max_attempts + 1):
        raw = await llm.structured(
            SYSTEM_PROMPT,
            user_prompt(passage, feedback),
            GENERATION_SCHEMA,
            temperature=0.0 if attempt == 1 else 0.4,
        )
        generated = GeneratedQuestion.model_validate(raw)
        feedback = check(generated, passage)
        if feedback is None:
            return Candidate(
                article_id=passage.article_id,
                breadcrumb=passage.breadcrumb,
                passage=passage.text,
                status="valide",
                attempts=attempt,
                **generated.model_dump(),
            )
    assert generated is not None
    return Candidate(
        article_id=passage.article_id,
        breadcrumb=passage.breadcrumb,
        passage=passage.text,
        status="rejete",
        attempts=max_attempts,
        rejection=feedback,
        **generated.model_dump(),
    )
