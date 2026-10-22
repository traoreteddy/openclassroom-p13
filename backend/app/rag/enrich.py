"""Enrichissement des sections par le LLM local : questions hypothétiques et résumé.

Pour chaque section (parent), phi4-mini génère en un appel, en sortie structurée :
- 3 **questions hypothétiques** auxquelles la section répond ;
- un **résumé** de 2 phrases.

Chacun devient un vecteur supplémentaire qui renvoie à la section (index multi-vecteurs) :
une question d'utilisateur ressemble souvent plus à une autre question qu'à un paragraphe.

Les générations sont mises en cache (clé = empreinte du texte de la section) : le coût
n'est payé qu'une fois, et une section inchangée n'est jamais régénérée.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.rag.llm import OllamaChat

logger = logging.getLogger(__name__)

ENRICH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
        "summary": {"type": "string"},
    },
    "required": ["questions", "summary"],
}

SYSTEM_PROMPT = """\
Tu prépares un index de recherche sur les ouvertures d'échecs.
On te donne une section d'article et son emplacement (fil d'Ariane).

Écris :
- "questions" : 3 questions variées qu'un joueur pourrait poser et auxquelles CETTE section
  répond (idée, plan, histoire, coups, joueurs). Chaque question est autonome : elle nomme
  l'ouverture ou la variante. Elle se termine par « ? ».
- "summary" : un résumé de la section en 2 phrases, sans rien ajouter qui n'y figure pas.
Réponds uniquement en JSON, en français."""


class Enrichment(BaseModel):
    questions: list[str] = Field(default_factory=list)
    summary: str = ""

    def cleaned(self) -> Enrichment:
        """Écarte les questions inexploitables (trop courtes, sans point d'interrogation)."""
        questions = [q.strip() for q in self.questions if len(q.strip()) >= 15 and "?" in q]
        return Enrichment(questions=questions[:3], summary=self.summary.strip())


def section_key(text: str) -> str:
    """Empreinte du contenu, insensible aux espaces : une normalisation qui ne fait que
    retirer ou ajouter des espaces (ex. « 1. e4 » → « 1.e4 ») ne force pas à régénérer."""
    return hashlib.sha1("".join(text.split()).encode()).hexdigest()


class EnrichmentCache:
    """Cache JSON Lines : une ligne par section, relu au démarrage, complété à chaque appel."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._items: dict[str, Enrichment] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self._items[record["key"]] = Enrichment.model_validate(record["enrichment"])

    def get(self, text: str) -> Enrichment | None:
        return self._items.get(section_key(text))

    def put(self, text: str, enrichment: Enrichment) -> None:
        key = section_key(text)
        self._items[key] = enrichment
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            record = {"key": key, "enrichment": enrichment.model_dump()}
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._items)


MAX_GENERATED_TOKENS = 400  # 3 questions + un résumé de 2 phrases tiennent largement


async def enrich_sections(
    sections: Sequence[tuple[str, str]],
    llm: OllamaChat,
    cache: EnrichmentCache,
    *,
    attempts: int = 2,
) -> dict[str, Enrichment]:
    """(fil d'Ariane, texte) → enrichissement, en ne générant que ce qui manque au cache.

    Une section qui échoue (délai dépassé, JSON invalide ou tronqué) est réessayée, puis
    sautée et signalée : elle sera indexée sans résumé, le reste du traitement continue.
    """
    result: dict[str, Enrichment] = {}
    for breadcrumb, text in sections:
        cached = cache.get(text)
        for attempt in range(1, attempts + 1):
            if cached is not None:
                break
            try:
                raw = await llm.structured(
                    SYSTEM_PROMPT,
                    f"Emplacement : {breadcrumb}\n\nSection :\n{text}",
                    ENRICH_SCHEMA,
                    temperature=0.0 if attempt == 1 else 0.3,
                )
                cached = Enrichment.model_validate(raw).cleaned()
                cache.put(text, cached)
            except (httpx.HTTPError, ValueError, ValidationError) as exc:
                logger.warning(
                    "Enrichissement échoué (%s, essai %d) : %s", breadcrumb, attempt, exc
                )
        if cached is not None:
            result[text] = cached
    return result
