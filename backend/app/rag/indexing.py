"""Unités d'index de production (configuration retenue en phase 4).

Chaque article devient un ensemble d'**unités** vectorisées, qui renvoient toutes à une
**section parente** (≤ 1 024 tokens), le texte effectivement transmis à l'agent :

- `enfant` : morceau découpé selon le type de la section (texte, liste, partie), précédé
  de son fil d'Ariane ;
- `resume` : résumé de la section généré par le LLM (index multi-vecteurs).

Mesure (phase 4, 50 requêtes) : recall@5 = 0,90 contre 0,78 pour le RAG de la phase 2,
6 requêtes gagnées, aucune perdue (McNemar, p = 0,031).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from app.rag.chunking import Chunk, TokenCounter
from app.rag.corpus import Article
from app.rag.typed_chunking import typed_hierarchical_chunks

UnitKind = Literal["enfant", "resume"]


@dataclass(frozen=True)
class IndexUnit:
    id: str
    kind: UnitKind
    text: str  # texte vectorisé
    parent_id: str
    parent_text: str  # texte renvoyé à l'agent
    chunk: Chunk  # métadonnées (article, section, ouverture, date, version…)


def build_units(
    article: Article, count: TokenCounter, summaries: Mapping[str, str] | None = None
) -> list[IndexUnit]:
    """Unités d'un article. `summaries` associe le texte d'une section à son résumé."""
    units: list[IndexUnit] = []
    first_child: dict[str, IndexUnit] = {}
    for item in typed_hierarchical_chunks(article, count):
        unit = IndexUnit(
            id=item.child.id,
            kind="enfant",
            text=item.child.text,
            parent_id=item.parent_id,
            parent_text=item.parent,
            chunk=item.child,
        )
        units.append(unit)
        first_child.setdefault(item.parent_id, unit)

    for parent_id, child in first_child.items():
        summary = (summaries or {}).get(child.parent_text, "").strip()
        if summary:
            units.append(
                IndexUnit(
                    id=f"resume:{parent_id}",
                    kind="resume",
                    text=summary,
                    parent_id=parent_id,
                    parent_text=child.parent_text,
                    chunk=child.chunk,
                )
            )
    return units


def parent_sections(article: Article, count: TokenCounter) -> list[tuple[str, str]]:
    """(fil d'Ariane, texte) de chaque section parente : ce que le LLM doit résumer."""
    sections: dict[str, tuple[str, str]] = {}
    for item in typed_hierarchical_chunks(article, count):
        sections.setdefault(item.parent_id, (item.child.section, item.parent))
    return list(sections.values())
