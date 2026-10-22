"""Passages candidats pour générer les questions de référence (phase 1, étape 1.2).

Un passage = un paragraphe d'article, avec son chemin de section. Les sections sans
contenu exploitable (références, bibliographie, liens) sont écartées, ainsi que les
paragraphes trop courts pour contenir une réponse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.corpus import Article
from app.rag.text import normalize_text

EXCLUDED_SECTIONS = {
    "notes et références",
    "références",
    "notes",
    "bibliographie",
    "liens externes",
    "voir aussi",
    "articles connexes",
    "annexes",
}
MIN_CHARS = 200
MAX_MOVE_RATIO = 0.25  # au-delà, le passage est surtout une suite de coups
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
# Coup en notation française : « 4.Cf3 », « 4...Fc5 », « exd4 », « O-O », « Dxf7+ »
_MOVE = re.compile(
    r"^(\d+\.{1,3})?(O-O(-O)?|[RDTFC]?[a-h]?[1-8]?x?[a-h][1-8](=[DTFC])?)[+#!?]*[,;.)]*$"
)


@dataclass(frozen=True)
class CandidatePassage:
    article_id: str
    section_path: tuple[str, ...]  # ex. (« Défense sicilienne », « Variantes »)
    text: str

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.section_path)

    @property
    def move_ratio(self) -> float:
        """Part des mots qui sont des coups : 0 pour un texte rédigé, ~1 pour une partie."""
        return move_ratio(self.text)

    @property
    def is_prose(self) -> bool:
        return self.move_ratio <= MAX_MOVE_RATIO


def move_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return sum(bool(_MOVE.match(word)) for word in words) / len(words)


def iter_passages(article: Article, min_chars: int = MIN_CHARS) -> list[CandidatePassage]:
    path: list[str] = []
    passages: list[CandidatePassage] = []
    paragraph: list[str] = []

    def flush() -> None:
        text = normalize_text("\n".join(paragraph))
        paragraph.clear()
        excluded = any(title.lower() in EXCLUDED_SECTIONS for title in path)
        if len(text) >= min_chars and not excluded:
            passages.append(CandidatePassage(article.id, tuple(path), text))

    for line in article.markdown.splitlines():
        heading = _HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            del path[level - 1 :]
            path.append(heading.group(2).strip())
        elif line.strip():
            paragraph.append(line)
        else:
            flush()
    flush()
    return passages


def spread_sample(passages: list[CandidatePassage], count: int) -> list[CandidatePassage]:
    """Sélection déterministe de `count` passages répartis sur tout l'article."""
    if len(passages) <= count:
        return list(passages)
    step = len(passages) / count
    return [passages[int(i * step + step / 2)] for i in range(count)]
