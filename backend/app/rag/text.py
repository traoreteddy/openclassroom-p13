"""Normalisation du texte, partagée par l'ingestion et l'évaluation.

Wikipédia mélange deux notations des coups : lettres françaises (« Cf3 ») et figurines
Unicode (« ♘f3 »). Un joueur tape « Cf3 » : on ramène tout à la notation en lettres,
sinon une question ne retrouverait pas les passages écrits avec des figurines.
"""

from __future__ import annotations

import re

# Figurines blanches et noires → lettre française. Le pion n'a pas de lettre en notation.
FIGURINES = {
    "♔": "R",
    "♚": "R",
    "♕": "D",
    "♛": "D",
    "♖": "T",
    "♜": "T",
    "♗": "F",
    "♝": "F",
    "♘": "C",
    "♞": "C",
    "♙": "",
    "♟": "",
}
_FIGURINE_TABLE = str.maketrans(FIGURINES)
_SPACES = re.compile("[ \t\u00a0\u202f]+")  # espaces, tabulations, insécables
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_notation(text: str) -> str:
    """« 2.♘f3 ♞c6 » → « 2.Cf3 Cc6 »."""
    return text.translate(_FIGURINE_TABLE)


def normalize_text(text: str) -> str:
    """Notation en lettres, espaces simples (y compris insécables), au plus une ligne vide."""
    text = normalize_notation(text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANK_LINES.sub("\n\n", text).strip()


def match_key(text: str) -> str:
    """Forme de comparaison : normalisée, sur une seule ligne, en minuscules.

    Sert à vérifier qu'un extrait attendu est contenu dans un chunk, quel que soit le
    découpage (retours à la ligne, casse, figurines).
    """
    return " ".join(normalize_text(text).split()).lower()


def contains_excerpt(chunk: str, excerpt: str) -> bool:
    return match_key(excerpt) in match_key(chunk)
