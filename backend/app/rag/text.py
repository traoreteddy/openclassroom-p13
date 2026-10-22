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


# Un coup juste après un numéro : « e4 », « exd5 », « Cf3 », « Cbd7 », « Fxb4 », « O-O »
_NEXT_MOVE = r"(?=[RDTFC][a-h]?[1-8]?x?[a-h][1-8]|[a-h](?:x[a-h])?[1-8]|O-O|0-0)"
_WHITE_NUMBER = re.compile(r"\b(\d{1,3})\.\s+" + _NEXT_MOVE)
_BLACK_NUMBER = re.compile(r"\b(\d{1,3})\s*(?:\.\.\.|…)\s*" + _NEXT_MOVE)


def normalize_moves(text: str) -> str:
    """« 1. e4 » → « 1.e4 », « 4… Cd7 » / « 3 ... Ff5 » → « 4...Cd7 » / « 3...Ff5 ».

    Les questions écrivent souvent « 1.d4 Cf6 2.c4 » et le corpus « 1. d4 Cf6 2. c4 » : sans
    normalisation, le tokenizer découpe différemment et les vecteurs s'éloignent. Seuls les
    numéros suivis d'un vrai coup sont touchés (« 1987. Les Noirs » reste intact).
    """
    text = _BLACK_NUMBER.sub(r"\1...", text)
    return _WHITE_NUMBER.sub(r"\1.", text)


def normalize_query(text: str) -> str:
    """Normalisation appliquée aux questions avant leur encodage (même règle que le corpus)."""
    return normalize_moves(normalize_notation(text)).strip()


def normalize_text(text: str) -> str:
    """Notation en lettres, numéros de coups collés, espaces simples, au plus une ligne vide."""
    text = normalize_moves(normalize_notation(text))
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


# Coup en notation française : « 4.Cf3 », « 4...Fc5 », « exd4 », « O-O », « Dxf7+ »
_MOVE = re.compile(
    r"^(\d+\.{1,3})?(O-O(-O)?|[RDTFC]?[a-h]?[1-8]?x?[a-h][1-8](=[DTFC])?)[+#!?]*[,;.)]*$"
)


def move_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return sum(bool(_MOVE.match(word)) for word in words) / len(words)
