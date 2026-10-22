"""Découpage adapté au type de contenu (phase 4, étape 4.4).

Règle de décision : on détecte le type de chaque unité (section d'article, ou document
entier pour les types « procédure », « code », « courriel »), puis on applique la règle
de taille de ce type.

- **texte** (par défaut) : enfants récursifs de 256 tokens, recouvrement 10 %, renvoyant
  à leur section parente (≤ 1 024 tokens).
- **liste** (≥ 60 % de lignes courtes : variantes, codes ECO) : groupes de lignes entières
  (≤ 192 tokens), la phrase d'introduction « … : » étant répétée dans chaque groupe.
- **partie** (> 50 % de coups, ou titre « Exemple(s) de partie ») : une unité par partie
  (en-tête des joueurs + coups), jamais coupée ; une partie unique de plus de 1 024 tokens
  est coupée entre deux coups, son en-tête étant répété.
- **procédure** (au moins 3 lignes « 1. Mot… » ou « Étape 2 : ») : groupes d'étapes entières
  (≤ 512 tokens), titre de la procédure répété.
- **code** (au moins 2 lignes `def`, `class`, `import`…) : une fonction ou une classe entière
  (≤ 512 tokens).
- **courriel** (au moins 2 en-têtes « De : », « Objet : »…) : le message seul, sans citations
  ni signature (≤ 1 024 tokens).

Les trois derniers types n'existent pas dans le corpus Wikipédia : leurs règles sont
vérifiées par des tests sur de petits exemples fictifs, jamais indexés.
"""

from __future__ import annotations

import re
from enum import StrEnum

from app.rag.text import move_ratio

LIST_SHORT_LINE = 80  # caractères
LIST_MIN_SHARE = 0.6
LIST_MIN_LINES = 4
GAME_MOVE_SHARE = 0.5
_GAME_TITLE = re.compile(r"exemples?\s+de\s+parties?|partie\s+(?:modèle|célèbre|type)", re.I)
# Une étape commence par un mot (« 1. Placer… »), pas par un coup (« 1. e4 », « 5. Cf3 »)
_PROCEDURE_STEP = re.compile(
    r"^\s*(?:\d+[.)]|[ÉéEe]tape\s+\d+\s*:)\s+[A-ZÀÂÉÈÊÎÔÛÇ][a-zàâéèêëîïôûüç]{2,}", re.M
)
_EMAIL_HEADER = re.compile(r"^(?:de|from|à|to|objet|subject)\s*:", re.I | re.M)
_CODE_HINT = re.compile(
    r"^\s*(?:def |class |import |from \S+ import |function |const |#include)", re.M
)


class ContentType(StrEnum):
    TEXT = "texte"
    LIST = "liste"
    GAME = "partie"
    PROCEDURE = "procedure"
    CODE = "code"
    EMAIL = "courriel"


def body_lines(text: str) -> list[str]:
    """Lignes de contenu, sans les titres Markdown."""
    return [
        line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]


def classify(text: str, title: str = "") -> ContentType:
    lines = body_lines(text)
    body = "\n".join(lines)
    if len(_EMAIL_HEADER.findall(body)) >= 2:
        return ContentType.EMAIL
    if len(_CODE_HINT.findall(body)) >= 2:
        return ContentType.CODE
    if _GAME_TITLE.search(title) or move_ratio(body) > GAME_MOVE_SHARE:
        return ContentType.GAME
    if len(_PROCEDURE_STEP.findall(body)) >= 3 and move_ratio(body) < 0.2:
        return ContentType.PROCEDURE
    short = [line for line in lines if len(line.strip()) <= LIST_SHORT_LINE]
    if len(lines) >= LIST_MIN_LINES and len(short) / len(lines) >= LIST_MIN_SHARE:
        return ContentType.LIST
    return ContentType.TEXT


# --- Découpage par type ------------------------------------------------------------------

LIST_GROUP_TOKENS = 192


def split_list(text: str, count, max_tokens: int = LIST_GROUP_TOKENS) -> list[str]:
    """Groupes de lignes entières ; la phrase d'introduction (« … : ») est répétée."""
    lines = body_lines(text)
    intro = lines[0] if lines and lines[0].rstrip().endswith(":") else ""
    items = lines[1:] if intro else lines
    groups: list[str] = []
    current: list[str] = []
    for line in items:
        candidate = "\n".join([intro, *current, line]).strip()
        if current and count(candidate) > max_tokens:
            groups.append("\n".join([intro, *current]).strip())
            current = []
        current.append(line)
    if current:
        groups.append("\n".join([intro, *current]).strip())
    return groups


GAME_MAX_TOKENS = 1024
MOVE_LINE_SHARE = 0.5  # une ligne de coups : plus de la moitié des mots sont des coups


# En-tête de partie : des noms séparés par un tiret et une année
# (« Anatoli Karpov-Bent Larsen, Montréal (Canada), 1979 »), tiret simple ou demi-cadratin
_GAME_HEADER = re.compile("\\S\\s*[-\u2013]\\s*\\S.*\\b(1[5-9]\\d\\d|20\\d\\d)\\b")


def is_game_header(line: str) -> bool:
    return bool(_GAME_HEADER.search(line)) and move_ratio(line) <= MOVE_LINE_SHARE


def split_games(text: str) -> list[str]:
    """Une section « parties » → une partie par élément (en-tête + coups + commentaires).

    Une nouvelle partie commence à une ligne d'en-tête qui suit des coups. Les phrases
    d'introduction finissant par « : » placées juste avant l'en-tête (« La partie qui suit
    a introduit… : ») passent avec la partie suivante ; les analyses qui suivent une partie
    (« Les Noirs sont perdus : 29... Rh8… ») restent avec elle.
    """
    games: list[list[str]] = [[]]
    seen_moves = False
    for line in body_lines(text):
        if is_game_header(line) and seen_moves:
            intro: list[str] = []
            while (
                games[-1]
                and games[-1][-1].rstrip().endswith(":")
                and not (move_ratio(games[-1][-1]) > MOVE_LINE_SHARE)
            ):
                intro.insert(0, games[-1].pop())
            games.append(intro)
            seen_moves = False
        games[-1].append(line)
        seen_moves = seen_moves or move_ratio(line) > MOVE_LINE_SHARE
    return ["\n".join(game) for game in games if game]


def split_long_game(game: str, count, max_tokens: int = GAME_MAX_TOKENS) -> list[str]:
    """Une partie trop longue est coupée entre deux coups, en-tête répété à chaque morceau."""
    if count(game) <= max_tokens:
        return [game]
    lines = game.splitlines()
    header = "\n".join(line for line in lines if move_ratio(line) <= MOVE_LINE_SHARE)
    moves = " ".join(line for line in lines if move_ratio(line) > MOVE_LINE_SHARE).split()
    parts: list[str] = []
    current: list[str] = []
    for move in moves:
        if current and count(f"{header}\n{' '.join([*current, move])}") > max_tokens:
            parts.append(f"{header}\n{' '.join(current)}")
            current = []
        current.append(move)
    if current:
        parts.append(f"{header}\n{' '.join(current)}")
    return parts


PROCEDURE_GROUP_TOKENS = 512
CODE_CHUNK_TOKENS = 512
EMAIL_MAX_TOKENS = 1024
_QUOTE_START = re.compile(
    r"^(?:>|le .+ a écrit\s*:|on .+ wrote\s*:|-{2,}\s*message d'origine)", re.I
)
_SIGNATURE = re.compile(r"^(?:--\s*$|cordialement|bien à vous|bonne journée)", re.I)


def split_procedure(text: str, count, max_tokens: int = PROCEDURE_GROUP_TOKENS) -> list[str]:
    """Groupes d'étapes entières ; le titre (lignes avant la 1re étape) est répété."""
    lines = text.splitlines()
    first_step = next((i for i, line in enumerate(lines) if _PROCEDURE_STEP.match(line)), 0)
    title = "\n".join(lines[:first_step]).strip()
    steps: list[str] = []
    for line in lines[first_step:]:
        if _PROCEDURE_STEP.match(line) or not steps:
            steps.append(line)
        else:
            steps[-1] += "\n" + line  # suite d'une étape sur plusieurs lignes
    groups: list[str] = []
    current: list[str] = []
    for step in steps:
        if current and count("\n".join([title, *current, step])) > max_tokens:
            groups.append("\n".join([title, *current]).strip())
            current = []
        current.append(step)
    if current:
        groups.append("\n".join([title, *current]).strip())
    return groups


def split_code(text: str, count, max_tokens: int = CODE_CHUNK_TOKENS) -> list[str]:
    """Coupe aux frontières de fonctions et de classes (séparateurs du langage Python)."""
    from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.PYTHON, chunk_size=max_tokens, chunk_overlap=0, length_function=count
    )
    return splitter.split_text(text)


def clean_email(text: str) -> str:
    """Le message seul : sans les messages cités en dessous ni la signature."""
    kept: list[str] = []
    for line in text.splitlines():
        if _QUOTE_START.match(line.strip()) or _SIGNATURE.match(line.strip()):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def split_email(text: str, count, max_tokens: int = EMAIL_MAX_TOKENS) -> list[str]:
    message = clean_email(text)
    if count(message) <= max_tokens:
        return [message]
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from app.rag.chunking import FRENCH_SEPARATORS

    return RecursiveCharacterTextSplitter(
        separators=FRENCH_SEPARATORS,
        is_separator_regex=True,
        keep_separator=True,
        chunk_size=max_tokens,
        chunk_overlap=0,
        length_function=count,
    ).split_text(message)


def split_by_type(kind: ContentType, parent: str, count, prose_splitter) -> list[str]:
    """Morceaux à indexer pour un parent, selon son type (règle de décision du module)."""
    if kind is ContentType.LIST:
        return split_list(parent, count)
    if kind is ContentType.GAME:
        return [parent]  # une partie n'est jamais coupée à l'intérieur de son parent
    if kind is ContentType.PROCEDURE:
        return split_procedure(parent, count)
    if kind is ContentType.CODE:
        return split_code(parent, count)
    if kind is ContentType.EMAIL:
        return split_email(parent, count)
    return prose_splitter(parent)


MIN_CHILD_TOKENS = 30


def merge_short(pieces: list[str], count, min_tokens: int) -> list[str]:
    """Fusionne un morceau trop court (titre seul, phrase orpheline) avec le suivant,
    ou avec le précédent s'il est le dernier : il serait vide de sens une fois indexé."""
    merged: list[str] = []
    carry = ""
    for piece in pieces:
        piece = f"{carry}\n{piece}".strip() if carry else piece
        carry = ""
        if count(piece) < min_tokens:
            carry = piece
            continue
        merged.append(piece)
    if carry:
        if merged:
            merged[-1] = f"{merged[-1]}\n{carry}"
        else:
            merged.append(carry)
    return merged


def typed_hierarchical_chunks(
    article,
    count,
    *,
    child_size: int = 256,
    child_overlap: int = 25,
    parent_max_tokens: int = 1024,
    breadcrumb: bool = True,
    min_child_tokens: int = MIN_CHILD_TOKENS,
):
    """Stratégie hiérarchique dont les enfants sont découpés selon le type de la section."""
    from dataclasses import replace

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from app.rag.chunking import (
        FRENCH_SEPARATORS,
        HierarchicalChunk,
        build,
        markdown_sections,
        with_breadcrumb,
    )

    def recursive(size: int, overlap: int) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            separators=FRENCH_SEPARATORS,
            is_separator_regex=True,
            keep_separator=True,
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=count,
        )

    children_splitter = recursive(child_size, child_overlap)
    result: list[HierarchicalChunk] = []
    parent_index = 0
    for section, section_text in markdown_sections(article):
        kind = classify(section_text, section.split(" > ")[-1])
        if kind is ContentType.GAME:
            # Une partie = un parent, jamais coupée (sauf partie unique très longue).
            parents = [
                part
                for game in split_games(section_text)
                for part in split_long_game(game, count, parent_max_tokens)
            ]
        elif count(section_text) > parent_max_tokens:
            parents = recursive(parent_max_tokens, 0).split_text(section_text)
        else:
            parents = [section_text]
        for parent in parents:
            parent_id = f"parent:{article.id}:{parent_index:03d}"
            parent_index += 1
            pieces = merge_short(
                split_by_type(kind, parent, count, children_splitter.split_text),
                count,
                min_child_tokens,
            )
            if breadcrumb:
                pieces = [with_breadcrumb(section or article.title, p) for p in pieces]
            for child in build(article, [(section, p) for p in pieces], f"typed-{kind}", count):
                child = replace(
                    child, id=f"typed:{article.id}:{len(result):03d}", chunk_index=len(result)
                )
                result.append(HierarchicalChunk(child=child, parent_id=parent_id, parent=parent))
    return result
