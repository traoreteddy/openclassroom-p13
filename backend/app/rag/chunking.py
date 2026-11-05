"""Découpage des articles en chunks.

Stratégie de référence : **récursif, 512 tokens, recouvrement 10 %**, tokens comptés avec
le tokenizer du modèle d'embedding (la limite porte sur ce que le modèle lit vraiment).

Séparateurs, du plus fort au plus faible, adaptés au français. Ce sont des expressions
régulières qui ne consomment que l'espace : la ponctuation reste avec sa phrase et un
titre ouvre toujours le chunk qui le suit.

    titre Markdown → paragraphe → ligne → fin de phrase (. ? ! …) → ; : → , → espace

Chaque chunk garde sa section (fil d'Ariane) et les métadonnées filtrables de l'article.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.rag.corpus import Article, DocumentMetadata
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

FRENCH_SEPARATORS = [
    r"\n(?=#{1,6} )",  # avant un titre Markdown
    r"\n\n+",  # entre paragraphes
    r"\n",  # entre lignes
    r"(?<=[.!?…])\s+",  # après une fin de phrase
    r"(?<=[;:])\s+",  # après ; ou :
    r"(?<=,)\s+",  # après une virgule
    r"\s+",  # entre mots
    "",  # en dernier recours, entre caractères
]

_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "recursive"
    chunk_size: int = 512  # en tokens
    overlap: int = 51  # en tokens (10 %)
    # Un chunk plus petit est fusionné avec son voisin : un titre seul ou une phrase
    # d'introduction orpheline (« on peut trouver : ») n'apporte rien à la recherche.
    min_tokens: int = 50


@dataclass(frozen=True)
class Chunk:
    id: str
    article_id: str
    title: str
    opening: str
    opening_family: str
    section: str  # fil d'Ariane, ex. « Défense sicilienne > Variantes »
    chunk_index: int
    strategy: str
    text: str
    token_count: int
    metadata: DocumentMetadata = field(compare=False)


@lru_cache(maxsize=2)
def model_token_counter(model_name: str) -> TokenCounter:
    """Compte les tokens avec le tokenizer du modèle d'embedding (sans charger le modèle)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def prepare_document(article: Article) -> str:
    """Markdown normalisé, sans les sections de références."""
    kept: list[str] = []
    skipping_level: int | None = None
    for line in article.markdown.splitlines():
        heading = _HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            if skipping_level is not None and level <= skipping_level:
                skipping_level = None
            if heading.group(2).strip().lower() in EXCLUDED_SECTIONS:
                skipping_level = level
        if skipping_level is None:
            kept.append(line)
    return normalize_text("\n".join(kept))


def section_index(document: str) -> tuple[list[int], list[str]]:
    """Positions des titres et fil d'Ariane valable à partir de chacune."""
    offsets: list[int] = []
    breadcrumbs: list[str] = []
    path: list[str] = []
    for match in _HEADING.finditer(document):
        level = len(match.group(1))
        del path[level - 1 :]
        path.append(match.group(2).strip())
        offsets.append(match.start())
        breadcrumbs.append(" > ".join(path))
    return offsets, breadcrumbs


def merge_small(
    pieces: list[tuple[str, int]], count_tokens: TokenCounter, min_tokens: int
) -> list[tuple[str, int]]:
    """Fusionne chaque morceau trop petit avec le suivant (ou le précédent s'il est le dernier).

    La position retenue pour une fusion vers l'avant est celle du morceau suivant : c'est
    lui qui porte le contenu, donc la section du chunk.
    """
    merged: list[tuple[str, int]] = []
    carry: str | None = None
    for text, start in pieces:
        if carry is not None:
            text = f"{carry}\n{text}"
            carry = None
        if count_tokens(text) < min_tokens:
            carry = text
            continue
        merged.append((text, start))
    if carry is not None:
        if merged:
            last_text, last_start = merged[-1]
            merged[-1] = (f"{last_text}\n{carry}", last_start)
        else:
            merged.append((carry, 0))
    return merged


BASELINE = ChunkingConfig()


def chunk_article(
    article: Article, count_tokens: TokenCounter, config: ChunkingConfig = BASELINE
) -> list[Chunk]:
    document = prepare_document(article)
    splitter = RecursiveCharacterTextSplitter(
        separators=FRENCH_SEPARATORS,
        is_separator_regex=True,
        keep_separator=True,
        chunk_size=config.chunk_size,
        chunk_overlap=config.overlap,
        length_function=count_tokens,
        add_start_index=True,
    )
    offsets, breadcrumbs = section_index(document)
    pieces = [
        (p.page_content, p.metadata["start_index"]) for p in splitter.create_documents([document])
    ]
    chunks: list[Chunk] = []
    for index, (text, start) in enumerate(merge_small(pieces, count_tokens, config.min_tokens)):
        position = bisect.bisect_right(offsets, start) - 1
        section = breadcrumbs[position] if position >= 0 else article.title
        chunks.append(
            Chunk(
                id=f"{config.strategy}:{article.id}:{index:03d}",
                article_id=article.id,
                title=article.title,
                opening=article.opening,
                opening_family=article.opening_family,
                section=section,
                chunk_index=index,
                strategy=config.strategy,
                text=text,
                token_count=count_tokens(text),
                metadata=article.metadata,
            )
        )
    return chunks


# --- Briques de la stratégie hiérarchique ------------------------------------------------

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]


def build(
    article: Article, texts: list[tuple[str, str]], strategy: str, count: TokenCounter
) -> list[Chunk]:
    """Construit les chunks à partir de couples (section, texte)."""
    chunks = []
    for index, (section, text) in enumerate(t for t in texts if t[1].strip()):
        chunks.append(
            Chunk(
                id=f"{strategy}:{article.id}:{index:03d}",
                article_id=article.id,
                title=article.title,
                opening=article.opening,
                opening_family=article.opening_family,
                section=section or article.title,
                chunk_index=index,
                strategy=strategy,
                text=text.strip(),
                token_count=count(text),
                metadata=article.metadata,
            )
        )
    return chunks


def markdown_sections(article: Article) -> list[tuple[str, str]]:
    """(fil d'Ariane, texte) pour chaque section Markdown, titres conservés."""
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=False)
    sections = []
    for doc in splitter.split_text(prepare_document(article)):
        path = [doc.metadata[key] for _, key in HEADERS if key in doc.metadata]
        sections.append((" > ".join(path), doc.page_content))
    return sections


@dataclass(frozen=True)
class HierarchicalChunk:
    """Enfant indexé pour la recherche ; `parent` est le texte renvoyé à l'agent."""

    child: Chunk
    parent_id: str
    parent: str


def with_breadcrumb(breadcrumb: str, text: str) -> str:
    """Préfixe le texte par son fil d'Ariane (sans répéter un titre déjà présent en tête)."""
    first_line = text.lstrip().split("\n", 1)[0].lstrip("# ").strip()
    if first_line and breadcrumb.endswith(first_line):
        text = text.lstrip().split("\n", 1)[1] if "\n" in text.lstrip() else ""
    return f"{breadcrumb}\n\n{text.strip()}".strip()
