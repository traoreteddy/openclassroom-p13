"""Stratégies de découpage comparées en phase 3.

Toutes prennent un article et renvoient une liste de `Chunk` (même format que le
découpage de référence), ce qui permet de les évaluer avec le même banc.

| Famille      | Principe                                         | Fonction                |
|--------------|--------------------------------------------------|-------------------------|
| Fixe         | fenêtres de N tokens, sans regarder le texte     | `fixed_chunks`          |
| Récursive    | coupe aux séparateurs les plus forts possibles   | `recursive_chunks`      |
| Structurelle | une section (titre Markdown ou HTML) = un chunk  | `markdown_chunks`,      |
|              |                                                  | `html_chunks`           |
| Sémantique   | coupe là où le sens change entre phrases voisines| `semantic_chunks`       |
| Hiérarchique | petits chunks pour chercher, parent renvoyé      | `hierarchical_chunks`   |
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from functools import lru_cache
from typing import Any

from app.rag.chunking import (  # briques partagées avec la production
    EXCLUDED_SECTIONS,
    Chunk,
    ChunkingConfig,
    HierarchicalChunk,
    TokenCounter,
    build,
    chunk_article,
    markdown_sections,
    prepare_document,
    with_breadcrumb,
)
from app.rag.corpus import Article
from app.rag.text import normalize_text

# Fin de phrase en français (y compris les points de suspension « … »)
FRENCH_SENTENCE = r"(?<=[.?!…])\s+"


@lru_cache(maxsize=2)
def model_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


# --- Fixe -------------------------------------------------------------------------------


def fixed_chunks(
    article: Article, count: TokenCounter, *, tokenizer: Any, size: int, overlap: int
) -> list[Chunk]:
    """Fenêtres glissantes de `size` tokens : coupe au milieu des mots et des phrases."""
    ids = tokenizer.encode(prepare_document(article), add_special_tokens=False)
    step = size - overlap
    windows = [ids[start : start + size] for start in range(0, max(len(ids) - overlap, 1), step)]
    texts = [("", tokenizer.decode(window)) for window in windows]
    return build(article, texts, f"fixed-{size}-{overlap}", count)


# --- Récursive --------------------------------------------------------------------------


def recursive_chunks(
    article: Article, count: TokenCounter, *, size: int, overlap: int, min_tokens: int = 50
) -> list[Chunk]:
    config = ChunkingConfig(
        strategy=f"recursive-{size}-{overlap}",
        chunk_size=size,
        overlap=overlap,
        min_tokens=min_tokens,
    )
    return chunk_article(article, count, config)


# --- Structurelle -----------------------------------------------------------------------


def markdown_chunks(article: Article, count: TokenCounter) -> list[Chunk]:
    return build(article, markdown_sections(article), "markdown-sections", count)


_MW_HEADING = re.compile(
    r'<div class="mw-heading[^"]*">\s*(<h[2-6]\b.*?</h[2-6]>).*?</div>', re.DOTALL
)


def unwrap_wikipedia_headings(html: str) -> str:
    """Retire l'enveloppe `<div class="mw-heading">` des titres de Wikipédia.

    Sans cela, HTMLHeaderTextSplitter ne rattache pas le texte qui suit au titre : tout le
    contenu arrive sans section (métadonnées vides).
    """
    return _MW_HEADING.sub(r"\1", html)


_INLINE_TAGS = re.compile(r"</?(a|b|i|em|strong|span|abbr|small|sup|sub)\b[^>]*>", re.IGNORECASE)


def flatten_inline(html: str) -> str:
    """Retire les balises de mise en forme en gardant leur texte.

    HTMLHeaderTextSplitter déplace le texte des liens à la fin du bloc : « Le <a>gambit</a>
    porte le nom. » devient « Le porte le nom. \n gambit ». Les phrases sont cassées, alors
    que sur Wikipédia presque tous les noms propres sont des liens.
    """
    return _INLINE_TAGS.sub("", html)


def html_chunks(article: Article, count: TokenCounter, *, preprocess: bool = False) -> list[Chunk]:
    """Sections du HTML rendu de Wikipédia (h2, h3, h4), texte normalisé.

    `preprocess=True` retire aussi les balises de mise en forme (voir `flatten_inline`).
    """
    from langchain_text_splitters import HTMLHeaderTextSplitter

    splitter = HTMLHeaderTextSplitter(
        headers_to_split_on=[("h2", "h2"), ("h3", "h3"), ("h4", "h4")]
    )
    texts = []
    html = unwrap_wikipedia_headings(article.html)
    if preprocess:
        html = flatten_inline(html)
    for doc in splitter.split_text(html):
        path = [doc.metadata[key] for key in ("h2", "h3", "h4") if key in doc.metadata]
        if any(title.strip().lower() in EXCLUDED_SECTIONS for title in path):
            continue
        text = normalize_text(doc.page_content)
        # Le découpeur émet chaque titre comme un document à part : artefact retiré.
        # Le bruit éditorial de Wikipédia (bandeaux, encadrés) est en revanche conservé :
        # c'est le coût réel d'un découpage à partir du HTML.
        if path and text == path[-1].strip():
            continue
        breadcrumb = " > ".join([article.title, *path])
        texts.append((breadcrumb, text))
    return build(article, texts, "html-sections" + ("-flat" if preprocess else ""), count)


# --- Sémantique -------------------------------------------------------------------------


def semantic_chunks(
    article: Article, count: TokenCounter, *, embeddings: Any, percentile: float = 95
) -> list[Chunk]:
    """`SemanticChunker` : coupe quand la distance entre phrases voisines dépasse le
    percentile donné de toutes les distances de l'article."""
    from langchain_experimental.text_splitter import SemanticChunker

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=percentile,
        sentence_split_regex=FRENCH_SENTENCE,
    )
    texts = [("", text) for text in splitter.split_text(prepare_document(article))]
    return build(article, texts, f"semantic-p{percentile:g}", count)


# --- Hiérarchique -----------------------------------------------------------------------


def hierarchical_chunks(
    article: Article,
    count: TokenCounter,
    *,
    child_size: int = 256,
    child_overlap: int = 25,
    parent_max_tokens: int | None = None,
    breadcrumb: bool = False,
) -> list[HierarchicalChunk]:
    """Parents = sections Markdown ; enfants = découpage récursif de chaque parent.

    Avec `parent_max_tokens`, une section trop longue est découpée (récursivement) en
    plusieurs parents : le texte renvoyé à l'agent reste borné.

    Avec `breadcrumb`, le texte **indexé** de chaque enfant commence par le fil d'Ariane de
    sa section (« Défense slave > Principales variantes > Variante Alapine ») ; le parent
    renvoyé à l'agent est inchangé.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from app.rag.chunking import FRENCH_SEPARATORS

    splitter = RecursiveCharacterTextSplitter(
        separators=FRENCH_SEPARATORS,
        is_separator_regex=True,
        keep_separator=True,
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        length_function=count,
    )
    parents: list[tuple[str, str]] = []
    for section, text in markdown_sections(article):
        if parent_max_tokens is not None and count(text) > parent_max_tokens:
            big = RecursiveCharacterTextSplitter(
                separators=FRENCH_SEPARATORS,
                is_separator_regex=True,
                keep_separator=True,
                chunk_size=parent_max_tokens,
                chunk_overlap=0,
                length_function=count,
            )
            parents.extend((section, part) for part in big.split_text(text))
        else:
            parents.append((section, text))

    result: list[HierarchicalChunk] = []
    for parent_index, (section, text) in enumerate(parents):
        parent_id = f"parent:{article.id}:{parent_index:03d}"
        pieces = splitter.split_text(text)
        if breadcrumb:
            pieces = [with_breadcrumb(section or article.title, piece) for piece in pieces]
        for child in build(article, [(section, t) for t in pieces], "hierarchical", count):
            child = replace(
                child, id=f"hierarchical:{article.id}:{len(result):03d}", chunk_index=len(result)
            )
            result.append(HierarchicalChunk(child=child, parent_id=parent_id, parent=text))
    return result


def add_generated_units(
    items: list[HierarchicalChunk],
    generated: dict[str, list[str]],
    kind: str,
    count: TokenCounter,
    *,
    breadcrumb: bool = False,
) -> list[HierarchicalChunk]:
    """Ajoute des unités d'indexation supplémentaires (questions, résumé) par parent.

    `generated` associe un texte de parent à ses textes générés. Chaque texte généré devient
    un vecteur de plus, qui renvoie au même parent que les enfants (index multi-vecteurs).
    """
    by_parent: dict[str, HierarchicalChunk] = {}
    for item in items:
        by_parent.setdefault(item.parent_id, item)
    extra: list[HierarchicalChunk] = []
    for parent_id, first in by_parent.items():
        for index, text in enumerate(generated.get(first.parent, [])):
            if breadcrumb:
                text = f"{first.child.section}\n\n{text}"
            unit = replace(
                first.child,
                id=f"{kind}:{parent_id}:{index}",
                text=text,
                token_count=count(text),
                strategy=f"hierarchical+{kind}",
            )
            extra.append(HierarchicalChunk(child=unit, parent_id=parent_id, parent=first.parent))
    return items + extra


# --- Mesures descriptives ---------------------------------------------------------------


def sentences(text: str) -> list[str]:
    return [s for s in re.split(FRENCH_SENTENCE, text.replace("\n", " ")) if len(s.split()) >= 3]


StrategyFn = Callable[[Article], list[Chunk]]
