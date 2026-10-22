"""Découpage de référence : récursif, séparateurs français, sections, petits chunks."""

from __future__ import annotations

import pytest

from app.rag.chunking import (
    ChunkingConfig,
    chunk_article,
    merge_small,
    prepare_document,
    section_index,
)
from tests.rag.test_text_and_passages import article


def words(text: str) -> int:
    """Compteur de « tokens » simple pour les tests : un mot = un token."""
    return len(text.split())


SENTENCE = "Les Noirs contestent le centre et cherchent un jeu actif sur l'aile dame."
MARKDOWN = (
    "# Défense test\n\n"
    + " ".join([SENTENCE] * 6)
    + "\n\n## Variantes\n### Najdorf\n"
    + " ".join([SENTENCE] * 6)
    + "\n\n## Notes et références\nRéférence 1. Référence 2.\n"
    + "\n## Histoire\n"
    + " ".join([SENTENCE] * 3)
)
CONFIG = ChunkingConfig(strategy="test", chunk_size=60, overlap=6, min_tokens=10)


def test_prepare_document_drops_reference_sections_and_normalizes():
    document = prepare_document(article(MARKDOWN + "\n\n2.♘f3"))

    assert "Notes et références" not in document
    assert "Référence 1" not in document
    assert "## Histoire" in document  # la section suivante est conservée
    assert "2.Cf3" in document


def test_section_index_builds_breadcrumbs():
    _, breadcrumbs = section_index("# A\ntexte\n## B\n### C\n## D\n")

    assert breadcrumbs == ["A", "A > B", "A > B > C", "A > D"]


def test_chunks_respect_size_and_carry_metadata():
    chunks = chunk_article(article(MARKDOWN), words, CONFIG)

    assert all(c.token_count <= CONFIG.chunk_size for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].id == "test:test:000"
    assert {c.strategy for c in chunks} == {"test"}
    assert chunks[0].metadata.service_emetteur == "wikipedia-fr"


def test_chunks_know_their_section():
    chunks = chunk_article(article(MARKDOWN), words, CONFIG)
    sections = [c.section for c in chunks]

    assert sections[0] == "Défense test"
    assert "Défense test > Variantes > Najdorf" in sections
    assert sections[-1] == "Défense test > Histoire"


def test_splits_happen_after_sentences_not_inside():
    chunks = chunk_article(article(MARKDOWN), words, CONFIG)

    for chunk in chunks:
        assert not chunk.text.startswith((".", ",", ";"))
        assert chunk.text.rstrip().endswith((".", "dame.")) or chunk.text.startswith("#")


def test_headings_open_their_chunk():
    chunks = chunk_article(article(MARKDOWN), words, CONFIG)

    headed = [c for c in chunks if "## Histoire" in c.text]
    assert headed and headed[0].text.startswith("## Histoire")


@pytest.mark.parametrize(
    ("pieces", "expected"),
    [
        (
            [("titre", 0), ("un long paragraphe de contenu", 6)],
            [("titre\nun long paragraphe de contenu", 6)],
        ),
        (
            [("un long paragraphe de contenu", 0), ("fin", 30)],
            [("un long paragraphe de contenu\nfin", 0)],
        ),
        ([("a", 0), ("b", 2)], [("a\nb", 0)]),
        ([("un long paragraphe de contenu", 0)], [("un long paragraphe de contenu", 0)]),
    ],
    ids=["petit-en-tete", "petit-en-fin", "tout-petit", "rien-a-fusionner"],
)
def test_merge_small(pieces, expected):
    assert merge_small(pieces, words, min_tokens=3) == expected


def test_tiny_heading_is_merged_with_its_content():
    markdown = "# Défense test\n\n" + " ".join([SENTENCE] * 6)

    chunks = chunk_article(article(markdown), words, ChunkingConfig(chunk_size=60, overlap=0))

    assert chunks[0].text.startswith("# Défense test\n")
    assert chunks[0].token_count > 3
