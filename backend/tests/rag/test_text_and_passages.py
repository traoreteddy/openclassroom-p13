"""Normalisation du texte et extraction des passages candidats."""

from __future__ import annotations

import pytest

from app.rag.corpus import Article, DocumentMetadata
from app.rag.eval.passages import iter_passages, move_ratio, spread_sample
from app.rag.text import contains_excerpt, match_key, normalize_notation, normalize_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2.♘f3 ♞c6 3.♗c4 ♝c5", "2.Cf3 Cc6 3.Fc4 Fc5"),
        ("♕xf7+ ♔e7", "Dxf7+ Re7"),
        ("♖e1 ♜d8", "Te1 Td8"),
        ("8.♙d4", "8.d4"),
        ("1.e4 e5 2.Cf3", "1.e4 e5 2.Cf3"),
    ],
)
def test_figurines_become_french_letters(raw, expected):
    assert normalize_notation(raw) == expected


def test_normalize_text_spaces_and_blank_lines():
    raw = "  Ligne   un\u00a0avec\u202fespaces  \n\n\n\nLigne deux\n"

    assert normalize_text(raw) == "Ligne un avec espaces\n\nLigne deux"


def test_excerpt_matches_across_case_lines_and_notation():
    chunk = "Texte.\nLes noirs répondent 4...Fc5\n5.Cc3 d6, avec une symétrie."

    assert contains_excerpt(chunk, "les Noirs répondent 4...♝c5 5.♘c3 d6")
    assert not contains_excerpt(chunk, "les Noirs répondent 4...Fe7")
    assert match_key("A\nB") == "a b"


def article(markdown: str) -> Article:
    return Article(
        id="test",
        title="Test",
        opening="Test Opening",
        opening_family="Test Opening",
        metadata=DocumentMetadata(
            date="2026-01-01T00:00:00Z",
            version=1,
            service_emetteur="wikipedia-fr",
            confidentialite="public",
            type_document="article",
            licence="CC BY-SA 4.0",
            source_url="https://example.test",
        ),
        fetched_at="2026-01-01T00:00:00+00:00",
        text=markdown,
        markdown=markdown,
        html="",
    )


PROSE = "Les Noirs contestent le centre et cherchent un jeu actif sur l'aile dame. " * 4
MOVES = "1.e4 c5 2.Cf3 d6 3.d4 cxd4 4.Cxd4 Cf6 5.Cc3 a6 6.Fe2 e5 7.Cb3 Fe7 " * 4


def test_passages_keep_section_path_and_skip_references():
    markdown = (
        f"# Défense test\n\n{PROSE}\n\n## Variantes\n### Najdorf\n{PROSE}\n\n"
        f"## Notes et références\n{PROSE}\n\n## Histoire\ncourt\n"
    )

    passages = iter_passages(article(markdown))

    assert [p.section_path for p in passages] == [
        ("Défense test",),
        ("Défense test", "Variantes", "Najdorf"),
    ]
    assert passages[1].breadcrumb == "Défense test > Variantes > Najdorf"


def test_prose_filter_rejects_move_lists():
    markdown = f"# Défense test\n\n{PROSE}\n\n{MOVES}\n"

    prose, moves = iter_passages(article(markdown))

    assert prose.is_prose
    assert not moves.is_prose
    assert move_ratio(MOVES) > 0.9
    assert move_ratio("") == 0.0


@pytest.mark.parametrize(("available", "count", "expected"), [(10, 3, 3), (2, 3, 2), (3, 3, 3)])
def test_spread_sample(available, count, expected):
    passages = iter_passages(article("\n\n".join(f"{i} {PROSE}" for i in range(available))))

    sample = spread_sample(passages, count)

    assert len(sample) == expected
    assert len({p.text for p in sample}) == expected
