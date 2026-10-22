"""Découpage par type, unités d'index et enrichissement par le LLM (phase 4).

Les exemples de procédure, de code et de courriel sont FICTIFS : ces types n'existent pas
dans le corpus ; ils servent uniquement à vérifier les règles de découpage.
"""

from __future__ import annotations

import json

import pytest

from app.rag.chunking import with_breadcrumb
from app.rag.enrich import Enrichment, EnrichmentCache, enrich_sections
from app.rag.eval.strategies import add_generated_units, hierarchical_chunks
from app.rag.indexing import build_units, parent_sections
from app.rag.text import normalize_text
from app.rag.typed_chunking import (
    ContentType,
    classify,
    clean_email,
    split_code,
    split_email,
    split_list,
    split_procedure,
    typed_hierarchical_chunks,
)
from tests.rag.test_chunking import MARKDOWN, SENTENCE, words
from tests.rag.test_text_and_passages import article

# --- Exemples fictifs ---------------------------------------------------------------------

PROCEDURE = """Installer l'échiquier avant une partie officielle
1. Placer l'échiquier avec une case blanche dans le coin droit de chaque joueur.
2. Disposer les tours dans les coins, puis les cavaliers à côté.
3. Placer les fous à côté des cavaliers.
4. Mettre la dame sur la case de sa couleur.
5. Placer le roi sur la case restante de la première rangée."""

CODE = """import chess


def legal_moves(fen):
    board = chess.Board(fen)
    return [board.san(m) for m in board.legal_moves]


class Coach:
    def advise(self, fen):
        return legal_moves(fen)[:3]
"""

EMAIL = """De : arbitre@club-exemple.test
À : joueurs@club-exemple.test
Objet : Horaires du tournoi

Bonjour à tous,
Le tournoi commence samedi à 14 h. Merci d'arriver 15 minutes avant.

Cordialement,
L'arbitre
Le 3 mars, Paul a écrit :
> Quel est l'horaire ?"""

LIST = """La partie française est subdivisée en de nombreuses variantes parmi lesquelles :
la variante Tarrasch 3. Cd2
la variante Paulsen 3. Cc3
la variante Winawer 3. Cc3 Fb4
la variante classique 3. Cc3 Cf6
la variante Rubinstein 3. Cc3 dxe4"""

GAME = "1. e4 e5 2. Cf3 Cc6 3. Fc4 Fc5 4. b4 Fxb4 5. c3 Fa5 6. d4 exd4 7. O-O d3 8. Db3 Df6"


@pytest.mark.parametrize(
    ("text", "title", "expected"),
    [
        (SENTENCE * 3, "", ContentType.TEXT),
        (LIST, "", ContentType.LIST),
        (GAME, "", ContentType.GAME),
        ("Kasparov - Karpov, Séville 1987\n" + SENTENCE, "Exemples de parties", ContentType.GAME),
        (PROCEDURE, "", ContentType.PROCEDURE),
        (CODE, "", ContentType.CODE),
        (EMAIL, "", ContentType.EMAIL),
        ("1. e4 e6\n2. d4 d5\n3. Cd2 Cf6\n4. e5 Cfd7", "", ContentType.GAME),  # coups ≠ étapes
    ],
    ids=["texte", "liste", "partie", "titre-partie", "procedure", "code", "courriel", "coups"],
)
def test_classify(text, title, expected):
    assert classify(text, title) is expected


def test_list_groups_whole_lines_and_repeat_the_introduction():
    groups = split_list(LIST, words, max_tokens=22)

    assert len(groups) > 1
    for group in groups:
        assert group.startswith("La partie française est subdivisée")
        assert all(line.startswith(("La partie", "la variante")) for line in group.splitlines())
    assert sum(g.count("la variante") for g in groups) == 5  # aucune ligne perdue ni coupée


def test_list_without_introduction():
    groups = split_list("a b\nc d\ne f\ng h", words, max_tokens=4)

    assert groups == ["a b\nc d", "e f\ng h"]


def test_procedure_keeps_steps_whole_and_repeats_the_title():
    groups = split_procedure(PROCEDURE, words, max_tokens=40)

    assert len(groups) > 1
    assert all(g.startswith("Installer l'échiquier") for g in groups)
    assert sum(g.count("\n2. ") + g.count("\n1. ") for g in groups) == 2
    for group in groups:
        for line in group.splitlines()[1:]:
            assert line[0].isdigit()  # chaque ligne est une étape entière


def test_procedure_multiline_step_stays_together():
    text = "Titre\n1. Première étape\nqui continue ici.\n2. Deuxième étape"

    assert split_procedure(text, words, max_tokens=100) == [text]


def test_code_is_split_at_function_and_class_boundaries():
    chunks = split_code(CODE, words, max_tokens=12)

    assert any(c.startswith("def legal_moves") for c in chunks)
    assert any(c.startswith("class Coach") for c in chunks)


def test_email_drops_signature_and_quoted_thread():
    message = clean_email(EMAIL)

    assert "Le tournoi commence samedi" in message
    assert "Cordialement" not in message
    assert "Quel est l'horaire" not in message
    assert split_email(EMAIL, words) == [message]


def test_long_email_is_split():
    long_email = "Objet : Test\nDe : a@b.test\n" + " ".join([SENTENCE] * 20)

    assert len(split_email(long_email, words, max_tokens=50)) > 1


# --- Hiérarchique typé et unités ------------------------------------------------------------

TYPED_MARKDOWN = (
    "# Défense test\n\n"
    + " ".join([SENTENCE] * 4)
    + "\n\n## Variantes\n"
    + LIST
    + "\n\n## Exemples de parties\n"
    + GAME
)


def test_typed_hierarchy_applies_one_rule_per_section_type():
    items = typed_hierarchical_chunks(
        article(TYPED_MARKDOWN), words, child_size=20, child_overlap=2
    )
    by_type: dict[str, list] = {}
    for item in items:
        by_type.setdefault(item.child.strategy, []).append(item)

    assert set(by_type) == {"typed-texte", "typed-liste", "typed-partie"}
    (game,) = by_type["typed-partie"]
    assert game.child.text.endswith(normalize_text(GAME))  # partie entière, jamais coupée
    assert all("subdivisée" in i.child.text for i in by_type["typed-liste"])
    assert all(i.child.text.startswith("Défense test") for i in items)  # fil d'Ariane


def test_long_sections_are_split_into_bounded_parents():
    long = "# Long\n\n" + " ".join([SENTENCE] * 40)

    items = typed_hierarchical_chunks(article(long), words, parent_max_tokens=100)

    assert len({i.parent_id for i in items}) > 1
    assert max(words(i.parent) for i in items) <= 100


def test_build_units_adds_summaries_pointing_to_their_section():
    page = article(TYPED_MARKDOWN)
    sections = parent_sections(page, words)
    summaries = {sections[0][1]: "Résumé de l'introduction."}

    units = build_units(page, words, summaries)

    summary = [u for u in units if u.kind == "resume"]
    assert len(summary) == 1
    assert summary[0].id == f"resume:{summary[0].parent_id}"
    assert summary[0].parent_text == sections[0][1]
    assert all(u.parent_id.startswith("parent:test:") for u in units)


def test_build_units_without_summaries():
    units = build_units(article(TYPED_MARKDOWN), words, None)

    assert {u.kind for u in units} == {"enfant"}


@pytest.mark.parametrize(
    ("breadcrumb", "text", "expected"),
    [
        ("A > B", "Texte", "A > B\n\nTexte"),
        ("A > B", "## B\nTexte", "A > B\n\nTexte"),  # titre déjà dans le fil : pas répété
        ("A > B", "## C\nTexte", "A > B\n\n## C\nTexte"),
        ("A > B", "## B", "A > B"),
    ],
)
def test_with_breadcrumb(breadcrumb, text, expected):
    assert with_breadcrumb(breadcrumb, text) == expected


def test_generated_units_point_to_the_parent():
    items = hierarchical_chunks(article(MARKDOWN), words, child_size=20, child_overlap=2)
    first_parent = items[0].parent

    enriched = add_generated_units(
        items, {first_parent: ["Question 1 ?", "Question 2 ?"]}, "question", words
    )

    extra = enriched[len(items) :]
    assert [u.child.text for u in extra] == ["Question 1 ?", "Question 2 ?"]
    assert {u.parent for u in extra} == {first_parent}
    assert extra[0].child.strategy == "hierarchical+question"


# --- Enrichissement ------------------------------------------------------------------------


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def structured(self, system, user, schema, *, temperature=0.0):
        self.calls.append(user)
        return {
            "questions": ["Pourquoi joue-t-on cette défense test ?", "trop court", "Sans point"],
            "summary": "  Un résumé.  ",
        }


async def test_enrichment_is_generated_once_then_cached(tmp_path):
    llm = FakeLLM()
    sections = [("Défense test", "Texte A"), ("Défense test > B", "Texte B")]

    first = await enrich_sections(sections, llm, EnrichmentCache(tmp_path / "cache.jsonl"))
    reloaded = EnrichmentCache(tmp_path / "cache.jsonl")
    again = await enrich_sections(sections, llm, reloaded)

    assert len(llm.calls) == 2  # pas de nouvel appel au 2e passage
    assert "Emplacement : Défense test > B" in llm.calls[1]
    assert again == first
    assert len(reloaded) == 2


async def test_enrichment_is_cleaned(tmp_path):
    cache = EnrichmentCache(tmp_path / "cache.jsonl")
    result = await enrich_sections([("T", "Texte")], FakeLLM(), cache)

    assert result["Texte"].questions == ["Pourquoi joue-t-on cette défense test ?"]
    assert result["Texte"].summary == "Un résumé."


def test_cache_file_format(tmp_path):
    cache = EnrichmentCache(tmp_path / "c.jsonl")
    cache.put("Texte", Enrichment(questions=["Q ?"], summary="R"))

    record = json.loads((tmp_path / "c.jsonl").read_text().splitlines()[0])

    assert set(record) == {"key", "enrichment"}
    assert cache.get("Texte") == Enrichment(questions=["Q ?"], summary="R")
    assert cache.get("Autre") is None


# --- Corrections issues de la revue par un pair ---------------------------------------------

GAMES_SECTION = """## Exemples de parties
La partie suivante fit sensation à l'époque :
Anatoli Karpov-Bent Larsen, Montréal (Canada), 1979
1. e4 d5 2. exd5 Dxd5 3. Cc3 Da5 4. d4 Cf6 5. Fd2 Fg4
La partie qui suit a introduit une ligne principale :
Alexeï Chirov-Valeri Salov, Madrid (Espagne), 1997
1. e4 d5 2. exd5 Dxd5 3. Cc3 Da5 4. d4 Cf6 5. Cf3 c6
Les Noirs sont perdus, comme le prouvent les variantes suivantes :
29... Rh8 30. Dh4+ Rxg7 31. Tg1+ Dg6 32. Txg6+."""


def test_games_are_split_one_per_unit_with_their_introduction_and_analysis():
    from app.rag.typed_chunking import split_games

    first, second = split_games(GAMES_SECTION)

    assert "Karpov-Bent Larsen" in first and "Chirov" not in first
    assert second.startswith("La partie qui suit a introduit")  # introduction rattachée
    assert second.rstrip().endswith("32. Txg6+.")  # l'analyse reste avec sa partie


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Anatoli Karpov-Bent Larsen, Montréal (Canada), 1979", True),
        ("Vassili Panov – Alexander Kotov, URSS 1939", True),  # noqa: RUF001 (demi-cadratin)
        ("Les Noirs sont perdus :", False),
        ("1. e4 d5 2. exd5 Dxd5 3. Cc3 1999", False),
    ],
)
def test_game_header_detection(line, expected):
    from app.rag.typed_chunking import is_game_header

    assert is_game_header(line) is expected


def test_a_very_long_game_is_cut_between_moves_with_its_header_repeated():
    from app.rag.typed_chunking import split_long_game

    game = "Kasparov-Karpov, Séville, 1987\n" + " ".join(f"{i}. e4 e5" for i in range(1, 40))

    parts = split_long_game(game, words, max_tokens=30)

    assert len(parts) > 1
    assert all(p.startswith("Kasparov-Karpov, Séville, 1987\n") for p in parts)
    assert all(words(p) <= 30 for p in parts)
    assert split_long_game("En-tête, 2000\n1. e4", words, max_tokens=30) == ["En-tête, 2000\n1. e4"]


def test_typed_hierarchy_never_cuts_a_game():
    page = article("# Défense test\n\n" + " ".join([SENTENCE] * 3) + "\n\n" + GAMES_SECTION)

    games = [
        i for i in typed_hierarchical_chunks(page, words) if i.child.strategy == "typed-partie"
    ]

    assert len(games) == 2
    assert all(i.parent == i.child.text.split("\n\n", 1)[1] for i in games)


@pytest.mark.parametrize(
    ("pieces", "expected"),
    [
        (
            ["## Titre", "un long morceau de texte qui suit"],
            ["## Titre\nun long morceau de texte qui suit"],
        ),
        (["un long morceau de texte ici", "fin"], ["un long morceau de texte ici\nfin"]),
        (["a"], ["a"]),
    ],
)
def test_merge_short(pieces, expected):
    from app.rag.typed_chunking import merge_short

    assert merge_short(pieces, words, min_tokens=3) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1. e4 c6 2. d4 d5 3. Cc3", "1.e4 c6 2.d4 d5 3.Cc3"),
        ("4… Cd7 5. Cg5", "4...Cd7 5.Cg5"),
        ("3 ... Ff5", "3...Ff5"),
        ("9. 0-0 e6 10. O-O-O", "9.0-0 e6 10.O-O-O"),
        ("8. Fxb4 exd5", "8.Fxb4 exd5"),
        ("En 1987. Les Noirs", "En 1987. Les Noirs"),
        ("Il y a 3. Cela suffit", "Il y a 3. Cela suffit"),
    ],
)
def test_move_numbers_are_normalized(raw, expected):
    from app.rag.text import normalize_moves

    assert normalize_moves(raw) == expected


def test_query_normalization_matches_the_corpus():
    from app.rag.text import normalize_query, normalize_text

    assert normalize_query("Que jouer après 1. e4 c5 2. ♘f3 ?") == normalize_text(
        "Que jouer après 1.e4 c5 2.Cf3 ?"
    )


def test_cache_key_ignores_whitespace():
    from app.rag.enrich import section_key

    assert section_key("1. e4 e5\n2. Cf3") == section_key("1.e4 e5 2.Cf3")
    assert section_key("1.e4 e5") != section_key("1.d4 d5")


class FlakyLLM:
    """1er appel : JSON tronqué ; 2e appel : réponse valide (section 1) ; ensuite : délai."""

    def __init__(self) -> None:
        self.calls = 0

    async def structured(self, system, user, schema, *, temperature=0.0):
        import httpx

        self.calls += 1
        if self.calls == 1:
            raise ValueError("Unterminated string")
        if self.calls == 2:
            return {"questions": ["Pourquoi joue-t-on cette ouverture ?"], "summary": "Résumé."}
        raise httpx.ReadTimeout("trop long")


async def test_failing_sections_are_retried_then_skipped(tmp_path, caplog):
    llm = FlakyLLM()
    sections = [("A", "Texte A"), ("B", "Texte B")]

    result = await enrich_sections(sections, llm, EnrichmentCache(tmp_path / "c.jsonl"))

    assert list(result) == ["Texte A"]  # A réussit au 2e essai, B est sautée
    assert llm.calls == 4
    assert "Enrichissement échoué (B, essai 2)" in caplog.text


async def test_llm_options_bound_the_generation(respx_mock, http_client):
    from app.rag.llm import OllamaChat

    route = respx_mock.post("http://ollama.test/api/chat").respond(
        json={"message": {"content": "{}"}}
    )

    await OllamaChat(
        http_client, base_url="http://ollama.test", model="m", max_tokens=400
    ).structured("s", "u", {})

    body = json.loads(route.calls.last.request.content)
    assert body["options"] == {"temperature": 0.0, "num_predict": 400}
