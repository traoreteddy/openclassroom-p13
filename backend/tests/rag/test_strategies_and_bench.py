"""Stratégies de découpage de la phase 3 et banc de comparaison (sans modèle réel)."""

from __future__ import annotations

import numpy as np
import pytest

from app.rag.eval.bench import (
    CachedEmbedder,
    MemoryIndex,
    benchmark_queries,
    chunk_cohesion,
    cut_similarity,
    found_within_budget,
    run_strategy,
)
from app.rag.eval.datasets import ExpectedPassage, RefQuestion
from app.rag.eval.late_chunking import spans
from app.rag.eval.strategies import (
    fixed_chunks,
    hierarchical_chunks,
    html_chunks,
    markdown_chunks,
    markdown_sections,
    recursive_chunks,
    semantic_chunks,
    sentences,
)
from tests.rag.test_chunking import MARKDOWN, SENTENCE, words
from tests.rag.test_text_and_passages import article


class WordTokenizer:
    """Faux tokenizer : un mot = un token."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


class KeywordModel:
    """Faux modèle d'embedding : un axe par mot-clé (sicilienne, evans, dame, autre)."""

    KEYWORDS = ("sicilienne", "evans", "dame")

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts, *, prompt_name=None, normalize_embeddings=True, **kwargs):
        self.calls += 1
        vectors = []
        for text in texts:
            lowered = text.lower()
            v = np.array([lowered.count(k) for k in self.KEYWORDS] + [0.1], dtype=np.float32)
            vectors.append(v / np.linalg.norm(v))
        return np.stack(vectors)


# --- Stratégies --------------------------------------------------------------------------


def test_fixed_windows_ignore_text_structure():
    chunks = fixed_chunks(article(MARKDOWN), words, tokenizer=WordTokenizer(), size=20, overlap=5)

    assert all(c.token_count <= 20 for c in chunks)
    assert chunks[0].strategy == "fixed-20-5"
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-5:] == second[:5]  # recouvrement de 5 mots


def test_recursive_strategy_name_carries_parameters():
    chunks = recursive_chunks(article(MARKDOWN), words, size=60, overlap=6, min_tokens=10)

    assert chunks[0].strategy == "recursive-60-6"


def test_markdown_sections_keep_titles_and_breadcrumbs():
    sections = markdown_sections(article(MARKDOWN))
    chunks = markdown_chunks(article(MARKDOWN), words)

    assert [s for s, _ in sections] == [
        "Défense test",
        "Défense test > Variantes > Najdorf",
        "Défense test > Histoire",
    ]
    assert chunks[1].text.startswith("## Variantes")
    assert all("Référence 1" not in c.text for c in chunks)


HTML = """
<div class="mw-parser-output">
<p>Intro du gambit.</p>
<div class="mw-heading mw-heading2"><h2 id="Histoire">Histoire</h2></div>
<p>Le gambit porte le nom de <a href="/wiki/W">William Evans</a>. Il fut joué en 1827.</p>
<div class="mw-heading"><h2>Notes et références</h2></div>
<p>Source 1.</p>
</div>
"""


def test_html_sections_drop_header_only_documents_and_references():
    page = article("# T\n\ntexte")
    page = page.model_copy(update={"html": HTML})

    chunks = html_chunks(page, words)
    texts = [c.text for c in chunks]

    assert "Histoire" not in texts  # titre seul : artefact retiré
    # HTML brut : le texte du lien est déplacé en fin de bloc, la phrase est cassée
    assert not any("nom de William Evans" in t for t in texts)
    assert any(t.endswith("William Evans") for t in texts)
    flat = html_chunks(page, words, preprocess=True)
    assert any("nom de William Evans" in c.text for c in flat)
    assert flat[0].strategy == "html-sections-flat"
    assert all("Source 1" not in t for t in texts)
    assert any(c.section == "Test > Histoire" for c in chunks)


def test_semantic_chunker_cuts_where_topic_changes():
    text = "# Test\n\n" + " ".join(
        ["La défense sicilienne est populaire."] * 4 + ["Le gambit Evans est ancien."] * 4
    )
    embedder = CachedEmbedder(KeywordModel(), cache_dir=None)  # type: ignore[arg-type]

    chunks = semantic_chunks(article(text), words, embeddings=embedder, percentile=90)

    assert len(chunks) >= 2
    assert "sicilienne" in chunks[0].text.lower()
    assert "evans" in chunks[-1].text.lower()
    assert all("sicilienne" not in c.text.lower() for c in chunks[1:])


def test_hierarchical_children_point_to_their_parent():
    items = hierarchical_chunks(article(MARKDOWN), words, child_size=20, child_overlap=0)

    assert len({i.parent_id for i in items}) == 3  # une par section
    for item in items:
        assert item.child.token_count <= 20
        assert item.child.text in item.parent or item.child.text.strip() in item.parent


def test_hierarchical_parents_can_be_capped():
    long = "# Long\n\n" + " ".join([SENTENCE] * 30)

    uncapped = hierarchical_chunks(article(long), words, child_size=30)
    capped = hierarchical_chunks(article(long), words, child_size=30, parent_max_tokens=100)

    assert len({i.parent_id for i in uncapped}) == 1
    assert len({i.parent_id for i in capped}) > 1
    assert max(words(i.parent) for i in capped) <= 100


def test_sentences_split_on_french_punctuation_and_drop_fragments():
    text = "Première phrase ici. Deuxième phrase là ! Et la suite… Fin courte"

    assert sentences(text) == ["Première phrase ici.", "Deuxième phrase là !", "Et la suite…"]


# --- Banc --------------------------------------------------------------------------------


@pytest.fixture
def embedder(tmp_path):
    return CachedEmbedder(KeywordModel(), cache_dir=tmp_path)


def test_cache_avoids_reencoding_even_across_instances(tmp_path):
    model = KeywordModel()
    first = CachedEmbedder(model, cache_dir=tmp_path)
    first.encode(["la sicilienne", "le gambit evans"])
    first.encode(["la sicilienne"])

    second = CachedEmbedder(model, cache_dir=tmp_path)
    second.encode(["la sicilienne"])

    assert first.encoded == 2
    assert second.encoded == 0  # relu depuis le disque
    assert model.calls == 1


def test_query_and_document_are_cached_separately(embedder):
    embedder.encode(["texte"])
    embedder.encode(["texte"], query=True)

    assert embedder.encoded == 2
    assert len(embedder.embed_documents(["texte"])) == 1
    assert len(embedder.embed_query("texte")) == 4


def make_chunks():
    from app.rag.eval.strategies import build

    page = article("# T\n\nx")
    return build(
        page,
        [
            ("", "La sicilienne est une défense."),
            ("", "Le gambit evans est ancien."),
            ("", "Le gambit dame est fermé."),
        ],
        "t",
        words,
    )


def test_memory_index_ranks_and_filters(embedder):
    chunks = make_chunks()
    index = MemoryIndex(chunks, embedder.encode([c.text for c in chunks]))
    query = embedder.encode(["evans"], query=True)[0]

    hits = index.search(query, 2, None)
    filtered = index.search(query, 2, "Autre ouverture")

    assert hits[0].text.startswith("Le gambit evans")
    assert len(hits) == 2
    assert filtered == []


def test_memory_index_returns_distinct_parents(embedder):
    chunks = make_chunks()
    parents = {c.id: ("p1", "PARENT") for c in chunks}
    index = MemoryIndex(chunks, embedder.encode([c.text for c in chunks]), parents)

    hits = index.search(embedder.encode(["evans"], query=True)[0], 5, None)

    assert [(h.chunk_id, h.text) for h in hits] == [("p1", "PARENT")]


def question(excerpt: str) -> RefQuestion:
    return RefQuestion(
        id="ref-01",
        question="evans ?",
        opening_family=None,
        expected=[ExpectedPassage(article_id="test", excerpt=excerpt)],
        answer="",
        provenance="redige+verifie",
    )


def test_budget_limits_the_text_sent(embedder):
    chunks = make_chunks()
    index = MemoryIndex(chunks, embedder.encode([c.text for c in chunks]))
    target = question("Le gambit dame est fermé et classique")  # absent : jamais trouvé
    evans = question("Le gambit evans est ancien.")

    assert found_within_budget(index, embedder, words, evans, budget=10)
    assert not found_within_budget(index, embedder, words, evans, budget=3)
    assert not found_within_budget(index, embedder, words, target, budget=100)


def test_cohesion_and_cut_similarity(embedder):
    chunks = make_chunks()
    same_topic = make_chunks()[1:2] + make_chunks()[1:2]

    assert chunk_cohesion(chunks, embedder) == 0.0  # chunks d'une phrase : ignorés
    assert 0 <= cut_similarity(chunks, embedder) < 0.5  # coupures entre sujets différents
    assert cut_similarity(same_topic, embedder) > 0.99


async def test_run_strategy_reports_all_metrics(embedder):
    chunks = make_chunks()
    queries = [question("Le gambit evans est ancien.")]

    result = await run_strategy("test", chunks, [article("# T\n\nx")], embedder, words, queries)

    assert result.chunks == 3
    assert result.reachable == 1.0
    assert result.recall_all == 1.0
    assert result.mrr_all == 1.0
    assert result.context_tokens > 0
    assert result.cohesion is None


def test_benchmark_queries_include_multi_hop_sub_questions():
    queries = benchmark_queries()

    assert len(queries) == 50
    assert sum(q.id.startswith("multi-") for q in queries) == 20


def test_late_chunking_spans_point_into_the_document():
    document, pieces = spans(article(MARKDOWN), words, size=40, overlap=0)

    for start, text in pieces:
        assert document[start : start + len(text)] == text
