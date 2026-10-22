"""Phase 1, étape 1.5 : construit Décision-50 (faut-il chercher dans la base ?).

- 30 questions « dans_corpus » : celles de Réf-30, dont la réponse est dans le corpus ;
- 20 questions où la recherche est inutile :
  - 8 « echecs_hors_corpus » : sujet d'échecs absent du corpus (vérifié par mot-clé) ;
  - 6 « autre_outil » : un autre outil répond (évaluation, théorie, légalité, vidéo) ;
  - 6 « hors_sujet » : sans rapport avec les échecs, ou simple échange.

Les recherches sur « autre_outil » et « hors_sujet » sont évitables ; celles sur
« echecs_hors_corpus » ne le sont pas toujours (l'agent ignore le contenu du corpus).
Les deux taux sont mesurés séparément (phase 5).

    uv run python -m scripts.build_decision50
"""

from __future__ import annotations

from collections import Counter

from app.rag.corpus import load_corpus
from app.rag.eval.datasets import EVAL_DIR, DecisionQuestion, RefQuestion, load_jsonl, save_jsonl
from app.rag.text import match_key
from scripts.fetch_corpus import CORPUS_DIR

# (question, mots-clés qui doivent être absents du corpus)
CHESS_OUT_OF_CORPUS: list[tuple[str, list[str]]] = [
    ("Quelles sont les idées du gambit letton pour les Noirs ?", ["gambit letton"]),
    ("Qu'est-ce que le pat aux échecs, et pourquoi donne-t-il partie nulle ?", ["pat "]),
    ("Comment mater avec un roi et une tour contre un roi seul ?", ["roi et tour", "tour contre"]),
    ("Quel est le plan des Noirs dans la défense Philidor ?", ["défense philidor"]),
    ("Quels sont les pièges classiques du gambit du roi ?", ["gambit du roi"]),
    (
        "Comment les Blancs organisent-ils leurs pièces dans le système de Londres ?",
        ["système de londres"],
    ),
    ("Contre qui Gukesh a-t-il gagné le titre de champion du monde ?", ["gukesh"]),
    ("Que dit la règle des cinquante coups ?", ["50 coups", "cinquante coups"]),
]

OTHER_TOOL: list[tuple[str, str]] = [
    (
        "Qui est mieux dans cette position : "
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2 ?",
        "evaluate_position",
    ),
    ("Quels coups jouent les maîtres après 1.e4 e5 2.Cf3 Cc6 3.Fb5 ?", "get_opening_theory"),
    (
        "Le coup Fb5 est-il légal dans la position "
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2 ?",
        "inspect_position",
    ),
    ("Tu peux me montrer une vidéo sur la défense sicilienne ?", "find_opening_videos"),
    ("Donne-moi les trois meilleurs coups pour les Blancs après 1.e4 c5.", "evaluate_position"),
    ("Combien de parties de maîtres ont été jouées avec 1.d4 d5 2.c4 ?", "get_opening_theory"),
]

OFF_TOPIC: list[str] = [
    "Quelle est la capitale de l'Australie ?",
    "Bonjour !",
    "Merci beaucoup pour ton aide.",
    "Donne-moi une recette de crêpes.",
    "Quel temps fera-t-il demain à Paris ?",
    "Combien font 17 fois 23 ?",
]


def main() -> None:
    corpus_text = " ".join(match_key(article.text) for article in load_corpus(CORPUS_DIR))
    ref30 = load_jsonl(RefQuestion, EVAL_DIR / "ref30.jsonl")

    items: list[DecisionQuestion] = []

    def add(question: str, category: str, useful: bool, note: str | None = None) -> None:
        items.append(
            DecisionQuestion(
                id=f"dec-{len(items) + 1:02d}",
                question=question,
                category=category,  # type: ignore[arg-type]
                search_useful=useful,
                note=note,
            )
        )

    for ref in ref30:
        add(ref.question, "dans_corpus", True, f"question {ref.id} de Réf-30")
    for question, keywords in CHESS_OUT_OF_CORPUS:
        present = [kw for kw in keywords if kw in corpus_text]
        if present:
            raise SystemExit(f"« {question} » : {present} présent(s) dans le corpus")
        add(question, "echecs_hors_corpus", False, f"absent du corpus : {', '.join(keywords)}")
    for question, tool in OTHER_TOOL:
        add(question, "autre_outil", False, f"outil attendu : {tool}")
    for question in OFF_TOPIC:
        add(question, "hors_sujet", False)

    save_jsonl(items, EVAL_DIR / "decision50.jsonl")
    print(f"Décision-50 : {len(items)} questions →", EVAL_DIR / "decision50.jsonl")
    print(dict(Counter(item.category for item in items)))


if __name__ == "__main__":
    main()
