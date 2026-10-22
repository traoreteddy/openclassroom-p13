"""Phase 1, étape 1.7 : construit Multi-10 (questions multi-sauts).

Chaque question exige deux recherches successives : la sous-question du saut 2 utilise
la réponse du saut 1 (`depends_on`). Trois chaînes traversent deux articles
(transpositions, filiation) ; les autres relient deux sections d'un même article.
Chaque extrait est vérifié mot pour mot dans l'article.

    uv run python -m scripts.build_multi10
"""

from __future__ import annotations

from collections import Counter

from app.rag.corpus import load_corpus
from app.rag.eval.datasets import (
    EVAL_DIR,
    ExpectedPassage,
    Hop,
    MultiHopQuestion,
    save_jsonl,
)
from app.rag.text import contains_excerpt
from scripts.fetch_corpus import CORPUS_DIR

# (question, [(sous-question, article, extrait), ...], réponse)
CHAINS: list[tuple[str, list[tuple[str, str, str]], str]] = [
    (
        "Quel est le code ECO de la défense où les Noirs répondent 3...Cf6 dans la partie "
        "italienne ?",
        [
            (
                "Comment s'appelle la défense 1.e4 e5 2.Cf3 Cc6 3.Fc4 Cf6 ?",
                "defense-des-deux-cavaliers",
                "Elle est une variante de la partie italienne caractérisée par les coups",
            ),
            (
                "Quel est le code ECO de la défense des deux cavaliers ?",
                "defense-des-deux-cavaliers",
                "Son code ECO s'étend de C55 à C59.",
            ),
        ],
        "C55 à C59 (défense des deux cavaliers).",
    ),
    (
        "Quel métier exerçait l'homme qui a donné son nom au gambit de la partie italienne "
        "joué par 4.b4 ?",
        [
            (
                "Quel gambit de la partie italienne commence par 4.b4 ?",
                "gambit-evans",
                "Le gambit Evans est une ouverture du jeu d'échecs qui commence par les coups : "
                "1. e4 e5 2. Cf3 Cc6 3. Fc4 Fc5 4. b4",
            ),
            (
                "Qui a donné son nom au gambit Evans ?",
                "gambit-evans",
                "Le gambit porte le nom du capitaine de navire gallois William Davies Evans",
            ),
        ],
        "Capitaine de navire (William Davies Evans, gallois).",
    ),
    (
        "En quelle année ont commencé les analyses des deux hommes qui ont donné leur nom à "
        "la défense 1.e4 c6 ?",
        [
            (
                "Comment s'appelle la défense 1.e4 c6 ?",
                "defense-caro-kann",
                "La défense Caro-Kann est une ouverture aux échecs qui se caractérise par les "
                "coups 1. e4 c6.",
            ),
            (
                "Qui a donné son nom à la défense Caro-Kann, et depuis quand ?",
                "defense-caro-kann",
                "Elle doit son nom au joueur anglais Horatio Caro et au théoricien viennois "
                "Marcus Kann qui en commencèrent l'analyse en 1886",
            ),
        ],
        "En 1886 (Horatio Caro et Marcus Kann).",
    ),
    (
        "La variante Steiner de la défense française peut transposer dans une autre défense "
        "après 2...c5 : par quels coups cette autre défense s'obtient-elle normalement ?",
        [
            (
                "Dans quelle défense la variante Steiner de la française transpose-t-elle après "
                "2...c5 ?",
                "defense-francaise",
                "2.c4, la variante Steiner, peut transposer dans la défense sicilienne après 2…c5",
            ),
            (
                "Par quels coups obtient-on la défense sicilienne ?",
                "defense-sicilienne",
                "elle s'obtient après les coups 1. e4 c5.",
            ),
        ],
        "1.e4 c5 (défense sicilienne).",
    ),
    (
        "La défense Caro-Kann peut s'obtenir par transposition depuis une autre ouverture : "
        "par quel coup cette ouverture commence-t-elle ?",
        [
            (
                "Depuis quelle ouverture la Caro-Kann peut-elle être obtenue par transposition ?",
                "defense-caro-kann",
                "La défense Caro-Kann peut aussi être obtenue par transposition des coups de "
                "l'ouverture anglaise",
            ),
            (
                "Par quel coup commence l'ouverture anglaise ?",
                "ouverture-anglaise",
                "elle débute par le coup 1. c4",
            ),
        ],
        "1.c4 (ouverture anglaise).",
    ),
    (
        "Quelle part de l'Encyclopédie des ouvertures d'échecs occupe l'ouverture dont la "
        "défense slave est une variante ?",
        [
            (
                "De quelle ouverture la défense slave est-elle une variante ?",
                "defense-slave",
                "la défense slave (ou les défenses slaves) est une variante du gambit dame",
            ),
            (
                "Quelle part de l'Encyclopédie des ouvertures d'échecs occupe le gambit dame ?",
                "gambit-dame",
                "occupe à lui seul pratiquement la totalité du volume D de l'Encyclopédie des "
                "ouvertures d'échecs",
            ),
        ],
        "Presque tout le volume D (gambit dame).",
    ),
    (
        "En l'honneur de qui est nommée l'ouverture dans laquelle Karpov a joué la variante "
        "Zaïtsev contre Kasparov ?",
        [
            (
                "Dans quelle ouverture Karpov a-t-il joué la variante Zaïtsev contre Kasparov ?",
                "partie-espagnole",
                "Anatoli Karpov usa à sept reprises de la variante Zaïtsev",
            ),
            (
                "Pourquoi la partie espagnole s'appelle-t-elle aussi Ruy Lopez ?",
                "partie-espagnole",
                "Elle est également appelée Ruy Lopez en hommage à Ruy López de Segura, moine "
                "espagnol du XVIe siècle",
            ),
        ],
        "Ruy López de Segura, moine espagnol du XVIe siècle (partie espagnole).",
    ),
    (
        "Quels joueurs soviétiques ont enrichi la théorie de la défense dont les codes ECO "
        "vont de E60 à E99 ?",
        [
            (
                "Quelle défense correspond aux codes ECO E60 à E99 ?",
                "defense-est-indienne",
                "la défense est-indienne appartient aux débuts semi-fermés (code ECO E60 à E99)",
            ),
            (
                "Quels joueurs soviétiques ont développé la défense est-indienne ?",
                "defense-est-indienne",
                "en particulier Issaak Boleslavski et David Bronstein",
            ),
        ],
        "Issaak Boleslavski et David Bronstein, puis Efim Geller (défense est-indienne).",
    ),
    (
        "Dans la variante de la Caro-Kann popularisée par Nigel Short, quel coup les Blancs "
        "jouent-ils le plus souvent au 5e coup ?",
        [
            (
                "Quelle variante de la Caro-Kann Nigel Short a-t-il popularisée ?",
                "defense-caro-kann",
                "popularisées par le grand maître anglais Nigel Short",
            ),
            (
                "Dans la variante Short de la Caro-Kann, que jouent les Blancs au 5e coup ?",
                "defense-caro-kann",
                "Les Blancs continuent le plus souvent par 5. Fe2 avec l'idée de jouer petit "
                "roque dans un futur proche.",
            ),
        ],
        "5.Fe2, pour préparer le petit roque (variante d'avance, ligne Short).",
    ),
    (
        "Dans le gambit qui porte le nom d'un capitaine de navire gallois, quelle est l'idée "
        "des Blancs après la retraite 5...Fe7 ?",
        [
            (
                "Quel gambit porte le nom d'un capitaine de navire gallois ?",
                "gambit-evans",
                "Le gambit porte le nom du capitaine de navire gallois William Davies Evans",
            ),
            (
                "Dans le gambit Evans, quelle est l'idée des Blancs après 5...Fe7 ?",
                "gambit-evans",
                "L'idée des blancs est de jouer Db3 et d'attaquer deux fois le pion faible en f7.",
            ),
        ],
        "Jouer Db3 pour attaquer deux fois le pion f7 (gambit Evans).",
    ),
]


def main() -> None:
    articles = {article.id: article for article in load_corpus(CORPUS_DIR)}
    questions: list[MultiHopQuestion] = []
    for index, (question, steps, answer) in enumerate(CHAINS, start=1):
        hops = []
        for position, (sub_question, article_id, excerpt) in enumerate(steps):
            if not contains_excerpt(articles[article_id].text, excerpt):
                raise SystemExit(f"Multi {index}, saut {position + 1} : extrait absent")
            hops.append(
                Hop(
                    sub_question=sub_question,
                    depends_on=position - 1 if position else None,
                    expected=ExpectedPassage(article_id=article_id, excerpt=excerpt),
                )
            )
        questions.append(
            MultiHopQuestion(
                id=f"multi-{index:02d}",
                question=question,
                hops=hops,
                answer=answer,
                provenance="redige+verifie",
            )
        )

    save_jsonl(questions, EVAL_DIR / "multi10.jsonl")
    cross = sum(len({h.expected.article_id for h in q.hops}) > 1 for q in questions)
    print(f"Multi-10 : {len(questions)} questions → {EVAL_DIR / 'multi10.jsonl'}")
    print(
        f"Chaînes entre deux articles : {cross} ; dans un même article : {len(questions) - cross}"
    )
    print("Sauts par question :", dict(Counter(len(q.hops) for q in questions)))


if __name__ == "__main__":
    main()
