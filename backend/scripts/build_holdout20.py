"""Phase 4, correction C5 : jeu de validation tenu à l'écart (Holdout-20).

Rédigé APRÈS le gel de la configuration et jamais utilisé pour un choix : il sert
seulement à vérifier que les gains mesurés sur les 50 requêtes de mise au point se
confirment sur des questions nouvelles (réponse à la critique de sur-ajustement).
Questions rédigées à la main ; extraits vérifiés mot pour mot.

    uv run python -m scripts.build_holdout20
"""

from __future__ import annotations

from collections import Counter

from app.rag.corpus import ARTICLES, load_corpus
from app.rag.eval.datasets import EVAL_DIR, ExpectedPassage, RefQuestion, save_jsonl
from app.rag.text import contains_excerpt
from scripts.fetch_corpus import CORPUS_DIR

# (question, article, extrait, réponse, catégorie)
QUESTIONS: list[tuple[str, str, str, str, str]] = [
    (
        "Que préparent les Noirs avec le coup 1...c6 dans la Caro-Kann ?",
        "defense-caro-kann",
        "Par 1...c6, les Noirs préparent la poussée centrale 2...d5.",
        "La poussée 2...d5.",
        "idée",
    ),
    (
        "Qu'est-ce que le gambit Alien dans la défense Caro-Kann ?",
        "defense-caro-kann",
        "1.e4 c6 2.d4 d5 3.Cc3 dxe4 4.Cxe4 Cf6 5.Ceg5 h6 6.Cxf7 est le gambit Alien.",
        "La suite 5.Ceg5 h6 6.Cxf7 après 4...Cf6.",
        "coups",
    ),
    (
        "De quel gambit se rapproche la variante Tartakover de la Caro-Kann ?",
        "defense-caro-kann",
        "la variante Tartakover se rapproche du gambit Blackmar-Diemer",
        "Du gambit Blackmar-Diemer.",
        "idée",
    ),
    (
        "Quelle ligne de la Caro-Kann avec 2.Cf3 et 3.Cc3 Bobby Fischer a-t-il jouée au début de "
        "sa carrière ?",
        "defense-caro-kann",
        "1.e4 c6 2.Cf3 d5 3.Cc3, parfois jouée par Bobby Fischer au début de sa carrière.",
        "La variante des deux cavaliers.",
        "coups",
    ),
    (
        "Dans la défense des deux cavaliers, quels sont les inconvénients pour les Noirs après "
        "4.d4 exd4 ?",
        "defense-des-deux-cavaliers",
        "leurs cases e5 et f7 sont faibles, et le Cavalier en f6 peut être facilement cloué",
        "Cases e5 et f7 faibles, cavalier f6 facile à clouer.",
        "idée",
    ),
    (
        "À quelle famille d'ouvertures appartient l'est-indienne, et depuis quand la joue-t-on "
        "en tournoi ?",
        "defense-est-indienne",
        "L'est-indienne fait partie des ouvertures dites hypermodernes, qui ont commencé à "
        "apparaître régulièrement dans les tournois dans les années 1920.",
        "Les ouvertures hypermodernes, depuis les années 1920.",
        "histoire",
    ),
    (
        "Qu'est-ce que la défense franco-Benoni ?",
        "defense-francaise",
        "La principale alternative à 2...d5 est 2...c5, la défense franco-Benoni",
        "La réponse 2...c5 à 2.d4, hybride de la française et de la Benoni.",
        "coups",
    ),
    (
        "Par quels coups obtient-on la variante classique de la défense française ?",
        "defense-francaise",
        "La variante classique s'obtient après les coups 1.e4 e6 2.d4 d5 3.Cc3 Cf6",
        "1.e4 e6 2.d4 d5 3.Cc3 Cf6.",
        "coups",
    ),
    (
        "Quel autre nom porte la variante Winawer de la défense française ?",
        "defense-francaise",
        "variante Winawer aussi appelée défense Nimzowitsch",
        "Défense Nimzowitsch.",
        "histoire",
    ),
    (
        "Dans la défense scandinave, pourquoi 3.Cc3 est-il un bon coup après 2...Dxd5 ?",
        "defense-scandinave",
        "Après 2...Dxd5, les Blancs jouent généralement 3.Cc3, qui force les Noirs à déplacer à "
        "nouveau la dame, perdant ainsi un temps.",
        "Il chasse la dame, qui perd un temps.",
        "idée",
    ),
    (
        "D'où vient le nom de la défense sicilienne ?",
        "defense-sicilienne",
        "reçut apparemment son nom du joueur italien Gioachino Greco",
        "Du joueur italien Gioachino Greco.",
        "histoire",
    ),
    (
        "Quels coups mènent à l'attaque Rossolimo contre la défense sicilienne ?",
        "defense-sicilienne",
        "L'attaque Rossolimo (B30-B31) : 1.e4 c5 2.Cf3 Cc6 3.Fb5",
        "1.e4 c5 2.Cf3 Cc6 3.Fb5.",
        "coups",
    ),
    (
        "Contre qui Garry Kasparov a-t-il joué une partie de gambit Evans au Mémorial Tal "
        "en 1995 ?",
        "gambit-evans",
        "Garry Kasparov-Viswanathan Anand, Mémorial Tal, Riga (Lettonie), 1995",
        "Contre Viswanathan Anand.",
        "partie",
    ),
    (
        "Par quels coups obtient-on le gambit slave ?",
        "defense-slave",
        "Il s'obtient après les coups 3.Cf3 Cf6 4.Cc3 dxc4 5.e4 b5.",
        "3.Cf3 Cf6 4.Cc3 dxc4 5.e4 b5.",
        "coups",
    ),
    (
        "À quel champion du monde correspond le style du coup 5...Ca6 dans la variante Alapine "
        "de la slave ?",
        "defense-slave",
        "5...Ca6 correspond bien au style de l'ex-champion du monde Vasily Smyslov",
        "Vasily Smyslov.",
        "idée",
    ),
    (
        "Pourquoi Philidor appelait-il le gambit dame « gambit d'Alep » ?",
        "gambit-dame",
        "Philidor le nomme le gambit d'Alep, en référence à son plus fort adversaire, Philippe "
        "Stamma, originaire de cette ville.",
        "En référence à Philippe Stamma, originaire d'Alep.",
        "histoire",
    ),
    (
        "Quelle est la première partie connue avec le gambit Evans ?",
        "gambit-evans",
        "La première partie connue avec le gambit est Evans-McDonnell (Londres, 1827)",
        "Evans-McDonnell, Londres 1827.",
        "histoire",
    ),
    (
        "Dans le gambit Evans, quelle retraite du fou est la plus populaire et la plus "
        "fructueuse ?",
        "gambit-evans",
        "Selon les bases de données de parties d'échecs, c'est la plus populaire et la plus "
        "fructueuse retraite du fou.",
        "5...Fa5.",
        "idée",
    ),
    (
        "D'où vient le nom de l'ouverture anglaise ?",
        "ouverture-anglaise",
        "elle a été jouée par le champion anglais Howard Staunton contre le Français Pierre "
        "Saint-Amant lors de leur match de 1843",
        "Du match Staunton - Saint-Amant de 1843.",
        "histoire",
    ),
    (
        "Quel est le code ECO de la défense berlinoise proprement dite, après 4.0-0 Cxe4 dans la "
        "partie espagnole ?",
        "partie-espagnole",
        "4.0-0 Cxe4 → C67 : défense berlinoise proprement dite",
        "C67.",
        "coups",
    ),
]


def main() -> None:
    articles = {article.id: article for article in load_corpus(CORPUS_DIR)}
    families = {ref.slug: ref.opening_family for ref in ARTICLES}
    questions = []
    for index, (question, article_id, excerpt, answer, _) in enumerate(QUESTIONS, start=1):
        if not contains_excerpt(articles[article_id].text, excerpt):
            raise SystemExit(f"Holdout {index} : extrait absent de {article_id}")
        questions.append(
            RefQuestion(
                id=f"holdout-{index:02d}",
                question=question,
                opening_family=families[article_id],
                expected=[ExpectedPassage(article_id=article_id, excerpt=excerpt)],
                answer=answer,
                provenance="redige+verifie",
            )
        )
    save_jsonl(questions, EVAL_DIR / "holdout20.jsonl")
    print(f"Holdout-20 : {len(questions)} questions → {EVAL_DIR / 'holdout20.jsonl'}")
    print("Catégories :", dict(Counter(q[4] for q in QUESTIONS)))
    print("Articles couverts :", len({q[1] for q in QUESTIONS}))


if __name__ == "__main__":
    main()
