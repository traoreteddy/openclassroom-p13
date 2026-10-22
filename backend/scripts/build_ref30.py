"""Phase 1, étape 1.4 : construit Réf-30 à partir des candidats relus.

Chaque candidat généré par phi4-mini (data/eval/ref30_candidates.jsonl) reçoit un
verdict de relecture, tracé ici :

- « validée »  : question, extrait et réponse gardés (au plus une retouche de forme) ;
- « corrigée » : question reformulée, extrait ajusté ou réponse rectifiée ;
- « récupérée » : candidat rejeté par le contrôle automatique, repris avec le bon extrait ;
- « rejetée »  : écarté, avec le motif.

Chaque extrait est revérifié mot pour mot dans l'article complet (après normalisation).

    uv run python -m scripts.build_ref30
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Literal, NamedTuple

from app.rag.corpus import ARTICLES, load_corpus
from app.rag.eval.datasets import EVAL_DIR, ExpectedPassage, RefQuestion, save_jsonl
from app.rag.text import contains_excerpt
from scripts.fetch_corpus import CORPUS_DIR

Verdict = Literal["validée", "corrigée", "récupérée", "rejetée"]


class Review(NamedTuple):
    candidate: int  # numéro du candidat (1 = première ligne du fichier)
    verdict: Verdict
    question: str = ""
    excerpt: str = ""
    answer: str = ""
    reason: str = ""  # ce qui a été corrigé, ou motif du rejet


REVIEWS: list[Review] = [
    Review(1, "rejetée", reason="liste de coups : pas de question exploitable"),
    Review(
        2,
        "corrigée",
        "Dans la défense Caro-Kann, à quelle attaque mène la variante d'échange quand les "
        "Blancs poursuivent par c2-c4 ?",
        "La « variante d'échange » est très souvent suivie de 4. c2-c4 menant à l'attaque Panov",
        "À l'attaque Panov (ou Panov-Botvinnik), qui donne un jeu plus dégagé.",
        "question circulaire reformulée",
    ),
    Review(
        3,
        "corrigée",
        "Dans la variante Short de la défense Caro-Kann, quel coup les Blancs jouent-ils le plus "
        "souvent au 5e coup, et dans quel but ?",
        "Les Blancs continuent le plus souvent par 5. Fe2 avec l'idée de jouer petit roque dans "
        "un futur proche.",
        "5.Fe2, pour préparer le petit roque.",
        "coup noir mal noté dans la question (4. e6 au lieu de 4...e6)",
    ),
    Review(
        4,
        "validée",
        "Dans la défense des deux cavaliers, quel coup typique de l'école berlinoise a été "
        "analysé en profondeur au XIXe siècle par Paul Rudolf von Bilguer ?",
        "le coup typique de l'école berlinoise 3...Cf6 fut surtout analysé en profondeur au XIXe "
        "siècle par Paul Rudolf von Bilguer",
        "3...Cf6.",
        "extrait raccourci à la phrase utile",
    ),
    Review(
        5,
        "corrigée",
        "Contre la défense des deux cavaliers, pourquoi le coup 4.Cg5 est-il inhabituel dans "
        "une ouverture ?",
        "les Blancs déplacent la même pièce une deuxième fois dans les tout premiers coups de "
        "l'ouverture",
        "Les Blancs déplacent la même pièce une deuxième fois dès les premiers coups.",
        "la question donnait la réponse (« déplacement du cavalier en g5 »)",
    ),
    Review(
        6,
        "corrigée",
        "Dans la défense des deux cavaliers, pourquoi le coup 4.c3 est-il une erreur pour les "
        "Blancs ?",
        "Le coup naturel 4.c3? est immédiatement puni par les Noirs qui prennent le pion e4 avec "
        "leur cavalier",
        "Les Noirs prennent aussitôt le pion e4 avec leur cavalier.",
        "la question ne correspondait pas à l'extrait",
    ),
    Review(
        7,
        "corrigée",
        "Quels joueurs soviétiques ont fait de la défense est-indienne une arme redoutable après "
        "la Seconde Guerre mondiale ?",
        "elle devint l'arme redoutable des joueurs soviétiques de l'époque (en particulier "
        "Issaak Boleslavski et David Bronstein",
        "Issaak Boleslavski et David Bronstein, puis Efim Geller.",
        "l'extrait ne contenait pas la réponse : bonne phrase reprise du passage",
    ),
    Review(
        8,
        "validée",
        "Quelle ouverture commence par 1.d4 Cf6 2.c4 g6 3.Cc3 Fg7 ?",
        "Considérée comme très dynamique, la défense est-indienne appartient aux débuts "
        "semi-fermés (code ECO E60 à E99)",
        "La défense est-indienne (codes ECO E60 à E99).",
    ),
    Review(
        9,
        "corrigée",
        "Dans la défense est-indienne, comment s'appelle la variante où les Blancs jouent 5.f3 ?",
        "5. f3 (variante Sämisch) - Code ECO E80-E89",
        "La variante Sämisch.",
        "la question donnait la réponse",
    ),
    Review(
        10,
        "corrigée",
        "Dans la défense française, quel est le nom de la variante 2.Fb5 et quel coup "
        "empêche-t-elle ?",
        "2.Fb5, la variante Bird, qui empêche le coup 2.d5 à cause du clouage du pion d.",
        "La variante Bird ; elle empêche ...d5 grâce au clouage du pion d.",
        "question reformulée pour ne pas donner la réponse",
    ),
    Review(
        11,
        "validée",
        "Quelle variante de la défense française est jouée après 3...dxe4 ?",
        "3...dxe4 (variante Rubinstein)",
        "La variante Rubinstein.",
    ),
    Review(
        12,
        "corrigée",
        "Dans la défense française, comment s'appelle la variante avec 5...Fa5 6.b4 ?",
        "5...Fa5 6.b4, variante Arménienne, peut être aussi joué.",
        "La variante arménienne.",
        "détail inventé retiré (« sacrifie un pion pour ouvrir la colonne b »)",
    ),
    Review(13, "rejetée", reason="liste de coups : pas de question exploitable"),
    Review(
        14,
        "corrigée",
        "Dans la défense scandinave, la retraite de la dame 3...Dd8 est-elle encore jouée "
        "aujourd'hui ?",
        "Dd8 se jouait beaucoup aux débuts de la Scandinave, mais est devenu rare dans la "
        "pratique moderne, même si cette variante reste solide pour les noirs",
        "Elle est devenue rare, mais reste solide : les Noirs égalisent si les Blancs jouent "
        "passivement.",
        "question trop longue, qui contenait la réponse",
    ),
    Review(
        15,
        "corrigée",
        "Contre la défense scandinave, que se passe-t-il quand les Blancs avancent leur pion "
        "par 2.e5 ?",
        "les noirs vont sortir rapidement leur Fou-Dame (Ff5 ou Fg4)",
        "Les Blancs relâchent la tension centrale et les Noirs sortent vite leur Fou-Dame "
        "(Ff5 ou Fg4).",
        "question reformulée",
    ),
    Review(
        16,
        "corrigée",
        "Pourquoi la défense sicilienne est-elle si populaire auprès des joueurs des Noirs ?",
        "elle offre aux noirs de réelles chances de gain sans devoir se contenter de l'égalisation",
        "Elle offre aux Noirs de réelles chances de gain, sans se contenter d'égaliser.",
        "l'extrait ne contenait pas la réponse : bonne phrase reprise du passage",
    ),
    Review(
        17,
        "récupérée",
        "À quel compositeur Eric Schiller compare-t-il la variante du dragon de la défense "
        "sicilienne ?",
        "Eric Schiller compare la variante Najdorf à Piotr Ilitch Tchaïkovski, la variante du "
        "dragon à Richard Wagner",
        "À Richard Wagner.",
        "extrait paraphrasé par phi4-mini : phrase exacte reprise",
    ),
    Review(18, "rejetée", reason="liste de codes ECO : pas de question exploitable"),
    Review(
        19,
        "corrigée",
        "Dans le gambit slave, quel coup a récemment regagné en popularité au plus haut niveau ?",
        "La variante a toutefois connu un gain de popularité récemment au top niveau avec le "
        "coup 6.Fe2, visant des compensations à long terme.",
        "6.Fe2, pour obtenir des compensations à long terme.",
        "l'extrait ne contenait pas la réponse : bonne phrase reprise du passage",
    ),
    Review(
        20,
        "corrigée",
        "Par quels coups obtient-on la variante Alapine de la défense slave ?",
        "Elle s'obtient après les coups 5. a4 Ff5.",
        "Après 5.a4 Ff5.",
        "détail inventé retiré (« empêchant 6.e4 »)",
    ),
    Review(
        21,
        "validée",
        "Quelle variante de la défense slave, surnommée Slave Caméléon, est atteinte après "
        "3.Cf3 Cf6 4.Cc3 a6 ?",
        "Également appelée Slave Caméléon, elle s'obtient après les coups 3.Cf3 Cf6 4.Cc3 a6.",
        "La variante Tchebanenko.",
    ),
    Review(
        22,
        "corrigée",
        "Quelle part de l'Encyclopédie des ouvertures d'échecs le gambit dame occupe-t-il ?",
        "occupe à lui seul pratiquement la totalité du volume D de l'Encyclopédie des ouvertures "
        "d'échecs",
        "Presque tout le volume D.",
        "question reformulée",
    ),
    Review(
        23,
        "corrigée",
        "Depuis quand le gambit dame est-il connu ?",
        "Le gambit dame était déjà connu des maîtres italiens du XVe siècle",
        "Depuis les maîtres italiens du XVe siècle ; il figure dans le manuscrit de Göttingen.",
        "la question (Stamma, Karpov) ne correspondait pas à l'extrait",
    ),
    Review(24, "rejetée", reason="hors périmètre d'un coach d'ouvertures (série Netflix)"),
    Review(
        25,
        "validée",
        "D'où vient le nom du gambit Evans ?",
        "Le gambit porte le nom du capitaine de navire gallois William Davies Evans, le premier "
        "joueur à l'avoir pratiqué.",
        "Du capitaine gallois William Davies Evans, le premier à l'avoir joué.",
        "question rendue plus naturelle",
    ),
    Review(
        26,
        "corrigée",
        "Dans le gambit Evans, dans quelle situation le coup Ca5 devient-il important ?",
        "le coup Ca5 est crucial dans certaines variantes, particulièrement si les Blancs ont un "
        "fou en c4 et une dame en b3",
        "Quand les Blancs ont un fou en c4 et une dame en b3.",
        "réponse fausse (Fa5 au lieu de Ca5)",
    ),
    Review(
        27,
        "corrigée",
        "Dans le gambit Evans, après la retraite 5...Fe7, quelle est l'idée des Blancs ?",
        "L'idée des blancs est de jouer Db3 et d'attaquer deux fois le pion faible en f7.",
        "Jouer Db3 pour attaquer deux fois le pion f7.",
        "question floue et réponse fausse (6.d4)",
    ),
    Review(
        28,
        "corrigée",
        "Par quel coup commence l'ouverture anglaise ?",
        "elle débute par le coup 1. c4",
        "1.c4 (codes ECO A10 à A39).",
        "la question (Grünfeld, nimzo-indienne) ne correspondait pas à l'extrait",
    ),
    Review(
        29,
        "corrigée",
        "Quelle est l'intention principale des Blancs avec le coup 1.c4 dans l'ouverture "
        "anglaise ?",
        "Par le coup 1. c4, les blancs montrent leur intention de contrôler la case d5.",
        "Contrôler la case d5.",
        "réponse fausse (« contre le centre des noirs »)",
    ),
    Review(
        30,
        "corrigée",
        "Dans l'ouverture anglaise, à quoi ressemble la ligne 1.c4 e5 2.Cc3 ?",
        "La partie se déroule comme une Sicilienne fermée avec les couleurs inversées.",
        "À une sicilienne fermée jouée avec les couleurs inversées.",
        "question reformulée",
    ),
    Review(
        31,
        "récupérée",
        "Dans la partie espagnole, comment s'appelle la réponse rare 3...g5 ?",
        "3...g5 → défense ou gambit Brentano",
        "La défense (ou gambit) Brentano.",
        "extrait sans l'espace avant « ; » : phrase exacte reprise",
    ),
    Review(
        32,
        "validée",
        "Selon Reuben Fine, quel est le plus fort début ouvert pour les Blancs ?",
        "Reuben Fine avance que la partie espagnole est le plus fort début ouvert à la "
        "disposition des Blancs.",
        "La partie espagnole.",
        "question raccourcie",
    ),
    Review(
        33,
        "corrigée",
        "Quelle variante de la partie espagnole Anatoli Karpov a-t-il jouée sept fois contre "
        "Garry Kasparov en championnat du monde ?",
        "Anatoli Karpov usa à sept reprises de la variante Zaïtsev",
        "La variante Zaïtsev.",
        "joueurs intervertis dans la question",
    ),
    Review(
        34,
        "corrigée",
        "La partie italienne est-elle une ouverture adaptée aux débutants ?",
        "Cette ouverture est très employée par les débutants.",
        "Oui, elle est très employée par les débutants.",
        "la question ne correspondait pas à l'extrait",
    ),
]


def main() -> None:
    lines = (EVAL_DIR / "ref30_candidates.jsonl").read_text(encoding="utf-8").splitlines()
    candidates = [json.loads(line) for line in lines]
    articles = {article.id: article for article in load_corpus(CORPUS_DIR)}
    families = {ref.slug: ref.opening_family for ref in ARTICLES}
    assert sorted(r.candidate for r in REVIEWS) == list(range(1, len(candidates) + 1))

    questions: list[RefQuestion] = []
    for review in REVIEWS:
        if review.verdict == "rejetée":
            continue
        article_id = candidates[review.candidate - 1]["article_id"]
        text = articles[article_id].text
        if not contains_excerpt(text, review.excerpt):
            raise SystemExit(f"Candidat {review.candidate} : extrait absent de {article_id}")
        questions.append(
            RefQuestion(
                id=f"ref-{len(questions) + 1:02d}",
                question=review.question,
                opening_family=families[article_id],
                expected=[ExpectedPassage(article_id=article_id, excerpt=review.excerpt)],
                answer=review.answer,
                provenance="phi4-mini+relecture",
            )
        )

    save_jsonl(questions, EVAL_DIR / "ref30.jsonl")
    log = EVAL_DIR / "ref30_review.jsonl"
    log.write_text(
        "\n".join(json.dumps(r._asdict(), ensure_ascii=False) for r in REVIEWS) + "\n",
        encoding="utf-8",
    )

    verdicts = Counter(r.verdict for r in REVIEWS)
    per_article = Counter(q.expected[0].article_id for q in questions)
    print(f"Réf-30 : {len(questions)} questions → {EVAL_DIR / 'ref30.jsonl'}")
    print("Verdicts :", dict(verdicts))
    print("Par article :", dict(sorted(per_article.items())))


if __name__ == "__main__":
    main()
