"""Format des jeux de référence de l'évaluation RAG (phase 1).

Quatre jeux, stockés en JSON Lines dans `backend/data/eval/` :

| Fichier          | Contenu                                 | Mesure                      |
|------------------|-----------------------------------------|-----------------------------|
| ref30.jsonl      | 30 questions + extraits attendus        | recall@5, rappel, précision |
| decision50.jsonl | 50 questions, dont 20 hors corpus       | taux d'appels inutiles      |
| tasks30.jsonl    | 30 demandes + outil attendu             | taux d'appels corrects      |
| multi10.jsonl    | 10 questions multi-sauts + sous-questions | multi-sauts vs un passage |

Les passages attendus sont des **extraits de texte** copiés mot pour mot de l'article,
pas des identifiants de chunks : le même jeu reste valable pour tous les découpages
(un chunk est pertinent s'il contient l'extrait, voir `app.rag.text.contains_excerpt`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Provenance = Literal[
    "phi4-mini+relecture",  # généré par phi4-mini, relu et éventuellement corrigé
    "redige+verifie",  # rédigé à la main, extraits vérifiés automatiquement
]


class EvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExpectedPassage(EvalItem):
    article_id: str  # identifiant du corpus, ex. « defense-sicilienne »
    excerpt: str = Field(min_length=20)  # phrase copiée mot pour mot de l'article


class RefQuestion(EvalItem):
    """Réf-30 : une question et le ou les passages qui contiennent la réponse."""

    id: str
    question: str
    opening_family: str | None  # filtre attendu, nom Lichess (ex. « Sicilian Defense »)
    expected: list[ExpectedPassage] = Field(min_length=1)
    answer: str  # réponse de référence, courte
    provenance: Provenance


DecisionCategory = Literal[
    "dans_corpus",  # la recherche est utile
    "echecs_hors_corpus",  # sujet d'échecs absent du corpus : la recherche ne trouvera rien
    "autre_outil",  # un autre outil répond (évaluation, théorie, légalité d'un coup)
    "hors_sujet",  # sans rapport avec les échecs, ou simple échange
]


class DecisionQuestion(EvalItem):
    """Décision-50 : faut-il chercher dans la base de connaissances ?"""

    id: str
    question: str
    category: DecisionCategory
    search_useful: bool
    note: str | None = None


class ToolTask(EvalItem):
    """Tâches-30 : une demande et l'outil que l'agent doit appeler (ou aucun)."""

    id: str
    request: str
    expected_tool: str | None
    # Arguments qui doivent figurer dans l'appel (les autres sont libres)
    expected_arguments: dict[str, str | int | bool] = Field(default_factory=dict)
    note: str | None = None


class Hop(EvalItem):
    sub_question: str
    depends_on: int | None  # indice du saut précédent dont la réponse est utilisée
    expected: ExpectedPassage


class MultiHopQuestion(EvalItem):
    """Multi-10 : question à plusieurs sauts, décomposée en sous-questions dépendantes."""

    id: str
    question: str
    hops: list[Hop] = Field(min_length=2)
    answer: str
    provenance: Provenance


EVAL_DIR = Path(__file__).resolve().parents[3] / "data" / "eval"


def save_jsonl[T: EvalItem](items: list[T], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [item.model_dump_json(exclude_none=False) for item in items]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_jsonl[T: EvalItem](model: type[T], path: Path) -> list[T]:
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
