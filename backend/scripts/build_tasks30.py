"""Phase 1, étape 1.6 : construit Tâches-30 (quel outil l'agent doit-il appeler ?).

Les 30 demandes couvrent les 7 outils du catalogue et le cas « aucun outil ».
Les FEN sont calculées par python-chess à partir de suites de coups (pas de saisie
manuelle). `expected_arguments` liste seulement les arguments imposés par la demande.

    uv run python -m scripts.build_tasks30
"""

from __future__ import annotations

from collections import Counter

import chess

from app.rag.eval.datasets import EVAL_DIR, ToolTask, save_jsonl


def fen(moves: str) -> str:
    """FEN obtenue en jouant `moves` (notation anglaise SAN) depuis la position initiale."""
    board = chess.Board()
    for san in moves.split():
        board.push_san(san)
    return board.fen()


START = chess.STARTING_FEN
AFTER_E4 = fen("e4")
ITALIAN = fen("e4 e5 Nf3 Nc6 Bc4 Bc5")
SICILIAN = fen("e4 c5")
RUY_LOPEZ = fen("e4 e5 Nf3 Nc6 Bb5")
QUEENS_GAMBIT = fen("d4 d5 c4")
BONGCLOUD = fen("e4 e5 Ke2")
FOOLS_MATE = fen("f3 e5 g4 Qh4#")
SCHOLAR_TRY = fen("e4 e5 Qh5 Nc6 Bc4 Nf6")
FRENCH = fen("e4 e6 d4 d5")
NO_QUEEN = fen("e4 e5 Qh5 Nc6 Qxe5+ Nxe5")
CARO_KANN = fen("e4 c6 d4 d5 Nc3 dxe4 Nxe4")
KINGS_INDIAN = fen("d4 Nf6 c4 g6 Nc3 Bg7 e4 d6")

# (demande, outil attendu ou None, arguments imposés, note)
TASKS: list[tuple[str, str | None, dict[str, str | int | bool], str | None]] = [
    # inspect_position : identifier la position, coups légaux, mat
    (
        f"Quels coups sont légaux pour les Noirs dans cette position : {AFTER_E4} ?",
        "inspect_position",
        {"fen": AFTER_E4},
        None,
    ),
    (
        f"Les Blancs peuvent-ils roquer dans cette position : {ITALIAN} ?",
        "inspect_position",
        {"fen": ITALIAN},
        None,
    ),
    (
        f"Est-ce que c'est échec et mat ici : {FOOLS_MATE} ?",
        "inspect_position",
        {"fen": FOOLS_MATE},
        None,
    ),
    (f"Vérifie que cette FEN est correcte : {FRENCH}", "inspect_position", {"fen": FRENCH}, None),
    # get_opening_theory : coups théoriques, nom d'ouverture, parties de référence
    (
        f"Que jouent les maîtres dans cette position : {SICILIAN} ?",
        "get_opening_theory",
        {"fen": SICILIAN},
        None,
    ),
    (
        f"Comment s'appelle l'ouverture jouée dans cette position : {RUY_LOPEZ} ?",
        "get_opening_theory",
        {"fen": RUY_LOPEZ},
        None,
    ),
    (
        f"Montre-moi des parties de maîtres jouées dans cette position : {ITALIAN}",
        "get_opening_theory",
        {"fen": ITALIAN, "include_games": True},
        None,
    ),
    (
        f"Que jouent les joueurs en ligne, pas les maîtres, ici : {QUEENS_GAMBIT} ?",
        "get_opening_theory",
        {"fen": QUEENS_GAMBIT, "database": "lichess"},
        None,
    ),
    (
        f"Cette position est-elle encore dans la théorie : {BONGCLOUD} ?",
        "get_opening_theory",
        {"fen": BONGCLOUD},
        "in_theory indique si on est sorti de la théorie",
    ),
    # evaluate_position : qui est mieux, meilleurs coups
    (
        f"Qui est mieux dans cette position : {SCHOLAR_TRY} ?",
        "evaluate_position",
        {"fen": SCHOLAR_TRY},
        None,
    ),
    (
        f"Donne-moi les 3 meilleurs coups dans cette position : {CARO_KANN}",
        "evaluate_position",
        {"fen": CARO_KANN, "lines": 3},
        None,
    ),
    (
        f"Évalue rapidement cette position : {KINGS_INDIAN}",
        "evaluate_position",
        {"fen": KINGS_INDIAN},
        None,
    ),
    (f"Ma position est-elle perdue ? {NO_QUEEN}", "evaluate_position", {"fen": NO_QUEEN}, None),
    (
        f"Quel est le meilleur coup pour les Blancs ici : {FRENCH} ?",
        "evaluate_position",
        {"fen": FRENCH},
        None,
    ),
    # search_opening_knowledge : idées, plans, histoire
    (
        "Pourquoi les Noirs jouent-ils c5 dans la défense sicilienne ?",
        "search_opening_knowledge",
        {},
        None,
    ),
    ("Quelle est l'histoire du gambit Evans ?", "search_opening_knowledge", {}, None),
    (
        "Quels sont les plans typiques de la défense est-indienne ?",
        "search_opening_knowledge",
        {},
        None,
    ),
    ("Quels grands joueurs ont joué la défense Caro-Kann ?", "search_opening_knowledge", {}, None),
    (
        "Explique-moi l'idée de l'attaque Panov dans la Caro-Kann.",
        "search_opening_knowledge",
        {},
        None,
    ),
    (
        "Pourquoi dit-on que la partie espagnole est une ouverture forte ?",
        "search_opening_knowledge",
        {},
        None,
    ),
    # find_opening_videos : explications en vidéo
    ("Trouve-moi une vidéo pour apprendre la défense française.", "find_opening_videos", {}, None),
    (
        "Je préfère apprendre en vidéo : tu as une explication courte du gambit dame ?",
        "find_opening_videos",
        {},
        None,
    ),
    (
        "As-tu une vidéo sur la partie italienne pour les débutants ?",
        "find_opening_videos",
        {},
        None,
    ),
    ("Montre-moi des vidéos sur l'ouverture anglaise.", "find_opening_videos", {}, None),
    # start_deep_analysis / get_deep_analysis : analyse longue, différée
    (
        f"Analyse cette position en profondeur, prends ton temps : {CARO_KANN}",
        "start_deep_analysis",
        {"fen": CARO_KANN},
        None,
    ),
    (
        f"Lance une analyse très profonde, à la profondeur 35, de cette position : {SICILIAN}",
        "start_deep_analysis",
        {"fen": SICILIAN, "depth": 35},
        None,
    ),
    ("Où en est l'analyse a1b2c3d4 ?", "get_deep_analysis", {"job_id": "a1b2c3d4"}, None),
    # aucun outil
    ("Merci, c'est très clair !", None, {}, "simple échange"),
    ("Quelle est la capitale du Japon ?", None, {}, "hors sujet"),
    ("Combien de cases y a-t-il sur un échiquier ?", None, {}, "culture générale : 64"),
]


def main() -> None:
    tasks = [
        ToolTask(
            id=f"task-{index:02d}",
            request=request,
            expected_tool=tool,
            expected_arguments=arguments,
            note=note,
        )
        for index, (request, tool, arguments, note) in enumerate(TASKS, start=1)
    ]
    for task in tasks:
        if "fen" in task.expected_arguments:
            assert chess.Board(str(task.expected_arguments["fen"])).is_valid(), task.id
    save_jsonl(tasks, EVAL_DIR / "tasks30.jsonl")
    print(f"Tâches-30 : {len(tasks)} demandes →", EVAL_DIR / "tasks30.jsonl")
    print(dict(Counter(task.expected_tool or "aucun outil" for task in tasks)))


if __name__ == "__main__":
    main()
