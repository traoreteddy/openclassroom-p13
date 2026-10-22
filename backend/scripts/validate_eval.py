"""Phase 1, étape 1.8 : vérifie la cohérence des 4 jeux de référence.

uv run python -m scripts.validate_eval
"""

from __future__ import annotations

import json

from app.rag.corpus import load_corpus
from app.rag.eval.datasets import EVAL_DIR
from app.rag.eval.validate import EvalSets, validate
from scripts.fetch_corpus import CORPUS_DIR

TOOL_NAMES = {
    "inspect_position",
    "get_opening_theory",
    "evaluate_position",
    "search_opening_knowledge",
    "find_opening_videos",
    "start_deep_analysis",
    "get_deep_analysis",
}


def main() -> None:
    report = validate(EvalSets.load(EVAL_DIR), load_corpus(CORPUS_DIR), TOOL_NAMES)
    print(json.dumps(report.stats, ensure_ascii=False, indent=2))
    if report.ok:
        print("✓ Les 4 jeux de référence sont cohérents.")
    else:
        print("\n".join(f"✗ {error}" for error in report.errors))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
