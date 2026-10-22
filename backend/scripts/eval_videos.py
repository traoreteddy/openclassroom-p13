"""Mesure de pertinence de `find_opening_videos` sur les 10 familles d'ouvertures du corpus.

Pour chaque ouverture : requêtes construites, candidats ramenés, vidéos retenues après le
filtre qualité, part des titres qui nomment l'ouverture (FR ou EN), langue et durée.
Coût : 201 unités de quota par ouverture non cachée (2 recherches + 1 lecture de détails),
soit 2 010 unités pour une mesure complète.

    uv run python -m scripts.eval_videos            # cache MongoDB si disponible
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from app.core.config import settings
from app.rag.eval.datasets import EVAL_DIR
from app.tools.catalog import build_registry, default_dependencies
from app.tools.videos import OPENING_NAMES_FR

OPENINGS = [
    "Italian Game",
    "Ruy Lopez",
    "Sicilian Defense",
    "French Defense",
    "Caro-Kann Defense",
    "Scandinavian Defense",
    "Queen's Gambit",
    "Slav Defense",
    "King's Indian Defense",
    "English Opening",
]
RESULTS = EVAL_DIR / "results" / "videos-10.json"


def name_terms(opening: str) -> set[str]:
    """Mots distinctifs du nom (FR et EN), 4 lettres et plus, sans les mots génériques."""
    generic = {"game", "defense", "opening", "gambit", "partie", "défense", "ouverture"}
    words = re.findall(r"[^\W\d_]{4,}", f"{opening} {OPENING_NAMES_FR.get(opening, '')}".lower())
    return {w for w in words if w not in generic}


async def main() -> dict:
    deps = default_dependencies(settings)
    registry = build_registry(settings, deps)
    rows = []
    started = time.perf_counter()
    try:
        for opening in OPENINGS:
            result = await registry.execute(
                "find_opening_videos", {"opening": opening, "max_results": 3}
            )
            if not result.ok:
                rows.append({"opening": opening, "error": result.error})
                print(f"  ✗ {opening}: {result.error['code']} {result.error['message']}")
                continue
            data = result.data or {}
            videos = data["videos"]
            terms = name_terms(opening)
            named = [any(t in v["title"].lower() for t in terms) for v in videos]
            rows.append(
                {
                    "opening": opening,
                    "queries": data["queries"],
                    "from_cache": data["from_cache"],
                    "retained": len(videos),
                    "named": sum(named),
                    "languages": [v["language"] for v in videos],
                    "durations_min": [round((v["duration_s"] or 0) / 60) for v in videos],
                    "views": [v["views"] for v in videos],
                    "titles": [v["title"] for v in videos],
                    "quota_used_today": data["quota_used_today"],
                }
            )
            mark = "✓" if videos and all(named) else "~" if videos else "✗"
            print(f"  {mark} {opening}: {len(videos)} vidéo(s), {sum(named)} nomment l'ouverture")
            for v in videos:
                print(f"      · [{v['language']}] {v['title'][:70]} ({v['views']} vues)")
    finally:
        await deps.aclose()

    scored = [r for r in rows if "error" not in r]
    retained = sum(r["retained"] for r in scored)
    named = sum(r["named"] for r in scored)
    french = sum(r["languages"].count("fr") for r in scored)
    summary = {
        "openings": len(OPENINGS),
        "with_videos": sum(1 for r in scored if r["retained"]),
        "retained_total": retained,
        "named_share": round(named / retained, 2) if retained else 0,
        "french_share": round(french / retained, 2) if retained else 0,
        "quota_used_today": scored[-1]["quota_used_today"] if scored else None,
        "seconds": round(time.perf_counter() - started, 1),
        "rows": rows,
    }
    print(
        f"\n{summary['with_videos']}/{summary['openings']} ouvertures avec vidéos ; "
        f"{retained} vidéos retenues, {summary['named_share']:.0%} nomment l'ouverture, "
        f"{summary['french_share']:.0%} en français ; quota du jour : {summary['quota_used_today']}"
    )
    return summary


def write(summary: dict) -> None:
    Path(RESULTS).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"→ {RESULTS}")


if __name__ == "__main__":
    write(asyncio.run(main()))
