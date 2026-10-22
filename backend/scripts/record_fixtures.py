"""Ré-enregistre les fixtures de test à partir des vraies API (« cassettes »).

À lancer à la main, quand le format d'une API change ou pour remplacer les fixtures
construites par des réponses réelles. Coût : 2 requêtes Lichess, 1 recherche YouTube
(100 unités de quota).

    LICHESS_TOKEN=... YOUTUBE_API_KEY=... uv run python -m scripts.record_fixtures
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
ITALIAN_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
RARE_FEN = "rnbqkbn1/pppppppp/7r/7p/P7/R7/1PPPPPPP/1NBQKBNR w Kq - 2 3"


def save(relative: str, payload: Any) -> None:
    path = FIXTURES / relative
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"enregistré : {path.relative_to(FIXTURES.parent.parent)}")


async def main() -> int:
    token = os.environ.get("LICHESS_TOKEN")
    key = os.environ.get("YOUTUBE_API_KEY")
    if not token or not key:
        print("LICHESS_TOKEN et YOUTUBE_API_KEY sont requis.", file=sys.stderr)
        return 1

    async with httpx.AsyncClient(timeout=10) as client:
        lichess = {"Authorization": f"Bearer {token}"}
        masters = await client.get(
            "https://explorer.lichess.ovh/masters",
            params={"fen": ITALIAN_FEN, "moves": 5, "topGames": 4},
            headers=lichess,
        )
        masters.raise_for_status()
        save("lichess/masters_italian.json", masters.json())

        rare = await client.get(
            "https://explorer.lichess.ovh/lichess",
            params={"fen": RARE_FEN, "moves": 5, "topGames": 0, "recentGames": 0},
            headers=lichess,
        )
        rare.raise_for_status()
        save("lichess/lichess_rare.json", rare.json())

        videos = await client.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "type": "video",
                "q": "partie italienne explication ouverture",
                "maxResults": 3,
                "relevanceLanguage": "fr",
                "safeSearch": "strict",
                "key": key,
            },
        )
        videos.raise_for_status()
        save("youtube/search_italian.json", videos.json())

    print("Relancer les tests : les valeurs attendues peuvent devoir être ajustées.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
