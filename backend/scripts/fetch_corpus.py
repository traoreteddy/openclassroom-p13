"""Télécharge le corpus Wikipédia FR et le fige dans backend/data/corpus/.

À lancer une seule fois (ou pour rafraîchir le corpus) ; l'ingestion lit ensuite les
fichiers figés, ce qui rend les mesures reproductibles.

    uv run python -m scripts.fetch_corpus
    WIKIPEDIA_CONTACT="https://mon-site.example" uv run python -m scripts.fetch_corpus
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx

from app.rag.corpus import ARTICLES, DEFAULT_CONTACT, WikipediaClient, save_article

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"


async def main() -> None:
    async with httpx.AsyncClient(timeout=30) as http:
        # Contact exigé par Wikimedia dans le User-Agent (URL ou e-mail)
        contact = os.environ.get("WIKIPEDIA_CONTACT", DEFAULT_CONTACT)
        wikipedia = WikipediaClient(http, contact=contact)
        for ref in ARTICLES:
            article = await wikipedia.fetch(ref)
            path = save_article(article, CORPUS_DIR)
            print(
                f"{article.title:<28} rev {article.metadata.version}  "
                f"{len(article.text):>6} car. texte  {len(article.html):>7} car. HTML"
                f"  → {path.name}"
            )


if __name__ == "__main__":
    asyncio.run(main())
