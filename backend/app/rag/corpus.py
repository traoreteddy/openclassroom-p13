"""Corpus du RAG : articles Wikipédia FR sur les ouvertures, figés avec leurs métadonnées.

Chaque article est récupéré en trois formats, un par famille de découpage :
- `text`     : texte brut, titres au format wiki (« == Variantes == ») ;
- `markdown` : même texte, titres convertis en Markdown (« ## Variantes ») ;
- `html`     : HTML rendu de l'article (découpage par sections HTML).
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict

from app.tools.http import get_json
from app.tools.resilience import ResiliencePolicy, call_with_resilience

WIKIPEDIA_API = "https://fr.wikipedia.org/w/api.php"
SERVICE = "Wikipédia FR"
LICENCE = "CC BY-SA 4.0"
# Politique Wikimedia (https://w.wiki/4wJS, T400119) : le User-Agent doit identifier le
# client ET donner un moyen de contact (URL ou e-mail), sinon l'API répond 403.
DEFAULT_CONTACT = "https://github.com/teddfullstack"
PAUSE_S = 1.0
FETCH_POLICY = ResiliencePolicy(timeout_s=20, max_attempts=4, backoff_initial_s=2)

Confidentiality = Literal["public", "interne", "confidentiel"]
DocumentType = Literal["article", "procedure", "code", "tableau", "courriel"]


class ArticleRef(BaseModel):
    """Article à récupérer, avec le nom d'ouverture utilisé par Lichess (filtre commun)."""

    model_config = ConfigDict(frozen=True)

    title: str
    opening: str  # nom Lichess, ex. « Italian Game: Two Knights Defense »
    opening_family: str  # famille Lichess, ex. « Italian Game »

    @property
    def slug(self) -> str:
        return slugify(self.title)


ARTICLES: tuple[ArticleRef, ...] = (
    ArticleRef(title="Partie italienne", opening="Italian Game", opening_family="Italian Game"),
    ArticleRef(
        title="Défense des deux cavaliers",
        opening="Italian Game: Two Knights Defense",
        opening_family="Italian Game",
    ),
    ArticleRef(
        title="Gambit Evans", opening="Italian Game: Evans Gambit", opening_family="Italian Game"
    ),
    ArticleRef(title="Partie espagnole", opening="Ruy Lopez", opening_family="Ruy Lopez"),
    ArticleRef(
        title="Défense sicilienne", opening="Sicilian Defense", opening_family="Sicilian Defense"
    ),
    ArticleRef(
        title="Défense française", opening="French Defense", opening_family="French Defense"
    ),
    ArticleRef(
        title="Défense Caro-Kann", opening="Caro-Kann Defense", opening_family="Caro-Kann Defense"
    ),
    ArticleRef(title="Gambit dame", opening="Queen's Gambit", opening_family="Queen's Gambit"),
    ArticleRef(
        title="Défense slave",
        opening="Queen's Gambit Declined: Slav Defense",
        opening_family="Slav Defense",
    ),
    ArticleRef(
        title="Défense est-indienne",
        opening="King's Indian Defense",
        opening_family="King's Indian Defense",
    ),
    ArticleRef(
        title="Ouverture anglaise", opening="English Opening", opening_family="English Opening"
    ),
    ArticleRef(
        title="Défense scandinave",
        opening="Scandinavian Defense",
        opening_family="Scandinavian Defense",
    ),
)


class DocumentMetadata(BaseModel):
    """Métadonnées filtrables, communes à tous les documents du corpus."""

    date: str  # date de la révision (ISO 8601)
    version: int  # numéro de révision Wikipédia
    service_emetteur: str
    confidentialite: Confidentiality
    type_document: DocumentType
    licence: str
    source_url: str


class Article(BaseModel):
    id: str
    title: str
    opening: str
    opening_family: str
    metadata: DocumentMetadata
    fetched_at: str
    text: str
    markdown: str
    html: str


def slugify(title: str) -> str:
    table = str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc")
    return re.sub(r"[^a-z0-9]+", "-", title.lower().translate(table)).strip("-")


def wiki_to_markdown(text: str) -> str:
    """« == Titre == » → « ## Titre », « === Sous-titre === » → « ### Sous-titre »."""

    def heading(match: re.Match[str]) -> str:
        level = len(match.group(1))
        return f"{'#' * level} {match.group(2).strip()}"

    return re.sub(r"^(={2,6})\s*(.+?)\s*\1\s*$", heading, text, flags=re.MULTILINE)


_HTML_NOISE = [
    re.compile(r"<style[^>]*>.*?</style>", re.S),
    re.compile(r'<sup[^>]*class="[^"]*reference[^"]*"[^>]*>.*?</sup>', re.S),
    re.compile(r"<!--.*?-->", re.S),
]


def clean_html(html: str) -> str:
    """Retire le bruit qui ne sert pas au découpage : styles, appels de notes, commentaires."""
    for pattern in _HTML_NOISE:
        html = pattern.sub("", html)
    return html


Sleep = Callable[[float], Awaitable[None]]


class WikipediaClient:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        api_url: str = WIKIPEDIA_API,
        contact: str = DEFAULT_CONTACT,
        pause_s: float = PAUSE_S,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._client = client
        self._user_agent = f"P13-FFE-POC/0.1 ({contact})"
        self._api_url = api_url
        self._pause_s = pause_s
        self._sleep = sleep

    async def fetch(self, ref: ArticleRef) -> Article:
        query = await self._get(
            {
                "action": "query",
                "prop": "extracts|revisions",
                "rvprop": "ids|timestamp",
                "explaintext": 1,
                "exsectionformat": "wiki",
                "redirects": 1,
                "titles": ref.title,
            }
        )
        page = query["query"]["pages"][0]
        if page.get("missing"):
            raise LookupError(f"Article introuvable sur Wikipédia FR : {ref.title}")
        revision = page["revisions"][0]
        parsed = await self._get(
            {
                "action": "parse",
                "oldid": revision["revid"],
                "prop": "text",
                "disableeditsection": 1,
                "disabletoc": 1,
            }
        )
        text = page["extract"].strip()
        title = page["title"]
        return Article(
            id=ref.slug,
            title=title,
            opening=ref.opening,
            opening_family=ref.opening_family,
            metadata=DocumentMetadata(
                date=revision["timestamp"],
                version=revision["revid"],
                service_emetteur="wikipedia-fr",
                confidentialite="public",
                type_document="article",
                licence=LICENCE,
                source_url=f"https://fr.wikipedia.org/w/index.php?oldid={revision['revid']}",
            ),
            fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
            text=text,
            markdown=f"# {title}\n\n{wiki_to_markdown(text)}",
            html=clean_html(parsed["parse"]["text"]),
        )

    async def _get(self, params: dict[str, Any]) -> Any:
        params = {**params, "format": "json", "formatversion": 2, "maxlag": 5}

        async def call() -> Any:
            return await get_json(
                self._client,
                self._api_url,
                service=SERVICE,
                params=params,
                headers={"User-Agent": self._user_agent},
            )

        payload = await call_with_resilience(
            call, tool="wikipedia", policy=FETCH_POLICY, sleep=self._sleep
        )
        await self._sleep(self._pause_s)
        return payload


def save_article(article: Article, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{article.id}.json"
    path.write_text(article.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_corpus(directory: Path) -> list[Article]:
    return [
        Article.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ]
