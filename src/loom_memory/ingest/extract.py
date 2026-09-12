"""Extraction du texte selon la source : brut, page web (trafilatura), fichier markdown."""

import re
from dataclasses import dataclass
from pathlib import Path

import httpx
import trafilatura

from loom_memory.settings import Settings

_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


class ExtractionError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Extracted:
    title: str
    text: str


def extract_from_html(html: str, url: str) -> Extracted:
    """Contenu principal en markdown + titre de la page. Lève ExtractionError si vide."""
    text = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_formatting=True,
        include_links=False,
        include_images=False,
        include_tables=True,
        favor_precision=True,
    )
    if not text or not text.strip():
        raise ExtractionError(f"aucun contenu exploitable extrait de {url}")
    meta = trafilatura.extract_metadata(html, default_url=url)
    title = (meta.title if meta and meta.title else None) or _first_h1(text) or url
    return Extracted(title=title.strip(), text=text.strip())


async def fetch_url(url: str, settings: Settings) -> Extracted:
    """Télécharge la page puis délègue à extract_from_html. Seul point réseau du projet."""
    headers = {"User-Agent": settings.user_agent, "Accept": "text/html,application/xhtml+xml"}
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=settings.fetch_timeout_s, headers=headers
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise ExtractionError(f"téléchargement impossible : {url} ({exc})") from exc
    ctype = resp.headers.get("content-type", "")
    if "html" not in ctype and "xml" not in ctype:
        raise ExtractionError(f"type de contenu non pris en charge : {ctype or 'inconnu'}")
    return extract_from_html(resp.text, url=str(resp.url))


def read_markdown_file(path: Path) -> Extracted:
    """Lit un fichier markdown local. Le titre est le premier H1, sinon le nom du fichier."""
    if not path.is_file():
        raise ExtractionError(f"fichier introuvable : {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ExtractionError(f"fichier vide : {path}")
    title = _first_h1(text) or path.stem
    return Extracted(title=title, text=text)


def _first_h1(text: str) -> str | None:
    m = _H1.search(text)
    return m.group(1).strip() if m else None
