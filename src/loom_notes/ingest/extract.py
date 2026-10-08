"""Extraction du texte selon la source : brut, page web, fichier markdown.

Pages web : conversion structurelle HTML → markdown (titres, listes, tableaux, code conservés)
après retrait du bruit évident (navigation, pied de page, scripts). On privilégie la structure à
la détection fine du contenu principal : les pages mémorisées sont choisies par Denis, pas
moissonnées — et un titre perdu coûte plus cher au retrieval qu'un menu résiduel.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import httpx
from lxml import etree
from lxml import html as lxml_html
from lxml.html import HtmlElement, tostring
from markdownify import markdownify

from loom_notes.settings import Settings

_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_NOISE_XPATH = (
    "//nav | //footer | //script | //style | //noscript | //iframe | //svg | //form | //template"
    " | //*[@aria-hidden='true'] | //*[@role='navigation'] | //*[@role='banner' and .//nav]"
)
_INLINE_TAGS = ("span", "strong", "b", "em", "i", "code", "a", "small", "abbr", "time")


class ExtractionError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Extracted:
    title: str
    text: str


def extract_from_html(html: str, url: str) -> Extracted:
    """Contenu de la page en markdown structuré + titre. Lève ExtractionError si vide."""
    try:
        doc = cast(HtmlElement, lxml_html.fromstring(html))  # pyright: ignore[reportUnknownMemberType]
    except (ValueError, etree.ParserError) as exc:
        raise ExtractionError(f"HTML illisible : {url} ({exc})") from exc
    title = _page_title(doc) or url
    for node in _elements(doc, _NOISE_XPATH):
        parent = node.getparent()
        if parent is not None:
            parent.remove(node)
    _separate_glued_inlines(doc)
    roots = _elements(doc, "//main") or _elements(doc, "//article") or _elements(doc, "//body")
    fragment = cast(str, tostring(roots[0] if roots else doc, encoding="unicode"))
    text = markdownify(
        fragment, heading_style="ATX", strip=["img", "a"], bullets="-", code_language=""
    )
    text = _tidy(text)
    if not text:
        raise ExtractionError(f"aucun contenu exploitable extrait de {url}")
    return Extracted(title=title, text=text)


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


def _elements(doc: HtmlElement, xpath: str) -> list[HtmlElement]:
    return cast(list[HtmlElement], doc.xpath(xpath))


def _page_title(doc: HtmlElement) -> str | None:
    for xpath in (
        "//meta[@property='og:title']/@content",
        "//head/title/text()",
        "//h1//text()",
    ):
        values = cast(list[str], doc.xpath(xpath))
        joined = " ".join(" ".join(values).split())
        if joined:
            return joined
    return None


def _separate_glued_inlines(doc: HtmlElement) -> None:
    """`<span>Label</span>Texte` ou `<span>Label</span><strong>…` sans espace entre les deux :
    on insère un espace, sinon le markdown colle les mots (« Ce qui arriveL'agent »)."""
    for el in doc.iter(*_INLINE_TAGS):
        text = "".join(el.itertext()).rstrip()
        if not text:
            continue
        ends_word = text[-1].isalnum()
        ends_sentence = text[-1] in ".!?:;"
        tail = el.tail or ""
        if tail:
            if (ends_word and tail[0].isupper()) or (ends_sentence and tail[0].isalpha()):
                el.tail = " " + tail
            continue
        nxt = el.getnext()
        if ends_word and nxt is not None and isinstance(nxt.tag, str) and nxt.tag in _INLINE_TAGS:
            head = "".join(nxt.itertext()).lstrip()
            if head and head[0].isalnum():
                el.tail = " "


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
