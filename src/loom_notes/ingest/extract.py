"""Extraction du texte selon la source : brut, page web, fichier markdown.

Pages web : conversion structurelle HTML → markdown (titres, listes, tableaux, code conservés)
après retrait du bruit évident (navigation, pied de page, scripts). On privilégie la structure à
la détection fine du contenu principal : les pages mémorisées sont choisies par l'utilisateur,
pas moissonnées — et un titre perdu coûte plus cher au retrieval qu'un menu résiduel.
"""

import asyncio
import ipaddress
import re
import socket
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
    """Télécharge la page puis délègue à extract_from_html. Seul point réseau du projet.

    Chaque requête, redirections comprises, passe par `_guard` : sans `fetch_allow_private`,
    un hôte qui est ou se résout en adresse non publique est refusé. La page est lue par
    morceaux et coupée au-delà de `fetch_max_bytes`.
    """
    try:
        scheme = httpx.URL(url).scheme
    except httpx.InvalidURL as exc:
        raise ExtractionError(f"URL invalide : {url}") from exc
    if scheme not in ("http", "https"):
        raise ExtractionError(f"seules les URL http(s) sont acceptées : {url}")

    async def guard(request: httpx.Request) -> None:
        if not settings.fetch_allow_private:
            await _refuse_non_public(request.url)

    headers = {"User-Agent": settings.user_agent, "Accept": "text/html,application/xhtml+xml"}
    try:
        async with (
            httpx.AsyncClient(
                follow_redirects=True,
                timeout=settings.fetch_timeout_s,
                headers=headers,
                event_hooks={"request": [guard]},
            ) as client,
            client.stream("GET", url) as resp,
        ):
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "html" not in ctype and "xml" not in ctype:
                raise ExtractionError(f"type de contenu non pris en charge : {ctype or 'inconnu'}")
            body = await _read_capped(resp, settings.fetch_max_bytes)
            encoding = resp.encoding or "utf-8"
            final_url = str(resp.url)
    except httpx.HTTPError as exc:
        raise ExtractionError(f"téléchargement impossible : {url} ({exc})") from exc
    return extract_from_html(body.decode(encoding, errors="replace"), url=final_url)


async def _refuse_non_public(url: httpx.URL) -> None:
    """Refuse un hôte qui est, ou dont une des adresses DNS est, non publique."""
    host = url.host
    try:
        addresses = {ipaddress.ip_address(host)}
    except ValueError:
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(
                host, url.port, type=socket.SOCK_STREAM
            )
        except socket.gaierror as exc:
            raise ExtractionError(f"hôte introuvable : {host}") from exc
        addresses = {ipaddress.ip_address(info[4][0]) for info in infos}
    for address in addresses:
        if not _is_public(address):
            raise ExtractionError(
                f"adresse non publique refusée : {host} ({address}). "
                "LOOM_NOTES_FETCH_ALLOW_PRIVATE=true pour l'autoriser."
            )


def _is_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Ni boucle locale, ni réseau privé, ni lien local, ni réservée, ni multicast."""
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_global and not address.is_multicast


async def _read_capped(resp: httpx.Response, max_bytes: int) -> bytes:
    """Lit le corps (décompressé) sans dépasser max_bytes."""
    declared = resp.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_bytes:
        raise ExtractionError(f"page trop volumineuse : {declared} octets (maximum {max_bytes})")
    chunks: list[bytes] = []
    size = 0
    async for chunk in resp.aiter_bytes():
        size += len(chunk)
        if size > max_bytes:
            raise ExtractionError(f"page trop volumineuse : plus de {max_bytes} octets")
        chunks.append(chunk)
    return b"".join(chunks)


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
