import functools
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest

from loom_notes.ingest.extract import (
    ExtractionError,
    extract_from_html,
    fetch_url,
    read_markdown_file,
)
from loom_notes.settings import Settings

HTML = """<html><head><title>Ma page</title></head><body>
<nav>Accueil Contact</nav>
<main><h1>Titre principal</h1>
<p>Premier paragraphe assez long pour être considéré comme du contenu principal.</p>
<p>Deuxième paragraphe, tout aussi long, qui parle de plomberie et de devis pour les artisans.</p>
</main><footer>© 2026</footer></body></html>"""


def test_extract_from_html_keeps_main_content() -> None:
    ex = extract_from_html(HTML, url="https://example.org/page")
    assert ex.title == "Ma page"
    assert ex.text.startswith("# Titre principal")
    assert "plomberie" in ex.text
    assert "Accueil Contact" not in ex.text and "© 2026" not in ex.text


def test_extract_keeps_structure_and_separates_glued_labels() -> None:
    html = """<html><head><meta property="og:title" content="Guide"><title>x</title></head><body>
    <header><nav>menu</nav></header>
    <section><p class="eyebrow">Engagements</p><h2>Ce que vous voulez savoir.</h2>
    <p><span class="lbl">Ce qui arrive</span>L'agent plante.</p>
    <p><span class="lbl">Ce que je fais</span><strong>Il reprend.</strong>État persisté.</p>
    <p>Des <code>tokens</code>s et du <em>texte</em>, normal.</p>
    <h3>Détail</h3><ul><li>un</li><li>deux</li></ul>
    <pre><code>x = 1</code></pre>
    <table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>
    </section><footer>pied</footer></body></html>"""
    ex = extract_from_html(html, url="https://example.org/g")
    assert ex.title == "Guide"
    assert "## Ce que vous voulez savoir." in ex.text
    assert "### Détail" in ex.text
    assert "Ce qui arrive L'agent plante." in ex.text
    assert "Ce que je fais **Il reprend.** État persisté." in ex.text
    assert "Des `tokens`s et du *texte*, normal." in ex.text
    assert "- un\n- deux" in ex.text
    assert "```\nx = 1\n```" in ex.text
    assert "| a | b |" in ex.text
    assert "menu" not in ex.text and "pied" not in ex.text


def test_extract_from_html_empty_raises() -> None:
    with pytest.raises(ExtractionError):
        extract_from_html("<html><body></body></html>", url="https://example.org/vide")


def test_read_markdown_file_title_from_h1(tmp_path: Path) -> None:
    p = tmp_path / "note.md"
    p.write_text("# Ma note\n\ncontenu", encoding="utf-8")
    ex = read_markdown_file(p)
    assert ex.title == "Ma note"
    assert ex.text.startswith("# Ma note")


def test_read_markdown_file_title_fallback_to_stem(tmp_path: Path) -> None:
    p = tmp_path / "sans-titre.md"
    p.write_text("juste du texte", encoding="utf-8")
    assert read_markdown_file(p).title == "sans-titre"


def test_read_markdown_file_missing(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError):
        read_markdown_file(tmp_path / "absent.md")


# ---------- fetch_url : le réseau est remplacé par httpx.MockTransport ----------

PUBLIC = "http://93.184.216.34"  # adresse publique littérale : ni DNS ni connexion


def _serve(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> list[str]:
    """Branche handler à la place du réseau ; renvoie les URL réellement demandées."""
    seen: list[str] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return handler(request)

    client = functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(record))
    monkeypatch.setattr(httpx, "AsyncClient", client)
    return seen


def _page(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=HTML)


async def test_fetch_url_reads_a_public_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = _serve(monkeypatch, _page)
    ex = await fetch_url(f"{PUBLIC}/page", Settings(data_dir=tmp_path))
    assert ex.title == "Ma page"
    assert seen == [f"{PUBLIC}/page"]


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:6333/collections",
        "http://localhost:8080/",
        "http://[::1]/",
        "http://10.0.0.1/",
        "http://192.168.1.10/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::ffff:127.0.0.1]/",
    ],
)
async def test_fetch_url_refuses_non_public_hosts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, url: str
) -> None:
    seen = _serve(monkeypatch, _page)
    with pytest.raises(ExtractionError, match="adresse non publique"):
        await fetch_url(url, Settings(data_dir=tmp_path))
    assert seen == []


async def test_fetch_url_checks_every_redirect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://127.0.0.1:6333/collections"})

    seen = _serve(monkeypatch, handler)
    with pytest.raises(ExtractionError, match="adresse non publique"):
        await fetch_url(f"{PUBLIC}/redirige", Settings(data_dir=tmp_path))
    assert seen == [f"{PUBLIC}/redirige"]


async def test_fetch_url_allow_private(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = _serve(monkeypatch, _page)
    settings = Settings(data_dir=tmp_path, fetch_allow_private=True)
    assert (await fetch_url("http://127.0.0.1:8000/", settings)).title == "Ma page"
    assert seen == ["http://127.0.0.1:8000/"]


async def test_fetch_url_refuses_a_declared_oversize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _serve(monkeypatch, _page)  # Content-Length annonce plus de 100 octets
    with pytest.raises(ExtractionError, match="trop volumineuse"):
        await fetch_url(f"{PUBLIC}/", Settings(data_dir=tmp_path, fetch_max_bytes=100))


async def test_fetch_url_cuts_an_undeclared_oversize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def body() -> AsyncIterator[bytes]:
        for _ in range(10):
            yield b"<p>" + b"x" * 50 + b"</p>"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=body())

    _serve(monkeypatch, handler)
    with pytest.raises(ExtractionError, match="plus de 100 octets"):
        await fetch_url(f"{PUBLIC}/", Settings(data_dir=tmp_path, fetch_max_bytes=100))


@pytest.mark.parametrize("url", ["ftp://example.org/x", "file:///etc/passwd"])
async def test_fetch_url_refuses_other_schemes(tmp_path: Path, url: str) -> None:
    with pytest.raises(ExtractionError, match="http"):
        await fetch_url(url, Settings(data_dir=tmp_path))


async def test_fetch_url_refuses_non_html(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF")

    _serve(monkeypatch, handler)
    with pytest.raises(ExtractionError, match="type de contenu"):
        await fetch_url(f"{PUBLIC}/doc.pdf", Settings(data_dir=tmp_path))
