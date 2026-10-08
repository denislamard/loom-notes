from pathlib import Path

import pytest

from loom_notes.ingest.extract import ExtractionError, extract_from_html, read_markdown_file

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
