from pathlib import Path

import pytest

from loom_memory.ingest.extract import ExtractionError, extract_from_html, read_markdown_file

HTML = """<html><head><title>Ma page</title></head><body>
<nav>Accueil Contact</nav>
<main><h1>Titre principal</h1>
<p>Premier paragraphe assez long pour être considéré comme du contenu principal.</p>
<p>Deuxième paragraphe, tout aussi long, qui parle de plomberie et de devis pour les artisans.</p>
</main><footer>© 2026</footer></body></html>"""


def test_extract_from_html_keeps_main_content() -> None:
    ex = extract_from_html(HTML, url="https://example.org/page")
    assert ex.title in {"Ma page", "Titre principal"}
    assert "plomberie" in ex.text
    assert "Accueil Contact" not in ex.text


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
