from loom_memory.ingest.chunk import chunk_document
from loom_memory.settings import Settings

DOC = """# Guide

Intro courte.

## Installation

Étape un. Étape deux.

### Linux

Détail linux.

## Usage

Paragraphe d'usage.
"""


def test_sections_carry_heading_path(settings: Settings) -> None:
    settings.chunk_min_chars = 1  # pas de fusion des petites sections ici
    chunks = chunk_document("d", "Guide", DOC, settings)
    paths = [c.heading_path for c in chunks]
    assert paths == [
        "Guide",
        "Guide > Installation",
        "Guide > Installation > Linux",
        "Guide > Usage",
    ]
    assert [c.index for c in chunks] == [0, 1, 2, 3]
    assert chunks[2].embed_text.startswith("Guide > Installation > Linux\n\n")


def test_tiny_sections_are_merged(settings: Settings) -> None:
    chunks = chunk_document("d", "Guide", DOC, settings)
    assert len(chunks) == 1
    assert "Détail linux." in chunks[0].text


def test_long_section_is_split_with_overlap(settings: Settings) -> None:
    settings.chunk_target_chars = 300
    settings.chunk_max_chars = 400
    settings.chunk_overlap_chars = 40
    paragraphs = [f"Paragraphe numéro {i} " + "mot " * 30 for i in range(12)]
    text = "# T\n\n## S\n\n" + "\n\n".join(paragraphs)
    chunks = chunk_document("d", "T", text, settings)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 + 60 for c in chunks)
    assert all(c.heading_path == "T > S" for c in chunks)
    # Le chevauchement : la fin du chunk n reparaît au début du chunk n+1.
    tail = chunks[0].text[-20:]
    assert tail in chunks[1].text


def test_headings_inside_code_fences_are_ignored(settings: Settings) -> None:
    settings.chunk_min_chars = 1
    text = "# T\n\ntexte\n\n```bash\n# pas un titre\necho ok\n```\n\nsuite"
    chunks = chunk_document("d", "T", text, settings)
    assert len(chunks) == 1
    assert "# pas un titre" in chunks[0].text


def test_oversized_paragraph_is_hard_split(settings: Settings) -> None:
    settings.chunk_target_chars = 200
    settings.chunk_max_chars = 200
    settings.chunk_overlap_chars = 20
    text = "Phrase. " * 100
    chunks = chunk_document("d", "T", text, settings)
    assert all(len(c.text) <= 200 for c in chunks)
    assert sum(c.text.count("Phrase") for c in chunks) >= 100
