"""Chunking structurel : découpage par titres markdown, puis retaille des sections trop
longues en paragraphes avec chevauchement. Chaque chunk connaît son chemin de titres."""

import re
from dataclasses import dataclass

from loom_notes.models import Chunk
from loom_notes.settings import Settings

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_PARA_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


@dataclass(slots=True)
class _Section:
    heading_path: str
    body: str


def chunk_document(doc_id: str, title: str, text: str, settings: Settings) -> list[Chunk]:
    sections = _split_sections(title, text)
    pieces: list[tuple[str, str]] = []
    for section in sections:
        for piece in _split_body(section.body, settings):
            pieces.append((section.heading_path, piece))
    pieces = _merge_tiny(pieces, settings)
    return [
        Chunk(doc_id=doc_id, index=i, heading_path=path, text=body)
        for i, (path, body) in enumerate(pieces)
    ]


def _split_sections(title: str, text: str) -> list[_Section]:
    """Une section par titre rencontré ; le titre du document est la racine du chemin."""
    stack: list[tuple[int, str]] = []
    sections: list[_Section] = []
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(_Section(_path(title, stack), body))
        buffer.clear()

    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue
        m = None if in_fence else _HEADING.match(line)
        if m is None:
            buffer.append(line)
            continue
        flush()
        level, heading = len(m.group(1)), m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        # On ne répète pas le titre du document s'il est aussi le premier H1.
        if not (level == 1 and not stack and heading == title):
            stack.append((level, heading))
    flush()
    return sections or [_Section(title, "")]


def _path(title: str, stack: list[tuple[int, str]]) -> str:
    return " > ".join([title, *(h for _, h in stack)])


def _split_body(body: str, settings: Settings) -> list[str]:
    if not body:
        return []
    if len(body) <= settings.chunk_max_chars:
        return [body]
    paragraphs = [p.strip() for p in _PARA_SPLIT.split(body) if p.strip()]
    # Une unité seule doit tenir dans un chunk avec le chevauchement qui la précède.
    unit_max = max(settings.chunk_target_chars - settings.chunk_overlap_chars, 1)
    units: list[str] = []
    for p in paragraphs:
        units.extend(_split_long(p, unit_max) if len(p) > unit_max else [p])

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for unit in units:
        if current and size + len(unit) + 2 > settings.chunk_target_chars:
            chunks.append("\n\n".join(current))
            tail = _tail(chunks[-1], settings.chunk_overlap_chars)
            current = [tail] if tail else []
            size = len(tail)
        current.append(unit)
        size += len(unit) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Un paragraphe seul plus long que max : coupe aux phrases, sinon en dur."""
    out: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(paragraph):
        if len(sentence) > max_chars:
            if current:
                out.append(current)
                current = ""
            out.extend(sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars))
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            out.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        out.append(current)
    return out


def _tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return ""
    tail = text[-overlap:]
    cut = tail.find(" ")
    return tail[cut + 1 :] if cut != -1 else tail


def _merge_tiny(pieces: list[tuple[str, str]], settings: Settings) -> list[tuple[str, str]]:
    """Absorbe les morceaux trop courts dans le précédent quand ça tient."""
    merged: list[tuple[str, str]] = []
    for path, body in pieces:
        if (
            merged
            and len(body) < settings.chunk_min_chars
            and len(merged[-1][1]) + len(body) + 2 <= settings.chunk_max_chars
        ):
            prev_path, prev_body = merged[-1]
            merged[-1] = (prev_path, f"{prev_body}\n\n{body}")
        else:
            merged.append((path, body))
    return merged
