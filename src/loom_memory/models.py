"""Modèles de données partagés entre ingestion, store, CLI et serveur MCP."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

SourceKind = Literal["text", "url", "file"]


def utc_now_iso() -> str:
    # Microsecondes conservées : added_at sert de clé de tri, deux ajouts dans la même seconde
    # doivent rester ordonnés.
    return datetime.now(UTC).isoformat()


class Document(BaseModel):
    doc_id: str
    title: str
    project: str
    tags: list[str] = Field(default_factory=list)
    source_kind: SourceKind
    source: str | None = None
    added_at: str = Field(default_factory=utc_now_iso)
    content_hash: str
    text: str


class Chunk(BaseModel):
    doc_id: str
    index: int
    heading_path: str
    text: str

    @property
    def embed_text(self) -> str:
        """Texte réellement vectorisé : le chemin de titres préfixe le contenu."""
        return f"{self.heading_path}\n\n{self.text}" if self.heading_path else self.text


class DocSummary(BaseModel):
    doc_id: str
    title: str
    project: str
    tags: list[str]
    source_kind: SourceKind
    source: str | None
    added_at: str
    chars: int

    @classmethod
    def from_document(cls, doc: Document) -> "DocSummary":
        return cls(
            doc_id=doc.doc_id,
            title=doc.title,
            project=doc.project,
            tags=doc.tags,
            source_kind=doc.source_kind,
            source=doc.source,
            added_at=doc.added_at,
            chars=len(doc.text),
        )


class Hit(BaseModel):
    doc_id: str
    title: str
    project: str
    score: float
    snippet: str
    heading_path: str
    source: str | None
    added_at: str


class IngestResult(BaseModel):
    doc_id: str
    title: str
    project: str
    chunks: int
    duplicate_of: str | None = None
