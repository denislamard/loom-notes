"""API unique de la mémoire, consommée par la CLI et par le serveur MCP."""

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from loom_memory.embed.base import Embedder, Reranker
from loom_memory.ids import new_doc_id
from loom_memory.ingest import (
    Extracted,
    chunk_document,
    content_hash,
    fetch_url,
    read_markdown_file,
)
from loom_memory.models import DocSummary, Document, Hit, IngestResult, SourceKind
from loom_memory.retrieval import Retriever
from loom_memory.settings import Settings
from loom_memory.store.qdrant import MemoryStore


class NotFoundError(Exception):
    pass


class MemoryService:
    def __init__(
        self, settings: Settings, store: MemoryStore, embedder: Embedder, reranker: Reranker
    ) -> None:
        self._s = settings
        self._store = store
        self._embedder = embedder
        self._retriever = Retriever(settings, store, embedder, reranker)

    async def start(self, *, check_model: bool = True) -> None:
        """`check_model=False` uniquement pour `reindex`, qui remplace l'index existant."""
        await self._store.ensure_schema(
            self._embedder.model_name, self._embedder.dim, check_model=check_model
        )

    async def close(self) -> None:
        await self._store.close()

    # ---------- lecture ----------

    async def search(
        self, query: str, project: str | None = None, tags: Sequence[str] | None = None, k: int = 5
    ) -> list[Hit]:
        return await self._retriever.search(query, _clean_project(project), _clean_tags(tags), k)

    async def get(self, doc_id: str) -> Document:
        doc = await self._store.get_document(doc_id)
        if doc is None:
            raise NotFoundError(f"document inconnu : {doc_id}")
        return doc

    async def list_docs(self, project: str | None = None, n: int = 20) -> list[DocSummary]:
        docs = await self._store.list_documents(_clean_project(project), n)
        return [DocSummary.from_document(d) for d in docs]

    async def projects(self) -> dict[str, int]:
        return await self._store.project_counts()

    # ---------- écriture ----------

    async def add_text(
        self, text: str, title: str, project: str, tags: Sequence[str] | None = None
    ) -> IngestResult:
        title = title.strip()
        if len(title) < 3:
            raise ValueError("titre trop court")
        return await self._ingest(Extracted(title, text.strip()), "text", None, project, tags)

    async def add_url(
        self, url: str, project: str, tags: Sequence[str] | None = None
    ) -> IngestResult:
        extracted = await fetch_url(url.strip(), self._s)
        return await self._ingest(extracted, "url", url.strip(), project, tags)

    async def add_file(
        self, path: str | Path, project: str, tags: Sequence[str] | None = None
    ) -> IngestResult:
        p = Path(path).expanduser().resolve()
        extracted = read_markdown_file(p)
        return await self._ingest(extracted, "file", str(p), project, tags)

    async def update(self, doc_id: str, text: str) -> IngestResult:
        old = await self.get(doc_id)
        return await self._ingest(
            Extracted(old.title, text.strip()),
            old.source_kind,
            old.source,
            old.project,
            old.tags,
            doc_id=old.doc_id,
            added_at=old.added_at,
        )

    async def delete(self, doc_id: str) -> DocSummary:
        doc = await self.get(doc_id)
        await self._store.delete_document(doc_id)
        return DocSummary.from_document(doc)

    async def _ingest(
        self,
        extracted: Extracted,
        kind: SourceKind,
        source: str | None,
        project: str,
        tags: Sequence[str] | None,
        *,
        doc_id: str | None = None,
        added_at: str | None = None,
    ) -> IngestResult:
        project_clean = _clean_project(project)
        if not project_clean:
            raise ValueError("project est obligatoire")
        if not extracted.text:
            raise ValueError("contenu vide")
        digest = content_hash(extracted.text)
        existing = await self._store.find_by_hash(digest)
        if existing is not None and existing.doc_id != doc_id:
            return IngestResult(
                doc_id=existing.doc_id,
                title=existing.title,
                project=existing.project,
                chunks=0,
                duplicate_of=existing.doc_id,
            )
        doc = Document(
            doc_id=doc_id or new_doc_id(),
            title=extracted.title,
            project=project_clean,
            tags=_clean_tags(tags) or [],
            source_kind=kind,
            source=source,
            content_hash=digest,
            text=extracted.text,
            **({"added_at": added_at} if added_at else {}),
        )
        n = await self._index(doc)
        return IngestResult(doc_id=doc.doc_id, title=doc.title, project=doc.project, chunks=n)

    async def _index(self, doc: Document) -> int:
        chunks = chunk_document(doc.doc_id, doc.title, doc.text, self._s)
        embeddings = await asyncio.to_thread(
            self._embedder.embed_documents, [c.embed_text for c in chunks]
        )
        await self._store.upsert_document(doc, chunks, embeddings)
        return len(chunks)

    # ---------- maintenance ----------

    async def export(self, path: Path) -> int:
        """Un document JSON par ligne. C'est la sauvegarde de référence."""
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with path.open("w", encoding="utf-8") as f:
            async for doc in self._store.iter_documents():
                f.write(doc.model_dump_json() + "\n")
                n += 1
        return n

    async def import_(self, path: Path) -> tuple[int, int]:
        """Réimporte un export, en ignorant les doc_id déjà présents.

        Renvoie (importés, ignorés)."""
        imported = skipped = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                doc = Document.model_validate(json.loads(line))
                if await self._store.get_document(doc.doc_id) is not None:
                    skipped += 1
                    continue
                await self._index(doc)
                imported += 1
        return imported, skipped

    async def reindex(self) -> int:
        """Reconstruit tous les chunks depuis `documents` avec le modèle courant."""
        docs = [d async for d in self._store.iter_documents()]
        await self._store.reset_chunks(self._embedder.model_name, self._embedder.dim)
        for doc in docs:
            await self._index(doc)
        return len(docs)


def _clean_project(project: str | None) -> str | None:
    p = (project or "").strip().lower()
    return p or None


def _clean_tags(tags: Sequence[str] | None) -> list[str] | None:
    if not tags:
        return None
    seen: dict[str, None] = {}
    for t in tags:
        t = t.strip().lower()
        if t:
            seen[t] = None
    return list(seen) or None


def build_service(settings: Settings | None = None, *, fake: bool = False) -> MemoryService:
    """Assemble le service. `fake=True` remplace les modèles par des factices (tests, démo CPU)."""
    settings = settings or Settings()
    embedder: Embedder
    reranker: Reranker
    if fake:
        from loom_memory.embed.fake import FakeEmbedder, FakeReranker

        embedder, reranker = FakeEmbedder(), FakeReranker()
    else:
        from loom_memory.embed.bge_m3 import BgeM3Embedder
        from loom_memory.embed.reranker import BgeReranker

        embedder, reranker = BgeM3Embedder(settings), BgeReranker(settings)
    return MemoryService(settings, MemoryStore.open(settings), embedder, reranker)
