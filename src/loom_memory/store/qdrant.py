"""Persistance Qdrant (mode local embarqué).

Deux collections :
- `documents` : sans vecteur, texte intégral + métadonnées. C'est la source de vérité.
- `memory`    : chunks vectorisés (dense + sparse), reconstructible depuis `documents`.
"""

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm
from qdrant_client.conversions import common_types as ct

from loom_memory.embed.base import Embedding
from loom_memory.ids import point_id
from loom_memory.models import Chunk, Document, utc_now_iso
from loom_memory.settings import Settings

_SCROLL_PAGE = 256


class ModelMismatchError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    doc_id: str
    index: int
    heading_path: str
    text: str
    score: float
    title: str
    project: str
    source: str | None
    added_at: str


class MemoryStore:
    def __init__(self, settings: Settings, client: AsyncQdrantClient, *, is_local: bool) -> None:
        self._s = settings
        self._c = client
        self.is_local = is_local

    @classmethod
    def open(cls, settings: Settings) -> "MemoryStore":
        if settings.qdrant_url:
            client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
            return cls(settings, client, is_local=False)
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        return cls(settings, AsyncQdrantClient(path=str(settings.qdrant_path)), is_local=True)

    async def close(self) -> None:
        await self._c.close()

    # ---------- schéma ----------

    async def ensure_schema(
        self, dense_model: str, dense_dim: int, *, check_model: bool = True
    ) -> None:
        """Crée les collections si besoin et vérifie que le modèle dense n'a pas changé."""
        meta = self._read_meta()
        if (
            check_model
            and meta is not None
            and (meta.get("dense_model") != dense_model or meta.get("dense_dim") != dense_dim)
        ):
            raise ModelMismatchError(
                f"la base a été indexée avec {meta.get('dense_model')} ({meta.get('dense_dim')}), "
                f"la configuration demande {dense_model} ({dense_dim}) : "
                "lance `loom-memory reindex`"
            )
        if not await self._c.collection_exists(self._s.documents_collection):
            await self._c.create_collection(self._s.documents_collection, vectors_config={})
            if not self.is_local:
                for field in ("content_hash", "project"):
                    await self._c.create_payload_index(
                        self._s.documents_collection, field, qm.PayloadSchemaType.KEYWORD
                    )
        if not await self._c.collection_exists(self._s.chunks_collection):
            await self._create_chunks_collection(dense_dim)
        if meta is None:
            self._write_meta(dense_model, dense_dim)

    async def reset_chunks(self, dense_model: str, dense_dim: int) -> None:
        """Supprime et recrée la collection de chunks (réindexation)."""
        if await self._c.collection_exists(self._s.chunks_collection):
            await self._c.delete_collection(self._s.chunks_collection)
        await self._create_chunks_collection(dense_dim)
        self._write_meta(dense_model, dense_dim)

    async def _create_chunks_collection(self, dense_dim: int) -> None:
        await self._c.create_collection(
            self._s.chunks_collection,
            vectors_config={"dense": qm.VectorParams(size=dense_dim, distance=qm.Distance.COSINE)},
            sparse_vectors_config={"sparse": qm.SparseVectorParams()},
        )
        # Les index payload sont ignorés en mode local ; ils comptent dès qu'on passe sur un
        # serveur Qdrant (Docker). On les déclare donc seulement dans ce cas.
        if not self.is_local:
            for field in ("project", "tags", "doc_id"):
                await self._c.create_payload_index(
                    self._s.chunks_collection, field, qm.PayloadSchemaType.KEYWORD
                )

    def _read_meta(self) -> dict[str, Any] | None:
        p = self._s.meta_path
        if not p.is_file():
            return None
        data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        return data

    def _write_meta(self, dense_model: str, dense_dim: int) -> None:
        self._s.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self._s.meta_path.write_text(
            json.dumps(
                {"dense_model": dense_model, "dense_dim": dense_dim, "indexed_at": utc_now_iso()},
                indent=2,
            ),
            encoding="utf-8",
        )

    # ---------- écriture ----------

    async def upsert_document(
        self, doc: Document, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks et embeddings de longueurs différentes")
        await self.delete_chunks(doc.doc_id)
        await self._c.upsert(
            self._s.documents_collection,
            points=[qm.PointStruct(id=doc.doc_id, vector={}, payload=doc.model_dump())],
        )
        if not chunks:
            return
        points = [
            qm.PointStruct(
                id=point_id(doc.doc_id, chunk.index),
                vector={
                    "dense": emb.dense,
                    "sparse": qm.SparseVector(indices=emb.sparse.indices, values=emb.sparse.values),
                },
                payload={
                    "doc_id": doc.doc_id,
                    "index": chunk.index,
                    "heading_path": chunk.heading_path,
                    "text": chunk.text,
                    "title": doc.title,
                    "project": doc.project,
                    "tags": doc.tags,
                    "source": doc.source,
                    "added_at": doc.added_at,
                },
            )
            for chunk, emb in zip(chunks, embeddings, strict=True)
        ]
        await self._c.upsert(self._s.chunks_collection, points=points)

    async def delete_chunks(self, doc_id: str) -> None:
        await self._c.delete(
            self._s.chunks_collection,
            points_selector=qm.FilterSelector(filter=_doc_filter(doc_id)),
        )

    async def delete_document(self, doc_id: str) -> None:
        await self.delete_chunks(doc_id)
        await self._c.delete(
            self._s.documents_collection, points_selector=qm.PointIdsList(points=[doc_id])
        )

    # ---------- lecture ----------

    async def get_document(self, doc_id: str) -> Document | None:
        try:
            points = await self._c.retrieve(
                self._s.documents_collection, ids=[doc_id], with_payload=True
            )
        except ValueError:  # id mal formé (pas un UUID)
            return None
        return Document.model_validate(points[0].payload) if points else None

    async def find_by_hash(self, content_hash: str) -> Document | None:
        points, _ = await self._c.scroll(
            self._s.documents_collection,
            scroll_filter=qm.Filter(
                must=[
                    qm.FieldCondition(key="content_hash", match=qm.MatchValue(value=content_hash))
                ]
            ),
            limit=1,
            with_payload=True,
        )
        return Document.model_validate(points[0].payload) if points else None

    async def iter_documents(self, project: str | None = None) -> AsyncIterator[Document]:
        offset: ct.PointId | None = None
        flt = _filter(project=project) if project else None
        while True:
            points, offset = await self._c.scroll(
                self._s.documents_collection,
                scroll_filter=flt,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
            )
            for p in points:
                yield Document.model_validate(p.payload)
            if offset is None:
                break

    async def list_documents(self, project: str | None, limit: int) -> list[Document]:
        docs = [d async for d in self.iter_documents(project)]
        docs.sort(key=lambda d: d.added_at, reverse=True)
        return docs[:limit]

    async def project_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        async for d in self.iter_documents():
            counts[d.project] = counts.get(d.project, 0) + 1
        return dict(sorted(counts.items()))

    async def count_documents(self) -> int:
        return (await self._c.count(self._s.documents_collection, exact=True)).count

    async def search(
        self,
        query: Embedding,
        project: str | None,
        tags: Sequence[str] | None,
        limit: int,
    ) -> list[ScoredChunk]:
        """Hybride dense + sparse, fusion RRF côté Qdrant, filtres dans chaque prefetch."""
        flt = _filter(project=project, tags=tags)
        res = await self._c.query_points(
            self._s.chunks_collection,
            prefetch=[
                qm.Prefetch(
                    query=query.dense, using="dense", limit=self._s.prefetch_limit, filter=flt
                ),
                qm.Prefetch(
                    query=qm.SparseVector(indices=query.sparse.indices, values=query.sparse.values),
                    using="sparse",
                    limit=self._s.prefetch_limit,
                    filter=flt,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        out: list[ScoredChunk] = []
        for p in res.points:
            pl: dict[str, Any] = p.payload or {}
            out.append(
                ScoredChunk(
                    doc_id=str(pl["doc_id"]),
                    index=int(pl["index"]),
                    heading_path=str(pl["heading_path"]),
                    text=str(pl["text"]),
                    score=float(p.score),
                    title=str(pl["title"]),
                    project=str(pl["project"]),
                    source=pl.get("source"),
                    added_at=str(pl["added_at"]),
                )
            )
        return out


def _doc_filter(doc_id: str) -> qm.Filter:
    return qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])


def _filter(*, project: str | None = None, tags: Sequence[str] | None = None) -> qm.Filter | None:
    must: list[qm.Condition] = []
    if project:
        must.append(qm.FieldCondition(key="project", match=qm.MatchValue(value=project)))
    for tag in tags or ():
        must.append(qm.FieldCondition(key="tags", match=qm.MatchValue(value=tag)))
    return qm.Filter(must=must) if must else None
