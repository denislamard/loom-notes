"""Recherche : embedding de la requête → hybride Qdrant → reranking → regroupement par document."""

import asyncio
from collections.abc import Sequence

from loom_memory.embed.base import Embedder, Reranker
from loom_memory.models import Hit
from loom_memory.settings import Settings
from loom_memory.store.qdrant import MemoryStore, ScoredChunk


class Retriever:
    def __init__(
        self, settings: Settings, store: MemoryStore, embedder: Embedder, reranker: Reranker
    ) -> None:
        self._s = settings
        self._store = store
        self._embedder = embedder
        self.reranker = reranker

    async def search(
        self, query: str, project: str | None, tags: Sequence[str] | None, k: int
    ) -> list[Hit]:
        query = query.strip()
        if not query:
            return []
        emb = await asyncio.to_thread(self._embedder.embed_query, query)
        candidates = await self._store.search(
            emb, project=project, tags=tags, limit=self._s.rerank_candidates
        )
        if not candidates:
            return []
        passages = [f"{c.heading_path}\n\n{c.text}" for c in candidates]
        scores = await asyncio.to_thread(self.reranker.score, query, passages)
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda cs: cs[1], reverse=True)
        return self._group(ranked, k)

    def _group(self, ranked: list[tuple[ScoredChunk, float]], k: int) -> list[Hit]:
        per_doc: dict[str, int] = {}
        hits: list[Hit] = []
        for chunk, score in ranked:
            if score < self._s.min_score:
                break
            n = per_doc.get(chunk.doc_id, 0)
            if n >= self._s.max_chunks_per_doc:
                continue
            per_doc[chunk.doc_id] = n + 1
            hits.append(
                Hit(
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    project=chunk.project,
                    score=round(score, 4),
                    snippet=_snippet(chunk.text, self._s.snippet_chars),
                    heading_path=chunk.heading_path,
                    source=chunk.source,
                    added_at=chunk.added_at,
                )
            )
            if len(hits) >= k:
                break
        return hits


def _snippet(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    cut = flat.rfind(" ", 0, limit)
    return flat[: cut if cut > limit // 2 else limit].rstrip() + "…"
