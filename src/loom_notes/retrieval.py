"""Recherche : embedding de la requête → hybride Qdrant → reranking → regroupement par document."""

import asyncio
from collections.abc import Sequence

from loom_notes.embed.base import Embedder, Reranker
from loom_notes.models import Hit
from loom_notes.settings import Settings
from loom_notes.store.qdrant import MemoryStore, ScoredChunk


class Retriever:
    def __init__(
        self, settings: Settings, store: MemoryStore, embedder: Embedder, reranker: Reranker
    ) -> None:
        self._s = settings
        self._store = store
        self._embedder = embedder
        self.reranker = reranker

    async def search(
        self,
        query: str,
        project: str | None,
        tags: Sequence[str] | None,
        k: int,
        *,
        min_score: float | None = None,
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
        threshold = self._s.min_score if min_score is None else min_score
        return self._group(ranked, k, threshold)

    def _group(
        self, ranked: list[tuple[ScoredChunk, float]], k: int, threshold: float
    ) -> list[Hit]:
        per_doc: dict[str, int] = {}
        hits: list[Hit] = []
        for chunk, score in ranked:
            if score < threshold:
                break
            n = per_doc.get(chunk.doc_id, 0)
            if n >= self._s.max_chunks_per_doc:
                continue
            per_doc[chunk.doc_id] = n + 1
            short = len(chunk.text) <= self._s.full_chunk_chars
            hits.append(
                Hit(
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    project=chunk.project,
                    score=round(score, 4),
                    snippet=chunk.text if short else _snippet(chunk.text, self._s.snippet_chars),
                    truncated=not short,
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
