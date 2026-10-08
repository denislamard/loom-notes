"""Modèles factices et déterministes pour les tests : pas de GPU, pas de téléchargement.
Le dense est un sac de mots haché, le sparse compte les mots — assez pour que la recherche
lexicale se comporte de façon plausible."""

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence

from loom_notes.embed.base import Embedding, SparseVec

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.casefold() for t in _TOKEN.findall(text) if len(t) > 1]


def _bucket(token: str, mod: int) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big") % mod


class FakeEmbedder:
    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return f"fake-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _one(self, text: str) -> Embedding:
        counts = Counter(_tokens(text))
        dense = [0.0] * self._dim
        for tok, n in counts.items():
            dense[_bucket(tok, self._dim)] += float(n)
        norm = math.sqrt(sum(x * x for x in dense)) or 1.0
        dense = [x / norm for x in dense]
        sparse_counts: dict[int, float] = {}
        for tok, n in counts.items():
            idx = _bucket(tok, 1 << 20)
            sparse_counts[idx] = sparse_counts.get(idx, 0.0) + float(n)
        items = sorted(sparse_counts.items())
        return Embedding(
            dense=dense,
            sparse=SparseVec(indices=[i for i, _ in items], values=[v for _, v in items]),
        )

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [self._one(t) for t in texts]

    def embed_query(self, text: str) -> Embedding:
        return self._one(text)


class FakeReranker:
    @property
    def model_name(self) -> str:
        return "fake-reranker"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        q = set(_tokens(query))
        if not q:
            return [0.0] * len(passages)
        return [len(q & set(_tokens(p))) / len(q) for p in passages]
