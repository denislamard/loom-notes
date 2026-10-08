"""Interfaces des modèles. Synchrones : le service les exécute dans un thread."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SparseVec:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True, slots=True)
class Embedding:
    dense: list[float]
    sparse: SparseVec


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]: ...

    def embed_query(self, text: str) -> Embedding: ...


class Reranker(Protocol):
    @property
    def model_name(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...
