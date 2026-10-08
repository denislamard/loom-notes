"""Interfaces des modèles. Synchrones : le service les exécute dans un thread."""

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


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


def resolve_device(device: str) -> str:
    """Résout `auto` en cuda, sinon mps, sinon cpu ; toute autre valeur est rendue telle quelle.

    torch n'est importé que pour `auto`, au chargement des vrais modèles.
    """
    if device != "auto":
        return device
    torch: Any = importlib.import_module("torch")
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
