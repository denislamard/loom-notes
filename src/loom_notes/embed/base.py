"""Interfaces des modèles. Synchrones : le service les exécute dans un thread."""

import functools
import importlib
import logging
import threading
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

log = logging.getLogger("loom_notes")


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


class ModelLoadError(RuntimeError):
    """Un modèle n'a pas pu être chargé, ou n'a pas tourné une première fois.

    Rien n'est gardé en cache : l'appel suivant recharge le modèle.
    """


@functools.cache
def resolve_device(device: str) -> str:
    """Résout `auto` en cuda, sinon mps, sinon cpu ; toute autre valeur est rendue telle quelle.

    `cuda` n'est choisi que si torch a des noyaux pour la carte : un GPU vu par torch mais trop
    ancien pour la roue installée fait échouer les modèles plus loin, sans message utile. Il est
    écarté avec un avertissement. torch n'est importé que pour `auto`, au chargement des vrais
    modèles ; le résultat est gardé, l'avertissement n'est donc écrit qu'une fois.
    """
    if device != "auto":
        return device
    torch: Any = importlib.import_module("torch")
    if torch.cuda.is_available():
        problem = unsupported_gpu(torch)
        if problem is None:
            return "cuda"
        log.warning("%s : modèles sur cpu (LOOM_NOTES_DEVICE pour forcer)", problem)
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def unsupported_gpu(torch: Any) -> str | None:
    """Dit pourquoi la carte 0 n'est pas utilisable par ce torch, ou None si elle l'est.

    Même règle que torch : un binaire `sm_XY` tourne sur une carte de même version majeure X
    et de version mineure au moins Y ; un code `compute_XY` (PTX) est compilé à la volée pour
    toute carte de version au moins X.Y. Une liste vide ne permet pas de juger : la carte est
    gardée. Les avertissements de torch sur la carte sont tus, celui-ci les remplace.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        major, minor = (int(v) for v in torch.cuda.get_device_capability(0))
        name = str(torch.cuda.get_device_name(0))
        archs = [str(a) for a in torch.cuda.get_arch_list()]
    binaries: list[tuple[int, int]] = []
    ptx: list[tuple[int, int]] = []
    for arch in archs:
        kind, _, version = arch.partition("_")
        if not version.isdigit() or len(version) < 2:
            continue
        target = (int(version[:-1]), int(version[-1]))
        (binaries if kind == "sm" else ptx if kind == "compute" else []).append(target)
    if not binaries and not ptx:
        return None
    if any(m == major and n <= minor for m, n in binaries) or any(
        (m, n) <= (major, minor) for m, n in ptx
    ):
        return None
    known = ", ".join(f"sm_{m}{n}" for m, n in sorted(binaries)) or "aucune"
    return (
        f"GPU {name} (sm_{major}{minor}) non pris en charge par ce torch "
        f"({torch.__version__}, architectures : {known})"
    )


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    line = text.splitlines()[0].strip() if text else ""
    return line or type(exc).__name__


class LazyModel[T]:
    """Modèle chargé à la première utilisation, une seule fois, et utilisé sous verrou.

    Le chargement de FlagEmbedding (transformers, tqdm) n'est pas sûr entre threads : deux
    chargements simultanés — le préchargement du serveur et une première requête — cassaient
    le modèle (« Cannot copy out of meta tensor », verrou de tqdm retiré). Le verrou les met
    en file, et sert aussi pour l'usage : rien ne garantit qu'un modèle FlagEmbedding se partage
    entre threads, et un seul appareil ne calcule de toute façon pas deux lots à la fois.

    Le modèle construit tourne une première fois (`check`) avant d'être gardé : un modèle qui
    ne tourne pas est jeté, l'erreur devient `ModelLoadError`, et l'appel suivant recharge.
    """

    def __init__(
        self,
        name: str,
        device: Callable[[], str],
        build: Callable[[Any, str], T],
        check: Callable[[T], object],
    ) -> None:
        self._name = name
        self._device = device
        self._build = build
        self._check = check
        self._lock = threading.Lock()
        self._model: T | None = None

    def run[R](self, fn: Callable[[T], R]) -> R:
        with self._lock:
            if self._model is None:
                self._model = self._load()
            return fn(self._model)

    def _load(self) -> T:
        try:
            flag: Any = importlib.import_module("FlagEmbedding")
        except ImportError as exc:
            raise ModelLoadError('FlagEmbedding absent : pip install "loom-notes[models]"') from exc
        device = "?"
        try:
            device = self._device()
            model = self._build(flag, device)
            self._check(model)
        except Exception as exc:
            raise ModelLoadError(
                f"Chargement du modèle {self._name} impossible sur {device} : "
                f"{_first_line(exc)}. Réessayer ; si l'erreur persiste, vérifier "
                "LOOM_NOTES_DEVICE et l'installation de torch."
            ) from exc
        return model
