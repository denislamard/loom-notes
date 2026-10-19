"""Chargement des vrais modèles, sans torch ni FlagEmbedding : des modules factices les remplacent.

Couvre la course du 19/10 (préchargement et première requête qui chargeaient BGE-M3 en même
temps), le modèle cassé qui restait en cache, et le choix du device face à un GPU trop ancien.
"""

import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client

from loom_notes.embed.base import ModelLoadError, resolve_device, unsupported_gpu
from loom_notes.embed.bge_m3 import BgeM3Embedder
from loom_notes.embed.reranker import BgeReranker
from loom_notes.server import create_server
from loom_notes.settings import Settings


class _Flag:
    """FlagEmbedding factice : compte les constructions, peut échouer au premier passage."""

    def __init__(self, *, slow: float = 0.0, broken: int = 0) -> None:
        self.built = 0
        self.slow = slow
        self.broken = broken  # nombre de modèles construits qui échouent au premier passage
        self.lock = threading.Lock()
        self.inside = 0
        self.overlap = False

    def _enter(self) -> None:
        with self.lock:
            self.inside += 1
            self.overlap = self.overlap or self.inside > 1
        time.sleep(self.slow)
        with self.lock:
            self.inside -= 1

    def module(self) -> types.ModuleType:
        flag = self

        class _Model:
            def __init__(self, name: str, **_: Any) -> None:
                flag._enter()
                with flag.lock:
                    flag.built += 1
                    self.ok = flag.built > flag.broken

            def _go(self) -> None:
                if not self.ok:
                    raise NotImplementedError(
                        "Cannot copy out of meta tensor; no data!\nPlease use to_empty()"
                    )
                flag._enter()

            def encode(self, texts: list[str], **_: Any) -> dict[str, Any]:
                self._go()
                return {
                    "dense_vecs": [[0.5, 0.5] for _ in texts],
                    "lexical_weights": [{"3": 0.7} for _ in texts],
                }

            def compute_score(self, pairs: list[list[str]], **_: Any) -> list[float]:
                self._go()
                return [0.9 for _ in pairs]

        module = types.ModuleType("FlagEmbedding")
        module.BGEM3FlagModel = _Model  # type: ignore[attr-defined]
        module.FlagReranker = _Model  # type: ignore[attr-defined]
        return module


@pytest.fixture
def cpu_settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", device="cpu")


def _install(monkeypatch: pytest.MonkeyPatch, flag: _Flag) -> None:
    monkeypatch.setitem(sys.modules, "FlagEmbedding", flag.module())


def test_two_simultaneous_first_calls_build_the_model_once(
    monkeypatch: pytest.MonkeyPatch, cpu_settings: Settings
) -> None:
    flag = _Flag(slow=0.2)
    _install(monkeypatch, flag)
    embedder = BgeM3Embedder(cpu_settings)
    threads = [threading.Thread(target=embedder.embed_query, args=("x",)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert flag.built == 1
    assert not flag.overlap  # ni deux chargements, ni deux passages en même temps


def test_a_model_that_does_not_run_is_not_kept(
    monkeypatch: pytest.MonkeyPatch, cpu_settings: Settings
) -> None:
    flag = _Flag(broken=1)
    _install(monkeypatch, flag)
    embedder = BgeM3Embedder(cpu_settings)
    with pytest.raises(ModelLoadError) as caught:
        embedder.embed_query("x")
    assert str(caught.value) == (
        "Chargement du modèle BAAI/bge-m3 impossible sur cpu : Cannot copy out of meta tensor; "
        "no data!. Réessayer ; si l'erreur persiste, vérifier LOOM_NOTES_DEVICE et "
        "l'installation de torch."
    )
    # L'appel suivant recharge : le modèle cassé n'a pas été gardé.
    assert embedder.embed_query("x").dense == [0.5, 0.5]
    assert flag.built == 2


def test_the_reranker_is_guarded_the_same_way(
    monkeypatch: pytest.MonkeyPatch, cpu_settings: Settings
) -> None:
    flag = _Flag(broken=1)
    _install(monkeypatch, flag)
    reranker = BgeReranker(cpu_settings)
    with pytest.raises(ModelLoadError, match="bge-reranker-v2-m3 impossible sur cpu"):
        reranker.score("q", ["p"])
    assert reranker.score("q", ["p", "r"]) == [0.9, 0.9]


def test_a_missing_flagembedding_is_a_model_load_error(
    monkeypatch: pytest.MonkeyPatch, cpu_settings: Settings
) -> None:
    monkeypatch.setitem(sys.modules, "FlagEmbedding", None)
    with pytest.raises(ModelLoadError, match=r"pip install \"loom-notes\[models\]\""):
        BgeM3Embedder(cpu_settings).embed_query("x")


async def test_a_model_load_error_reaches_the_client_as_a_tool_error(
    monkeypatch: pytest.MonkeyPatch, cpu_settings: Settings
) -> None:
    flag = _Flag(broken=99)
    _install(monkeypatch, flag)

    settings = cpu_settings.model_copy(update={"warmup_on_start": False})
    async with Client(create_server(settings, fake=False)) as c:
        r = await c.call_tool("search", {"query": "tarif"}, raise_on_error=False)
    assert r.is_error
    text = r.content[0].text  # type: ignore[union-attr]
    assert "Chargement du modèle BAAI/bge-m3 impossible sur cpu" in text
    assert "Traceback" not in text


def _torch(capability: tuple[int, int], archs: list[str], *, cuda: bool = True) -> Any:
    def warn() -> None:
        import warnings

        warnings.warn("bruit de torch", UserWarning, stacklevel=1)

    def capability_of(_: int) -> tuple[int, int]:
        warn()
        return capability

    def name_of(_: int) -> str:
        return "NVIDIA GeForce GTX 1050 Ti"

    cuda_ns = types.SimpleNamespace(
        is_available=lambda: cuda,
        get_device_capability=capability_of,
        get_device_name=name_of,
        get_arch_list=lambda: archs,
    )
    mps = types.SimpleNamespace(is_available=lambda: False)
    return types.SimpleNamespace(
        __version__="2.14.0+cu130", cuda=cuda_ns, backends=types.SimpleNamespace(mps=mps)
    )


CU130 = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


def test_an_old_gpu_is_left_aside_with_one_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "torch", _torch((6, 1), CU130))
    resolve_device.cache_clear()
    try:
        with _no_warning():
            assert resolve_device("auto") == "cpu"
            assert resolve_device("auto") == "cpu"
    finally:
        resolve_device.cache_clear()
    messages = [r.getMessage() for r in caplog.records if r.name == "loom_notes"]
    assert messages == [
        "GPU NVIDIA GeForce GTX 1050 Ti (sm_61) non pris en charge par ce torch (2.14.0+cu130, "
        "architectures : sm_75, sm_80, sm_86, sm_90, sm_100, sm_120) : modèles sur cpu "
        "(LOOM_NOTES_DEVICE pour forcer)"
    ]


class _no_warning:
    """Échoue si un avertissement Python sort pendant le bloc."""

    def __enter__(self) -> None:
        import warnings

        self._ctx = warnings.catch_warnings(record=True)
        self._seen = self._ctx.__enter__()
        warnings.simplefilter("always")

    def __exit__(self, *exc: object) -> None:
        self._ctx.__exit__(None, None, None)
        assert not self._seen, [str(w.message) for w in self._seen]


@pytest.mark.parametrize(
    ("capability", "archs", "usable"),
    [
        ((6, 1), CU130, False),  # GTX 1050 Ti face à la roue cu130
        ((7, 5), CU130, True),
        ((7, 0), ["sm_75", "sm_80"], False),  # même majeure, mineure trop basse
        ((8, 9), ["sm_80", "sm_86", "sm_90"], True),  # 8.9 : binaire sm_86 de même majeure
        ((12, 0), ["sm_80", "compute_90"], True),  # PTX compilé à la volée
        ((6, 1), [], True),  # rien à juger : la carte est gardée
    ],
)
def test_gpu_support_follows_the_torch_rule(
    capability: tuple[int, int], archs: list[str], usable: bool
) -> None:
    assert (unsupported_gpu(_torch(capability, archs)) is None) is usable


def test_a_supported_gpu_is_used_and_an_explicit_device_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", _torch((8, 6), CU130))
    resolve_device.cache_clear()
    try:
        assert resolve_device("auto") == "cuda"
        assert resolve_device("cuda:1") == "cuda:1"
    finally:
        resolve_device.cache_clear()
