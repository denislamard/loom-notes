"""BGE-M3 via FlagEmbedding : dense + sparse en un seul passage, chargé paresseusement."""

import importlib
from collections.abc import Sequence
from typing import Any

from loom_notes.embed.base import Embedding, SparseVec, resolve_device
from loom_notes.settings import Settings


class BgeM3Embedder:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._settings.dense_model

    @property
    def dim(self) -> int:
        return self._settings.dense_dim

    def _load(self) -> Any:
        if self._model is None:
            try:
                flag: Any = importlib.import_module("FlagEmbedding")
            except ImportError as exc:
                raise RuntimeError(
                    'FlagEmbedding absent : pip install "loom-notes[models]"'
                ) from exc
            device = resolve_device(self._settings.device)
            self._model = flag.BGEM3FlagModel(
                self._settings.dense_model,
                use_fp16=self._settings.use_fp16 and device.startswith("cuda"),
                devices=device,
            )
        return self._model

    def _encode(self, texts: Sequence[str]) -> list[Embedding]:
        out: Any = self._load().encode(
            list(texts),
            batch_size=16,
            max_length=1024,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vecs: Any = out["dense_vecs"]
        lexical: Any = out["lexical_weights"]
        result: list[Embedding] = []
        for dense, weights in zip(dense_vecs, lexical, strict=True):
            items = sorted((int(k), float(v)) for k, v in weights.items() if float(v) > 0)
            result.append(
                Embedding(
                    dense=[float(x) for x in dense],
                    sparse=SparseVec(indices=[i for i, _ in items], values=[v for _, v in items]),
                )
            )
        return result

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return self._encode(texts) if texts else []

    def embed_query(self, text: str) -> Embedding:
        return self._encode([text])[0]
