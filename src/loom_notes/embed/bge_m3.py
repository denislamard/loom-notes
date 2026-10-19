"""BGE-M3 via FlagEmbedding : dense + sparse en un seul passage, chargé paresseusement."""

from collections.abc import Sequence
from typing import Any

from loom_notes.embed.base import Embedding, LazyModel, SparseVec, resolve_device
from loom_notes.settings import Settings


class BgeM3Embedder:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = LazyModel[Any](
            settings.dense_model,
            lambda: resolve_device(settings.device),
            self._build,
            lambda model: self._run(model, ["warmup"]),
        )

    @property
    def model_name(self) -> str:
        return self._settings.dense_model

    @property
    def dim(self) -> int:
        return self._settings.dense_dim

    def _build(self, flag: Any, device: str) -> Any:
        return flag.BGEM3FlagModel(
            self._settings.dense_model,
            use_fp16=self._settings.use_fp16 and device.startswith("cuda"),
            devices=device,
        )

    @staticmethod
    def _run(model: Any, texts: list[str]) -> Any:
        return model.encode(
            texts,
            batch_size=16,
            max_length=1024,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    def _encode(self, texts: Sequence[str]) -> list[Embedding]:
        batch = list(texts)
        out: Any = self._model.run(lambda model: self._run(model, batch))
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
