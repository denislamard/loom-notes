"""Cross-encoder bge-reranker-v2-m3 via FlagEmbedding, chargé paresseusement."""

from collections.abc import Sequence
from typing import Any

from loom_notes.embed.base import LazyModel, resolve_device
from loom_notes.settings import Settings


class BgeReranker:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = LazyModel[Any](
            settings.reranker_model,
            lambda: resolve_device(settings.device),
            self._build,
            lambda model: self._run(model, [["warmup", "warmup"]]),
        )

    @property
    def model_name(self) -> str:
        return self._settings.reranker_model

    def _build(self, flag: Any, device: str) -> Any:
        return flag.FlagReranker(
            self._settings.reranker_model,
            use_fp16=self._settings.use_fp16 and device.startswith("cuda"),
            devices=device,
        )

    @staticmethod
    def _run(model: Any, pairs: list[list[str]]) -> Any:
        return model.compute_score(pairs, normalize=True, batch_size=16)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        scores: Any = self._model.run(lambda model: self._run(model, pairs))
        if isinstance(scores, int | float):
            return [float(scores)]
        return [float(s) for s in scores]
