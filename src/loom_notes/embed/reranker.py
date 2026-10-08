"""Cross-encoder bge-reranker-v2-m3 via FlagEmbedding, chargé paresseusement."""

import importlib
from collections.abc import Sequence
from typing import Any

from loom_notes.embed.base import resolve_device
from loom_notes.settings import Settings


class BgeReranker:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._settings.reranker_model

    def _load(self) -> Any:
        if self._model is None:
            try:
                flag: Any = importlib.import_module("FlagEmbedding")
            except ImportError as exc:
                raise RuntimeError(
                    'FlagEmbedding absent : pip install "loom-notes[models]"'
                ) from exc
            device = resolve_device(self._settings.device)
            self._model = flag.FlagReranker(
                self._settings.reranker_model,
                use_fp16=self._settings.use_fp16 and device.startswith("cuda"),
                devices=device,
            )
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        scores: Any = self._load().compute_score(
            [[query, p] for p in passages], normalize=True, batch_size=16
        )
        if isinstance(scores, int | float):
            return [float(scores)]
        return [float(s) for s in scores]
