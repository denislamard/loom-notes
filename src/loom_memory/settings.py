"""Configuration du serveur, surchargeable par variables d'environnement LOOM_MEMORY_*."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOOM_MEMORY_", env_file=".env", extra="ignore")

    # Stockage
    data_dir: Path = Field(default=_REPO_ROOT / "data")
    chunks_collection: str = "memory"
    documents_collection: str = "documents"

    # Modèles (100 % local)
    dense_model: str = "BAAI/bge-m3"
    dense_dim: int = 1024
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cuda"
    use_fp16: bool = True

    # Chunking (en caractères ; ~4 caractères par token en français)
    chunk_target_chars: int = 1800
    chunk_max_chars: int = 2400
    chunk_overlap_chars: int = 180
    chunk_min_chars: int = 200

    # Retrieval
    prefetch_limit: int = 20
    rerank_candidates: int = 30
    max_chunks_per_doc: int = 2
    snippet_chars: int = 300
    min_score: float = 0.0  # score reranker minimal ; 0 = pas de filtrage

    # Réseau (uniquement pour add_url)
    fetch_timeout_s: float = 20.0
    user_agent: str = "loom-memory/0.1 (+https://github.com/denislamard/loom-memory)"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def meta_path(self) -> Path:
        return self.data_dir / "meta.json"

    @property
    def export_path(self) -> Path:
        return self.data_dir / "export.jsonl"
