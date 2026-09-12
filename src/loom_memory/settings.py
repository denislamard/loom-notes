"""Configuration du serveur, surchargeable par variables d'environnement LOOM_MEMORY_*."""

import os
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOOM_MEMORY_", env_file=".env", extra="ignore")

    # Stockage. Sans qdrant_url : Qdrant embarqué dans data/qdrant (un seul processus à la fois).
    # Avec : serveur Qdrant (Docker), accès concurrent possible (chat + Cowork + CLI).
    data_dir: Path = Field(default=_REPO_ROOT / "data")
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    chunks_collection: str = "memory"
    documents_collection: str = "documents"

    # Modèles (100 % local)
    dense_model: str = "BAAI/bge-m3"
    dense_dim: int = 1024
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cuda"
    use_fp16: bool = True
    fake_models: bool = False  # modèles factices (tests, démo sans GPU)
    warmup_on_start: bool = True  # charge les modèles en tâche de fond au démarrage du serveur

    # Chunking (en caractères ; ~4 caractères par token en français)
    chunk_target_chars: int = 1800
    chunk_max_chars: int = 2400
    chunk_overlap_chars: int = 180
    chunk_min_chars: int = 200

    # Retrieval
    prefetch_limit: int = 15
    rerank_candidates: int = 20  # le reranker est le poste de coût dominant sur CPU
    max_chunks_per_doc: int = 2
    snippet_chars: int = 300  # longueur de l'extrait quand le chunk est trop long
    full_chunk_chars: int = 1000  # en dessous, search renvoie le chunk entier
    min_score: float = 0.1  # score reranker minimal ; réglé via `loom-memory eval`

    # Fichiers lisibles par add_file. Vide = add_file refusé. En variable d'environnement :
    # chemins séparés par ':' (LOOM_MEMORY_ALLOWED_ROOTS=/home/denis/dev:/home/denis/notes).
    allowed_roots: Annotated[list[Path], NoDecode] = []
    # Motifs refusés en plus de la liste de base (loom_memory.ingest.paths.BASE_DENY).
    deny_patterns: Annotated[list[str], NoDecode] = []

    # Réseau (uniquement pour add_url)
    fetch_timeout_s: float = 20.0
    user_agent: str = "loom-memory/0.1 (+https://github.com/denislamard/loom-memory)"

    @field_validator("allowed_roots", "deny_patterns", mode="before")
    @classmethod
    def _split_list(cls, v: object) -> object:
        if isinstance(v, str):
            return [x for x in v.split(os.pathsep) if x.strip()]
        return v

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def meta_path(self) -> Path:
        return self.data_dir / "meta.json"

    @property
    def export_path(self) -> Path:
        return self.data_dir / "export.jsonl"

    @property
    def golden_path(self) -> Path:
        return self.data_dir / "golden.jsonl"
