"""Configuration du serveur, surchargeable par variables d'environnement LOOM_NOTES_*."""

import os
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from loom_notes import __version__


def _default_data_dir() -> Path:
    """$XDG_DATA_HOME/loom-notes, sinon ~/.local/share/loom-notes (spécification XDG)."""
    xdg = os.environ.get("XDG_DATA_HOME", "")
    base = Path(xdg) if xdg and Path(xdg).is_absolute() else Path.home() / ".local" / "share"
    return base / "loom-notes"


class Settings(BaseSettings):
    # Variables d'environnement seulement : aucun .env n'est lu, quel que soit le dossier courant.
    model_config = SettingsConfigDict(env_prefix="LOOM_NOTES_", extra="ignore")

    # Stockage. Sans qdrant_url : Qdrant embarqué dans data_dir/qdrant (un seul processus à la
    # fois). Avec : serveur Qdrant (Docker), accès concurrent possible (chat + Cowork + CLI).
    data_dir: Path = Field(default_factory=_default_data_dir)
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    chunks_collection: str = "memory"
    documents_collection: str = "documents"

    # Modèles (100 % local)
    dense_model: str = "BAAI/bge-m3"
    dense_dim: int = 1024
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "auto"  # auto : cuda, sinon mps, sinon cpu ; ou une valeur torch explicite
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
    min_score: float = 0.1  # score reranker minimal ; réglé via `loom-notes eval`

    # Fichiers lisibles par add_file. Vide = add_file refusé. En variable d'environnement :
    # chemins séparés par ':' (LOOM_NOTES_ALLOWED_ROOTS=~/dev:~/notes).
    allowed_roots: Annotated[list[Path], NoDecode] = []
    # Motifs refusés en plus de la liste de base (loom_notes.ingest.paths.BASE_DENY).
    deny_patterns: Annotated[list[str], NoDecode] = []

    # Réseau (uniquement pour add_url)
    fetch_timeout_s: float = 20.0
    fetch_max_bytes: int = 5_000_000  # taille maximale d'une page, après décompression
    fetch_allow_private: bool = False  # True : autorise les adresses locales et privées
    user_agent: str = f"loom-notes/{__version__} (+https://github.com/denislamard/loom-notes)"

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
