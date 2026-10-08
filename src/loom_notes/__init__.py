"""loom-notes — mémoire locale pour Claude Desktop et les agents IA."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("loom-notes")  # source unique : pyproject.toml
except PackageNotFoundError:  # package non installé (exécution depuis les sources)
    __version__ = "0.0.0"
