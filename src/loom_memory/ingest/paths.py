"""Contrôle d'accès des fichiers lus par `add_file`.

Trois contrôles, dans cet ordre, et le message dit toujours lequel a mordu :
1. le chemin résolu (symlinks compris) est inclus dans une racine autorisée ;
2. aucun composant du chemin relatif ne correspond à un motif refusé ;
3. l'extension est celle d'un fichier texte/markdown.

La liste de refus de base est dans le code, volontairement : une garantie qu'un fichier de
configuration peut retirer n'est plus une garantie. La configuration ne peut qu'y ajouter.
"""

import fnmatch
from pathlib import Path

from loom_memory.settings import Settings

BASE_DENY: tuple[str, ...] = (
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx",
    "id_rsa*", "id_ecdsa*", "id_ed25519*",
    ".ssh", ".aws", ".gnupg", ".netrc", ".npmrc", ".pypirc",
    ".git", ".venv", "node_modules", "__pycache__",
)  # fmt: skip

ALLOWED_SUFFIXES: frozenset[str] = frozenset({".md", ".markdown", ".txt"})


class PathDeniedError(Exception):
    pass


def resolve_allowed(path: str | Path, settings: Settings) -> Path:
    """Renvoie le chemin réel si la lecture est autorisée, sinon lève PathDeniedError."""
    if not settings.allowed_roots:
        raise PathDeniedError("add_file désactivé : aucune racine dans LOOM_MEMORY_ALLOWED_ROOTS")
    target = Path(path).expanduser().resolve()
    root = next((r for r in _resolved_roots(settings) if target.is_relative_to(r)), None)
    if root is None:
        raise PathDeniedError(f"chemin hors des racines autorisées : {target}")
    deny = (*BASE_DENY, *settings.deny_patterns)
    for part in target.relative_to(root).parts:
        for pattern in deny:
            if fnmatch.fnmatchcase(part, pattern):
                raise PathDeniedError(f"chemin refusé par le motif '{pattern}' : {target}")
    if target.suffix.lower() not in ALLOWED_SUFFIXES:
        raise PathDeniedError(
            f"extension non autorisée ({target.suffix or 'aucune'}) ; "
            f"attendu : {', '.join(sorted(ALLOWED_SUFFIXES))}"
        )
    return target


def _resolved_roots(settings: Settings) -> list[Path]:
    return [Path(r).expanduser().resolve() for r in settings.allowed_roots]
