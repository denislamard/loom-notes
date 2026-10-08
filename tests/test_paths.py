import os
from pathlib import Path

import pytest

from loom_notes.ingest.paths import PathDeniedError, resolve_allowed
from loom_notes.settings import Settings


def test_inside_root_is_resolved(settings: Settings, tmp_path: Path) -> None:
    f = tmp_path / "notes" / "a.md"
    f.parent.mkdir()
    f.write_text("x")
    assert resolve_allowed(f, settings) == f.resolve()
    assert resolve_allowed(str(tmp_path / "notes" / ".." / "notes" / "a.md"), settings) == f


def test_no_roots_means_disabled(tmp_path: Path) -> None:
    with pytest.raises(PathDeniedError, match="désactivé"):
        resolve_allowed(tmp_path / "a.md", Settings(data_dir=tmp_path, allowed_roots=[]))


def test_outside_root_is_denied(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(PathDeniedError, match="hors des racines"):
        resolve_allowed(tmp_path.parent / "ailleurs.md", settings)


def test_symlink_escape_is_denied(settings: Settings, tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.md"
    outside.write_text("secret")
    link = tmp_path / "lien.md"
    link.symlink_to(outside)
    with pytest.raises(PathDeniedError, match="hors des racines"):
        resolve_allowed(link, settings)


@pytest.mark.parametrize("rel", [".env", ".env.local", "keys/id_rsa.md", ".git/notes.md", "x.pem"])
def test_deny_patterns(settings: Settings, tmp_path: Path, rel: str) -> None:
    with pytest.raises(PathDeniedError, match="refusé par le motif"):
        resolve_allowed(tmp_path / rel, settings)


def test_extra_deny_from_settings(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path, allowed_roots=[tmp_path], deny_patterns=["*.private.md"])
    with pytest.raises(PathDeniedError, match=r"\*\.private\.md"):
        resolve_allowed(tmp_path / "journal.private.md", s)


def test_only_text_suffixes(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(PathDeniedError, match="extension"):
        resolve_allowed(tmp_path / "dump.sqlite3", settings)
    with pytest.raises(PathDeniedError, match="extension"):
        resolve_allowed(tmp_path / "script.py", settings)


def test_roots_from_env_are_colon_separated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("LOOM_NOTES_ALLOWED_ROOTS", f"{a}{os.pathsep}{b}")
    monkeypatch.setenv("LOOM_NOTES_DENY_PATTERNS", "*.bak:*.old")
    s = Settings(data_dir=tmp_path)
    assert s.allowed_roots == [a, b]
    assert s.deny_patterns == ["*.bak", "*.old"]
