from pathlib import Path

import pytest

from loom_memory.service import MemoryService, NotFoundError, build_service
from loom_memory.settings import Settings
from loom_memory.store import ModelMismatchError

PLOMBERIE = (
    "# Relance de devis plomberie\n\nLes plombiers perdent des devis faute de relance. "
    "L'agent relance automatiquement à J+3 et J+7 par email puis SMS."
)
FLUTTER = (
    "# Localisation Flutter\n\nLes fichiers ARB portent les traductions. "
    "GoRouter gère la navigation et Riverpod l'état."
)


async def test_add_and_search_by_project(service: MemoryService) -> None:
    r1 = await service.add_text(PLOMBERIE, "Relance devis", "agent-pro", ["vente"])
    r2 = await service.add_text(FLUTTER, "Localisation", "vaultlyra")
    assert r1.chunks >= 1 and r2.chunks >= 1
    assert r1.duplicate_of is None

    hits = await service.search("relance devis plombiers")
    assert hits and hits[0].doc_id == r1.doc_id
    assert hits[0].project == "agent-pro"
    assert hits[0].heading_path.startswith("Relance devis")

    hits = await service.search("relance devis plombiers", project="vaultlyra")
    assert all(h.project == "vaultlyra" for h in hits)

    hits = await service.search("traductions ARB", tags=["vente"])
    assert all(h.doc_id == r1.doc_id for h in hits)

    assert await service.projects() == {"agent-pro": 1, "vaultlyra": 1}


async def test_duplicate_is_detected(service: MemoryService) -> None:
    r1 = await service.add_text(PLOMBERIE, "Relance devis", "agent-pro")
    r2 = await service.add_text(PLOMBERIE.upper(), "Autre titre", "autre")
    assert r2.duplicate_of == r1.doc_id
    assert r2.chunks == 0
    assert await service.projects() == {"agent-pro": 1}


async def test_update_keeps_id_and_reindexes(service: MemoryService) -> None:
    r = await service.add_text(PLOMBERIE, "Relance devis", "agent-pro", ["vente"])
    u = await service.update(r.doc_id, FLUTTER)
    assert u.doc_id == r.doc_id and u.duplicate_of is None
    doc = await service.get(r.doc_id)
    assert doc.text == FLUTTER and doc.tags == ["vente"] and doc.title == "Relance devis"
    hits = await service.search("traductions ARB")
    assert hits and hits[0].doc_id == r.doc_id
    for h in await service.search("plombiers relance"):  # l'ancien contenu a disparu
        assert "plomb" not in h.snippet


async def test_delete(service: MemoryService) -> None:
    r = await service.add_text(PLOMBERIE, "Relance devis", "agent-pro")
    summary = await service.delete(r.doc_id)
    assert summary.doc_id == r.doc_id
    with pytest.raises(NotFoundError):
        await service.get(r.doc_id)
    assert not await service.search("plombiers")
    with pytest.raises(NotFoundError):
        await service.delete(r.doc_id)


async def test_get_unknown_ids(service: MemoryService) -> None:
    with pytest.raises(NotFoundError):
        await service.get("pas-un-uuid")
    with pytest.raises(NotFoundError):
        await service.get("00000000-0000-0000-0000-000000000000")


async def test_validation(service: MemoryService) -> None:
    with pytest.raises(ValueError):
        await service.add_text("x", "ab", "p")
    with pytest.raises(ValueError):
        await service.add_text("texte", "titre", "  ")
    with pytest.raises(ValueError):
        await service.add_text("   ", "titre", "p")


async def test_add_file_and_list(service: MemoryService, tmp_path: Path) -> None:
    p = tmp_path / "note.md"
    p.write_text(FLUTTER, encoding="utf-8")
    r = await service.add_file(p, "VaultLyra")
    assert r.title == "Localisation Flutter" and r.project == "vaultlyra"
    docs = await service.list_docs()
    assert [d.doc_id for d in docs] == [r.doc_id]
    assert docs[0].source_kind == "file" and docs[0].source == str(p.resolve())


async def test_export_import_reindex(service: MemoryService, settings: Settings) -> None:
    r1 = await service.add_text(PLOMBERIE, "Relance devis", "agent-pro")
    r2 = await service.add_text(FLUTTER, "Localisation", "vaultlyra")
    out = settings.data_dir / "export.jsonl"
    assert await service.export(out) == 2

    await service.delete(r1.doc_id)
    imported, skipped = await service.import_(out)
    assert (imported, skipped) == (1, 1)
    assert (await service.get(r1.doc_id)).text == PLOMBERIE
    assert (await service.get(r2.doc_id)).text == FLUTTER

    assert await service.reindex() == 2
    hits = await service.search("relance devis plombiers")
    assert hits and hits[0].doc_id == r1.doc_id


async def test_model_mismatch_is_refused(settings: Settings) -> None:
    svc = build_service(settings, fake=True)
    await svc.start()
    await svc.close()
    settings.meta_path.write_text('{"dense_model": "autre", "dense_dim": 1024}', encoding="utf-8")
    svc = build_service(settings, fake=True)
    with pytest.raises(ModelMismatchError):
        await svc.start()
    await svc.start(check_model=False)  # voie réservée à reindex
    await svc.close()


def test_store_mode_follows_settings(settings: Settings) -> None:
    from loom_memory.store import MemoryStore

    assert MemoryStore.open(settings).is_local
    remote = MemoryStore.open(
        Settings(data_dir=settings.data_dir, qdrant_url="http://127.0.0.1:6333")
    )
    assert not remote.is_local


async def test_add_file_outside_roots_is_refused(service: MemoryService, tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-hors.md"
    outside.write_text("# Hors\n\ncontenu", encoding="utf-8")
    from loom_memory.ingest.paths import PathDeniedError

    with pytest.raises(PathDeniedError):
        await service.add_file(outside, "p")
    assert await service.projects() == {}
