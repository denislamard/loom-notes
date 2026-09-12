from pathlib import Path
from typing import Any

from fastmcp import Client

from loom_memory.server import create_server
from loom_memory.settings import Settings

TEXT = "Les plombiers perdent des devis faute de relance. L'agent relance à J+3 et J+7."


async def test_tools_are_exposed_with_guards(settings: Settings) -> None:
    async with Client(create_server(settings, fake=True)) as c:
        tools = {t.name: t for t in await c.list_tools()}
        assert set(tools) == {
            "search",
            "get",
            "list_docs",
            "projects",
            "add_text",
            "add_url",
            "add_file",
            "update",
            "delete",
        }
        for name in ("search", "get", "list_docs", "projects"):
            ann = tools[name].annotations
            assert ann is not None and ann.read_only_hint
        for name in ("add_text", "add_url", "add_file", "update", "delete"):
            assert "explicitement" in (tools[name].description or "")
        ann = tools["delete"].annotations
        assert ann is not None and ann.destructive_hint


async def test_round_trip(settings: Settings, tmp_path: Path) -> None:
    async with Client(create_server(settings, fake=True)) as c:
        added: dict[str, Any] = (
            await c.call_tool(
                "add_text", {"text": TEXT, "title": "Relance devis", "project": "agent-pro"}
            )
        ).structured_content or {}
        assert added["chunks"] == 1 and added["duplicate_of"] is None
        doc_id = added["doc_id"]

        note = tmp_path / "note.md"
        note.write_text("# Localisation Flutter\n\nLes fichiers ARB portent les traductions.")
        added_file = (
            await c.call_tool("add_file", {"path": str(note), "project": "vaultlyra"})
        ).structured_content or {}
        assert added_file["title"] == "Localisation Flutter"

        res = (await c.call_tool("search", {"query": "relance plombiers"})).structured_content
        hits: list[dict[str, Any]] = (res or {})["result"]
        assert hits[0]["doc_id"] == doc_id and hits[0]["project"] == "agent-pro"
        assert len(hits[0]["snippet"]) <= 301

        res = (
            await c.call_tool("search", {"query": "relance plombiers", "project": "vaultlyra"})
        ).structured_content
        assert all(h["project"] == "vaultlyra" for h in (res or {})["result"])

        got = (await c.call_tool("get", {"doc_id": doc_id})).structured_content or {}
        assert got["text"] == TEXT

        assert (await c.call_tool("projects", {})).structured_content == {
            "agent-pro": 1,
            "vaultlyra": 1,
        }

        listed = (await c.call_tool("list_docs", {"n": 1})).structured_content or {}
        assert listed["result"][0]["title"] == "Localisation Flutter"

        updated = (
            await c.call_tool("update", {"doc_id": doc_id, "text": "Nouveau contenu de relance."})
        ).structured_content or {}
        assert updated["doc_id"] == doc_id

        deleted = (await c.call_tool("delete", {"doc_id": doc_id})).structured_content or {}
        assert deleted["doc_id"] == doc_id
        assert (await c.call_tool("projects", {})).structured_content == {"vaultlyra": 1}


async def test_business_errors_are_tool_errors(settings: Settings) -> None:
    async with Client(create_server(settings, fake=True)) as c:
        r = await c.call_tool("get", {"doc_id": "inconnu"}, raise_on_error=False)
        assert r.is_error and "document inconnu" in r.content[0].text  # type: ignore[union-attr]

        r = await c.call_tool(
            "add_text", {"text": "   ", "title": "Titre", "project": "p"}, raise_on_error=False
        )
        assert r.is_error and "contenu vide" in r.content[0].text  # type: ignore[union-attr]

        r = await c.call_tool(
            "add_file", {"path": "/nulle/part.md", "project": "p"}, raise_on_error=False
        )
        assert r.is_error and "hors des racines" in r.content[0].text  # type: ignore[union-attr]

        r = await c.call_tool("search", {"query": "x", "k": 99}, raise_on_error=False)
        assert r.is_error
