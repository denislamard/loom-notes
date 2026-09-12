"""Serveur MCP (stdio) exposant la mémoire à Claude Desktop et aux agents LOOM.

Règle absolue : les tools d'écriture ne sont appelés que sur demande explicite de Denis.
Elle est répétée dans les instructions du serveur et dans chaque description de tool.
"""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator, Awaitable
from contextlib import asynccontextmanager
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from loom_memory import __version__
from loom_memory.ingest.extract import ExtractionError
from loom_memory.models import DocSummary, Document, Hit, IngestResult
from loom_memory.service import MemoryService, NotFoundError, build_service
from loom_memory.settings import Settings

log = logging.getLogger("loom_memory")

INSTRUCTIONS = """\
Mémoire personnelle de Denis (loom-memory). Son contenu est choisi et ajouté par Denis lui-même :
projets, produits, décisions, notes techniques, pages web qu'il a jugées utiles.

Lecture : appelle `search` avant de répondre à toute question qui touche aux projets, produits,
décisions passées ou notes de Denis, ou dès qu'il fait référence à quelque chose qu'il t'aurait
"déjà dit". Sans `project`, la recherche est globale et chaque résultat indique son projet.
Utilise `get` quand un extrait ne suffit pas.

Écriture : n'appelle JAMAIS `add_text`, `add_url`, `add_file`, `update` ou `delete` de ta propre
initiative. Uniquement quand Denis le demande explicitement dans son message courant
("ajoute ça à ma mémoire", "mémorise cette page", "supprime ce document"). En cas de doute,
demande-lui. Chaque tool d'écriture renvoie ce qu'il a réellement fait : montre-le à Denis.
"""

_WRITE_GUARD = (
    " N'appeler QUE si Denis le demande explicitement dans son message courant ; "
    "jamais de ta propre initiative."
)


def create_server(settings: Settings | None = None, *, fake: bool | None = None) -> FastMCP:
    settings = settings or Settings()
    holder: dict[str, MemoryService] = {}

    def svc() -> MemoryService:
        s = holder.get("svc")
        if s is None:
            raise ToolError("mémoire non initialisée")
        return s

    @asynccontextmanager
    async def lifespan(_: FastMCP) -> AsyncGenerator[None]:
        service = build_service(settings, fake=fake)
        await service.start()
        holder["svc"] = service
        warm: asyncio.Task[None] | None = None
        if settings.warmup_on_start:
            warm = asyncio.create_task(_warmup(service))
        try:
            yield
        finally:
            if warm is not None and not warm.done():
                warm.cancel()
            holder.pop("svc", None)
            await service.close()

    mcp = FastMCP("loom-memory", instructions=INSTRUCTIONS, version=__version__, lifespan=lifespan)

    # ---------- lecture ----------

    @mcp.tool(annotations={"read_only_hint": True})
    async def search(
        query: Annotated[str, Field(description="Question ou mots-clés, en langage naturel.")],
        project: Annotated[
            str | None, Field(description="Restreindre à un projet ; absent = recherche globale.")
        ] = None,
        tags: Annotated[
            list[str] | None, Field(description="Tous les tags doivent être présents.")
        ] = None,
        k: Annotated[int, Field(ge=1, le=10, description="Nombre de résultats.")] = 5,
    ) -> list[Hit]:
        """Recherche hybride (dense + sparse, reranking) dans la mémoire de Denis. À appeler avant
        de répondre sur ses projets, décisions ou notes. Renvoie des extraits courts avec doc_id,
        titre, projet et score ; utilise `get` pour lire un document en entier."""
        return await _call(svc().search(query, project, tags, k))

    @mcp.tool(annotations={"read_only_hint": True})
    async def get(
        doc_id: Annotated[str, Field(description="Identifiant renvoyé par search.")],
    ) -> Document:
        """Texte intégral d'un document de la mémoire, avec ses métadonnées."""
        return await _call(svc().get(doc_id))

    @mcp.tool(annotations={"read_only_hint": True})
    async def list_docs(
        project: Annotated[str | None, Field(description="Filtrer sur un projet.")] = None,
        n: Annotated[int, Field(ge=1, le=50)] = 20,
    ) -> list[DocSummary]:
        """Derniers documents ajoutés, du plus récent au plus ancien (métadonnées, sans texte)."""
        return await _call(svc().list_docs(project, n))

    @mcp.tool(annotations={"read_only_hint": True})
    async def projects() -> dict[str, int]:
        """Projets présents dans la mémoire et nombre de documents pour chacun. Utile pour choisir
        le filtre `project` de search."""
        return await _call(svc().projects())

    # ---------- écriture : uniquement sur demande explicite ----------

    @mcp.tool(description="Ajoute un texte brut à la mémoire de Denis." + _WRITE_GUARD)
    async def add_text(
        text: Annotated[str, Field(description="Contenu à mémoriser, tel quel.")],
        title: Annotated[
            str,
            Field(min_length=3, description="Titre court et explicite (apparaît dans list_docs)."),
        ],
        project: Annotated[str, Field(description="Projet de rattachement, ex. 'loom'.")],
        tags: Annotated[list[str] | None, Field(description="Tags optionnels.")] = None,
    ) -> IngestResult:
        return await _call(svc().add_text(text, title, project, tags))

    @mcp.tool(
        description="Télécharge une page web, en extrait le contenu principal et l'ajoute à la "
        "mémoire. Le titre est celui de la page." + _WRITE_GUARD
    )
    async def add_url(
        url: Annotated[str, Field(description="URL http(s) de la page.")],
        project: Annotated[str, Field(description="Projet de rattachement.")],
        tags: Annotated[list[str] | None, Field(description="Tags optionnels.")] = None,
    ) -> IngestResult:
        return await _call(svc().add_url(url, project, tags))

    @mcp.tool(
        description="Ajoute un fichier markdown local à la mémoire, découpé par titres. "
        "Le titre est le premier H1, sinon le nom du fichier." + _WRITE_GUARD
    )
    async def add_file(
        path: Annotated[str, Field(description="Chemin absolu du fichier .md sur la machine.")],
        project: Annotated[str, Field(description="Projet de rattachement.")],
        tags: Annotated[list[str] | None, Field(description="Tags optionnels.")] = None,
    ) -> IngestResult:
        return await _call(svc().add_file(path, project, tags))

    @mcp.tool(
        description="Remplace le texte d'un document existant (même doc_id, titre, projet et tags "
        "conservés ; chunks réindexés)." + _WRITE_GUARD,
        annotations={"idempotent_hint": True},
    )
    async def update(
        doc_id: Annotated[str, Field(description="Identifiant du document.")],
        text: Annotated[str, Field(description="Nouveau contenu intégral.")],
    ) -> IngestResult:
        return await _call(svc().update(doc_id, text))

    @mcp.tool(
        description="Supprime définitivement un document et ses chunks. Renvoie le résumé de ce "
        "qui a été supprimé." + _WRITE_GUARD,
        annotations={"destructive_hint": True},
    )
    async def delete(
        doc_id: Annotated[str, Field(description="Identifiant du document.")],
    ) -> DocSummary:
        return await _call(svc().delete(doc_id))

    return mcp


async def _call[T](awaitable: Awaitable[T]) -> T:
    """Convertit les erreurs métier en ToolError lisibles par Claude (sans trace)."""
    try:
        return await awaitable
    except (NotFoundError, ValueError, ExtractionError) as exc:
        raise ToolError(str(exc)) from exc


async def _warmup(service: MemoryService) -> None:
    try:
        await service.warmup()
        log.info("modèles chargés")
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("échec du préchargement des modèles")


def main() -> None:
    # stdio : stdout est réservé au protocole. On coupe les barres de progression et le bruit
    # des bibliothèques de modèles ; les logs partent sur stderr.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    create_server().run(show_banner=False)


if __name__ == "__main__":
    main()
