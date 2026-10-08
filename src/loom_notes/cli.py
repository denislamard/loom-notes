"""CLI d'administration : ingestion, recherche, sauvegarde, réindexation."""

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel

from loom_notes.ingest.extract import ExtractionError
from loom_notes.ingest.paths import PathDeniedError
from loom_notes.service import MemoryService, NotFoundError, build_service
from loom_notes.settings import Settings
from loom_notes.store import ModelMismatchError, StoreUnavailableError

app = typer.Typer(no_args_is_help=True, add_completion=False, help="loom-notes — mémoire locale.")

_fake = False
_data_dir: Path | None = None


@app.callback()
def _root(
    fake: Annotated[
        bool, typer.Option("--fake", help="Modèles factices (tests, sans GPU).")
    ] = False,
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Répertoire des données.")
    ] = None,
) -> None:
    global _fake, _data_dir
    _fake, _data_dir = fake, data_dir
    os.environ.setdefault("TQDM_DISABLE", "1")  # barres de progression de FlagEmbedding
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def _settings() -> Settings:
    return Settings(data_dir=_data_dir) if _data_dir else Settings()


def _run[T](fn: Callable[[MemoryService], Awaitable[T]], *, check_model: bool = True) -> T:
    async def go() -> T:
        svc = build_service(_settings(), fake=True if _fake else None)
        try:
            await svc.start(check_model=check_model)
            return await fn(svc)
        finally:
            await svc.close()

    try:
        return asyncio.run(go())
    except (
        NotFoundError,
        ValueError,
        ModelMismatchError,
        StoreUnavailableError,
        ExtractionError,
        PathDeniedError,
    ) as exc:
        typer.secho(f"erreur : {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc


def _dump(obj: BaseModel | list[BaseModel] | dict[str, int]) -> None:
    if isinstance(obj, BaseModel):
        typer.echo(obj.model_dump_json(indent=2))
    elif isinstance(obj, dict):
        typer.echo(json.dumps(obj, indent=2, ensure_ascii=False))
    else:
        typer.echo(json.dumps([o.model_dump() for o in obj], indent=2, ensure_ascii=False))


Tags = Annotated[list[str] | None, typer.Option("--tag", "-t", help="Tag (répétable).")]


@app.command("add-text")
def add_text(
    project: str,
    title: str,
    text: Annotated[str | None, typer.Argument(help="Texte ; lu sur stdin si absent.")] = None,
    tags: Tags = None,
) -> None:
    """Ajoute un texte brut."""
    content = text if text is not None else typer.get_text_stream("stdin").read()
    _dump(_run(lambda s: s.add_text(content, title, project, tags)))


@app.command("add-url")
def add_url(project: str, url: str, tags: Tags = None) -> None:
    """Télécharge une page web et l'ajoute."""
    _dump(_run(lambda s: s.add_url(url, project, tags)))


@app.command("add-file")
def add_file(project: str, path: Path, tags: Tags = None) -> None:
    """Ajoute un fichier markdown local."""
    _dump(_run(lambda s: s.add_file(path, project, tags)))


@app.command()
def search(
    query: str,
    project: Annotated[str | None, typer.Option("--project", "-p")] = None,
    tags: Tags = None,
    k: Annotated[int, typer.Option("--k", "-k", min=1, max=10)] = 5,
) -> None:
    """Recherche hybride + reranking."""
    hits = _run(lambda s: s.search(query, project, tags, k))
    if not hits:
        typer.echo("aucun résultat")
        return
    for h in hits:
        typer.secho(f"{h.score:.3f}  {h.title}  [{h.project}]  {h.doc_id}", bold=True)
        typer.echo(f"       {h.heading_path}{'  (extrait)' if h.truncated else ''}")
        typer.echo("       " + h.snippet.replace("\n", "\n       ") + "\n")


@app.command()
def get(doc_id: str) -> None:
    """Affiche un document complet."""
    _dump(_run(lambda s: s.get(doc_id)))


@app.command("list")
def list_docs(
    project: Annotated[str | None, typer.Option("--project", "-p")] = None,
    n: Annotated[int, typer.Option("--n", "-n", min=1, max=500)] = 20,
) -> None:
    """Derniers documents ajoutés."""
    for d in _run(lambda s: s.list_docs(project, n)):
        when = (d.updated_at or d.added_at)[:19]
        typer.echo(f"{when}  {d.project:<12}  {d.title}  ({d.chars} car.)  {d.doc_id}")


@app.command()
def projects() -> None:
    """Projets et nombre de documents."""
    _dump(_run(lambda s: s.projects()))


@app.command()
def delete(doc_id: str) -> None:
    """Supprime un document et ses chunks."""
    _dump(_run(lambda s: s.delete(doc_id)))


@app.command()
def export(path: Annotated[Path | None, typer.Argument()] = None) -> None:
    """Exporte tous les documents en JSONL (sauvegarde de référence)."""
    target = path or _settings().export_path
    n = _run(lambda s: s.export(target))
    typer.echo(f"{n} document(s) exporté(s) vers {target}")


@app.command("import")
def import_(path: Annotated[Path | None, typer.Argument()] = None) -> None:
    """Réimporte un export JSONL (les doc_id existants sont ignorés)."""
    source = path or _settings().export_path
    imported, skipped = _run(lambda s: s.import_(source))
    typer.echo(f"{imported} importé(s), {skipped} ignoré(s)")


@app.command("eval")
def eval_(
    path: Annotated[
        Path | None, typer.Argument(help="Jeu doré JSONL ; défaut data/golden.jsonl")
    ] = None,
    k: Annotated[int, typer.Option("--k", min=1, max=10)] = 10,
    as_json: Annotated[bool, typer.Option("--json", help="Rapport complet en JSON.")] = False,
) -> None:
    """Évalue le retrieval sur le jeu doré : recall@1, recall@5, MRR, seuil suggéré."""
    from loom_notes.evals import load_golden, run_eval

    golden = path or _settings().golden_path
    cases = load_golden(golden)
    if not cases:
        typer.secho(f"aucun cas dans {golden} — ajoute-en avec `loom-notes eval-add`", err=True)
        raise typer.Exit(1)
    report = _run(lambda s: run_eval(s, cases, k))
    if as_json:
        typer.echo(report.model_dump_json(indent=2))
        return
    for r in report.results:
        mark = "ok " if r.rank == 1 else ("~  " if r.rank else "KO ")
        rank = f"#{r.rank}" if r.rank else "absent"
        score = f"{r.expected_score:.3f}" if r.expected_score is not None else "  -  "
        typer.echo(f"{mark} {rank:>7}  {score}  {r.query}")
    typer.echo(
        f"\n{report.cases} cas — recall@1 {report.recall_at_1:.2f}  "
        f"recall@5 {report.recall_at_5:.2f}  MRR {report.mrr:.2f}"
    )
    if report.suggested_min_score is not None:
        noise = report.noise_removed_at_suggested
        typer.echo(
            f"min_score suggéré : {report.suggested_min_score} "
            f"(aucun cas perdu ; {noise:.0%} du bruit filtré)"
            if noise is not None
            else f"min_score suggéré : {report.suggested_min_score}"
        )


@app.command("eval-add")
def eval_add(
    query: str,
    doc_id: str,
    project: Annotated[str | None, typer.Option("--project", "-p")] = None,
    path: Annotated[Path | None, typer.Option("--path")] = None,
) -> None:
    """Ajoute un cas au jeu doré (le titre est relu depuis la base pour lisibilité)."""
    from loom_notes.evals import GoldenCase, append_golden

    doc = _run(lambda s: s.get(doc_id))
    case = GoldenCase(query=query, doc_id=doc.doc_id, title=doc.title, project=project)
    append_golden(path or _settings().golden_path, case)
    typer.echo(f"ajouté : « {query} » → {doc.title}")


@app.command()
def reindex() -> None:
    """Reconstruit les chunks depuis les documents avec le modèle courant."""
    n = _run(lambda s: s.reindex(), check_model=False)
    typer.echo(f"{n} document(s) réindexé(s)")


if __name__ == "__main__":
    app()
