from pathlib import Path

from loom_memory.evals import GoldenCase, append_golden, load_golden, run_eval
from loom_memory.service import MemoryService

DOCS = {
    "qdrant": (
        "Lancer Qdrant",
        "docker run qdrant avec le stockage dans le dépôt et restart unless-stopped",
    ),
    "roles": ("Rôles loom-fs", "reference sourcecode workspace sont les rôles plafonds de loom-fs"),
    "arb": ("Localisation", "les fichiers ARB portent les traductions flutter de vaultlyra"),
}


async def test_eval_metrics_and_threshold(service: MemoryService, tmp_path: Path) -> None:
    ids = {k: (await service.add_text(t, title, "p")).doc_id for k, (title, t) in DOCS.items()}
    golden = tmp_path / "golden.jsonl"
    append_golden(golden, GoldenCase(query="docker qdrant restart", doc_id=ids["qdrant"]))
    append_golden(golden, GoldenCase(query="rôles plafonds loom-fs", doc_id=ids["roles"]))
    append_golden(golden, GoldenCase(query="traductions ARB flutter", doc_id=ids["arb"], title="x"))
    append_golden(golden, GoldenCase(query="sujet inconnu", doc_id="00000000-0000-0000-0000-0"))
    cases = load_golden(golden)
    assert len(cases) == 4 and cases[2].title == "x"

    report = await run_eval(service, cases, k=10)
    assert report.cases == 4
    assert [r.rank for r in report.results[:3]] == [1, 1, 1]
    assert report.results[3].rank is None and report.results[3].expected_score is None
    assert report.recall_at_1 == report.recall_at_5 == 0.75
    assert abs(report.mrr - 0.75) < 1e-9
    assert report.suggested_min_score is not None and 0 < report.suggested_min_score <= 1
    assert report.noise_removed_at_suggested is not None


def test_load_missing_golden_is_empty(tmp_path: Path) -> None:
    assert load_golden(tmp_path / "absent.jsonl") == []
