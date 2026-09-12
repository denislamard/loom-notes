"""Évaluation du retrieval sur un jeu doré : requête → document attendu.

Le jeu est un JSONL (`data/golden.jsonl`), une ligne par cas. Les métriques sont celles qu'on
compare d'une version à l'autre : recall@1, recall@5, MRR. L'analyse de seuil sert à régler
`min_score` sans perdre de rappel.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from loom_memory.models import Hit
from loom_memory.service import MemoryService


class GoldenCase(BaseModel):
    query: str
    doc_id: str
    title: str | None = None  # informatif, pour relire le fichier
    project: str | None = None  # filtre appliqué à la recherche, comme le ferait Claude


class CaseResult(BaseModel):
    query: str
    doc_id: str
    rank: int | None = Field(description="Rang du document attendu (1 = premier), None si absent")
    expected_score: float | None
    top_doc_id: str | None
    top_score: float | None


class EvalReport(BaseModel):
    cases: int
    recall_at_1: float
    recall_at_5: float
    mrr: float
    suggested_min_score: float | None = Field(
        description="Plus haut seuil qui ne fait perdre aucun cas trouvé (score min des bons)"
    )
    noise_removed_at_suggested: float | None = Field(
        description="Part des résultats hors document attendu qui tomberaient sous ce seuil"
    )
    results: list[CaseResult]


def load_golden(path: Path) -> list[GoldenCase]:
    if not path.is_file():
        return []
    cases: list[GoldenCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(GoldenCase.model_validate(json.loads(line)))
    return cases


def append_golden(path: Path, case: GoldenCase) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(case.model_dump_json(exclude_none=True) + "\n")


async def run_eval(service: MemoryService, cases: list[GoldenCase], k: int = 10) -> EvalReport:
    results: list[CaseResult] = []
    positives: list[float] = []
    negatives: list[float] = []
    for case in cases:
        hits = await service.search(case.query, project=case.project, k=k, min_score=0.0)
        docs = _doc_order(hits)
        rank = docs.index(case.doc_id) + 1 if case.doc_id in docs else None
        expected = next((h.score for h in hits if h.doc_id == case.doc_id), None)
        if expected is not None:
            positives.append(expected)
        negatives.extend(h.score for h in hits if h.doc_id != case.doc_id)
        results.append(
            CaseResult(
                query=case.query,
                doc_id=case.doc_id,
                rank=rank,
                expected_score=expected,
                top_doc_id=hits[0].doc_id if hits else None,
                top_score=hits[0].score if hits else None,
            )
        )
    n = len(results)
    found = [r.rank for r in results if r.rank is not None]
    suggested = round(min(positives), 3) if positives else None
    noise = (
        sum(1 for s in negatives if s < suggested) / len(negatives)
        if suggested is not None and negatives
        else None
    )
    return EvalReport(
        cases=n,
        recall_at_1=(sum(1 for r in found if r == 1) / n) if n else 0.0,
        recall_at_5=(sum(1 for r in found if r <= 5) / n) if n else 0.0,
        mrr=(sum(1 / r for r in found) / n) if n else 0.0,
        suggested_min_score=suggested,
        noise_removed_at_suggested=round(noise, 3) if noise is not None else None,
        results=results,
    )


def _doc_order(hits: list[Hit]) -> list[str]:
    seen: dict[str, None] = {}
    for h in hits:
        seen.setdefault(h.doc_id, None)
    return list(seen)
