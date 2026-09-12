"""Identifiants : doc_id aléatoire, point_id déterministe (réécriture sans orphelins)."""

import uuid

_NAMESPACE = uuid.UUID("6f1c3a2e-9b7d-4e5a-8c0f-2d1e4b6a7c93")


def new_doc_id() -> str:
    return str(uuid.uuid4())


def point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{doc_id}:{chunk_index}"))
