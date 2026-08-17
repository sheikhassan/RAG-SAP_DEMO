"""Qdrant hybrid vector store: HNSW semantic search + BM25 keyword search.

Public API:
    upsert_chunks(chunks)           -> int
    query(text, allowed_roles, ...) -> list[RetrievedChunk]
    delete_by_source_id(source_id)  -> int
    collection_count()              -> int
    reset_collection()              -> None
    store_info()                    -> dict
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .config import (
    DENSE_VECTOR_NAME,
    EMBEDDING_DIM,
    HNSW_EF_CONSTRUCT,
    HNSW_EF_SEARCH,
    HNSW_FULL_SCAN_THRESHOLD,
    HNSW_M,
    HYBRID_PREFETCH_K,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_DIR,
    QDRANT_URL,
    SEARCH_MODE,
    SPARSE_VECTOR_NAME,
)
from .document_loader import Chunk
from .embeddings import embed, embed_one
from .sparse import embed_sparse, embed_sparse_one


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    score: float  # fused / primary score (higher is better)
    dense_score: float | None = None
    sparse_score: float | None = None
    search_mode: str = "hybrid"
    extra: dict = field(default_factory=dict)


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


@lru_cache(maxsize=1)
def _get_client() -> QdrantClient:
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, timeout=30)
    QDRANT_DIR.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_DIR))


def _hnsw_config() -> qm.HnswConfigDiff:
    return qm.HnswConfigDiff(
        m=HNSW_M,
        ef_construct=HNSW_EF_CONSTRUCT,
        full_scan_threshold=HNSW_FULL_SCAN_THRESHOLD,
    )


def _ensure_collection() -> None:
    client = _get_client()
    if client.collection_exists(QDRANT_COLLECTION):
        return
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={
            DENSE_VECTOR_NAME: qm.VectorParams(
                size=EMBEDDING_DIM,
                distance=qm.Distance.COSINE,
                hnsw_config=_hnsw_config(),
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qm.SparseVectorParams(
                index=qm.SparseIndexParams(on_disk=False),
            )
        },
    )
    # Payload indexes only help a Qdrant server; embedded local mode warns if set.
    if QDRANT_URL:
        for field_name in ("role", "system", "source_id"):
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field_name,
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )


def upsert_chunks(chunks: Sequence[Chunk]) -> int:
    """Embed (dense + BM25) and upsert chunks. Returns count upserted."""
    if not chunks:
        return 0
    _ensure_collection()

    documents = [c.text for c in chunks]
    dense_vecs = embed(documents)
    sparse_vecs = embed_sparse(documents)

    points: list[qm.PointStruct] = []
    for chunk, dense, (idx, val) in zip(chunks, dense_vecs, sparse_vecs):
        payload = {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            **{k: v for k, v in chunk.metadata.items() if v is not None},
        }
        points.append(
            qm.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector={
                    DENSE_VECTOR_NAME: dense,
                    SPARSE_VECTOR_NAME: qm.SparseVector(indices=idx, values=val),
                },
                payload=payload,
            )
        )

    _get_client().upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
    return len(points)


def delete_by_source_id(source_id: str) -> int:
    """Remove every chunk whose payload.source_id matches. Returns deleted count."""
    if not source_id:
        return 0
    _ensure_collection()
    client = _get_client()
    flt = qm.Filter(
        must=[qm.FieldCondition(key="source_id", match=qm.MatchValue(value=source_id))]
    )
    counted = client.count(collection_name=QDRANT_COLLECTION, count_filter=flt, exact=True)
    n = int(counted.count)
    if n == 0:
        return 0
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=qm.FilterSelector(filter=flt),
        wait=True,
    )
    return n


def reset_collection() -> None:
    client = _get_client()
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
    _ensure_collection()


def collection_count() -> int:
    _ensure_collection()
    return int(_get_client().count(collection_name=QDRANT_COLLECTION, exact=True).count)


def chunk_count_by_role() -> dict[str, int]:
    """Group counts of stored chunks by their `role` payload."""
    _ensure_collection()
    client = _get_client()
    if collection_count() == 0:
        return {}
    out: dict[str, int] = {}
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            with_payload=["role"],
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        for rec in records:
            payload = rec.payload or {}
            role = str(payload.get("role") or "common")
            out[role] = out.get(role, 0) + 1
        if offset is None:
            break
    return out


def store_info() -> dict:
    """Runtime facts for the Metrics tab / eval CLI."""
    _ensure_collection()
    info = _get_client().get_collection(QDRANT_COLLECTION)
    return {
        "backend": "qdrant",
        "mode": "server" if QDRANT_URL else "embedded",
        "url": QDRANT_URL or str(QDRANT_DIR),
        "collection": QDRANT_COLLECTION,
        "points": int(info.points_count or 0),
        "ann_algorithm": "HNSW",
        "distance": "cosine",
        "hnsw_m": HNSW_M,
        "hnsw_ef_construct": HNSW_EF_CONSTRUCT,
        "hnsw_ef_search": HNSW_EF_SEARCH,
        "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
        "dense_dim": EMBEDDING_DIM,
        "sparse_model": "Qdrant/bm25",
        "fusion": "reciprocal_rank_fusion",
        "search_mode_default": SEARCH_MODE,
        "status": str(info.status),
    }


def _build_filter(allowed_roles: Iterable[str], system: str) -> qm.Filter:
    roles = sorted({(r or "").lower() for r in allowed_roles if r}) or ["common"]
    must = [qm.FieldCondition(key="role", match=qm.MatchAny(any=roles))]
    system = (system or "both").lower()
    should = None
    if system != "both":
        should = [
            qm.FieldCondition(key="system", match=qm.MatchValue(value=system)),
            qm.FieldCondition(key="system", match=qm.MatchValue(value="both")),
        ]
    return qm.Filter(must=must, should=should)


def _points_to_chunks(
    points: Sequence[qm.ScoredPoint],
    search_mode: str,
) -> list[RetrievedChunk]:
    out: list[RetrievedChunk] = []
    for pt in points:
        payload = dict(pt.payload or {})
        text = str(payload.pop("text", "") or "")
        chunk_id = str(payload.pop("chunk_id", pt.id))
        score = float(pt.score or 0.0)
        out.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                metadata=payload,
                score=score,
                search_mode=search_mode,
            )
        )
    return out


def query(
    user_query: str,
    allowed_roles: Iterable[str],
    system: str = "both",
    top_k: int = 5,
    search_mode: str | None = None,
) -> list[RetrievedChunk]:
    """Hybrid / semantic / keyword search restricted to `allowed_roles`."""
    if not user_query or not user_query.strip():
        return []

    _ensure_collection()
    if collection_count() == 0:
        return []

    mode = (search_mode or SEARCH_MODE or "hybrid").strip().lower()
    if mode not in {"hybrid", "semantic", "keyword"}:
        mode = "hybrid"

    flt = _build_filter(allowed_roles, system)
    client = _get_client()
    prefetch_k = max(top_k, HYBRID_PREFETCH_K)
    search_params = qm.SearchParams(hnsw_ef=HNSW_EF_SEARCH, exact=False)

    if mode == "semantic":
        dense = embed_one(user_query)
        result = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=dense,
            using=DENSE_VECTOR_NAME,
            query_filter=flt,
            limit=top_k,
            search_params=search_params,
            with_payload=True,
        )
        return _points_to_chunks(result.points, mode)

    if mode == "keyword":
        idx, val = embed_sparse_one(user_query)
        result = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=qm.SparseVector(indices=idx, values=val),
            using=SPARSE_VECTOR_NAME,
            query_filter=flt,
            limit=top_k,
            with_payload=True,
        )
        return _points_to_chunks(result.points, mode)

    dense = embed_one(user_query)
    idx, val = embed_sparse_one(user_query)
    result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=[
            qm.Prefetch(
                query=dense,
                using=DENSE_VECTOR_NAME,
                filter=flt,
                limit=prefetch_k,
                params=search_params,
            ),
            qm.Prefetch(
                query=qm.SparseVector(indices=idx, values=val),
                using=SPARSE_VECTOR_NAME,
                filter=flt,
                limit=prefetch_k,
            ),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return _points_to_chunks(result.points, "hybrid")
