"""BM25 sparse (keyword) embeddings for hybrid search.

Uses FastEmbed's Qdrant/bm25 model so each chunk and query becomes a
sparse bag-of-words vector that Qdrant can index alongside dense HNSW.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from .config import SPARSE_MODEL_NAME


@lru_cache(maxsize=1)
def _get_model():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def embed_sparse(texts: Sequence[str]) -> list[tuple[list[int], list[float]]]:
    """Return (indices, values) sparse vectors for each input text."""
    if not texts:
        return []
    model = _get_model()
    out: list[tuple[list[int], list[float]]] = []
    for emb in model.embed(list(texts)):
        indices = emb.indices.tolist()
        values = emb.values.tolist()
        # Qdrant rejects empty sparse vectors; keep a zero dummy.
        if not indices:
            indices, values = [0], [0.0]
        out.append((indices, values))
    return out


def embed_sparse_one(text: str) -> tuple[list[int], list[float]]:
    return embed_sparse([text])[0]
