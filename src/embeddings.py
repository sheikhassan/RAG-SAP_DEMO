"""SentenceTransformer-based embedding wrapper, cached as a singleton."""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed(texts: Sequence[str]) -> list[list[float]]:
    """Return float embeddings for each input text (L2-normalized for cosine)."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
