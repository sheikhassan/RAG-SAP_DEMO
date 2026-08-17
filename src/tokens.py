"""Token estimation and context-window packing for grounded prompts.

We estimate tokens with a conservative English heuristic (~4 chars / token)
so we never depend on a model-specific tokenizer at query time. The packer
keeps the highest-ranked chunks that still fit the LLM context budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import (
    GENERATION_RESERVE_TOKENS,
    LLM_CONTEXT_WINDOW,
    MAX_CONTEXT_TOKENS,
    SYSTEM_RESERVE_TOKENS,
)
from .vector_store import RetrievedChunk


def estimate_tokens(text: str) -> int:
    """Conservative token estimate (chars/4, minimum 1 for non-empty)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


@dataclass
class ContextBudget:
    window: int
    reserved_system: int
    reserved_generation: int
    query_tokens: int
    available: int
    used: int
    utilization: float
    chunks_kept: int
    chunks_dropped: int


def available_context_tokens(user_query: str, system_prompt: str = "") -> int:
    """How many tokens we may spend on retrieved CONTEXT blocks."""
    query_tokens = estimate_tokens(user_query)
    system_tokens = estimate_tokens(system_prompt) or SYSTEM_RESERVE_TOKENS
    hard_cap = max(256, LLM_CONTEXT_WINDOW - system_tokens - query_tokens - GENERATION_RESERVE_TOKENS)
    return max(256, min(MAX_CONTEXT_TOKENS, hard_cap))


def pack_chunks(
    chunks: Sequence[RetrievedChunk],
    user_query: str,
    system_prompt: str = "",
) -> tuple[list[RetrievedChunk], ContextBudget]:
    """Keep rank-order chunks that fit the context budget; drop the rest."""
    budget = available_context_tokens(user_query, system_prompt)
    kept: list[RetrievedChunk] = []
    used = 0
    for chunk in chunks:
        cost = estimate_tokens(chunk.text) + 40  # header / source_id overhead
        if kept and used + cost > budget:
            break
        if not kept and cost > budget:
            truncated = chunk.text[: max(200, budget * 4)]
            slim = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=truncated,
                metadata=dict(chunk.metadata),
                score=chunk.score,
                dense_score=chunk.dense_score,
                sparse_score=chunk.sparse_score,
                search_mode=chunk.search_mode,
            )
            kept.append(slim)
            used = estimate_tokens(slim.text) + 40
            break
        kept.append(chunk)
        used += cost

    query_tokens = estimate_tokens(user_query)
    return kept, ContextBudget(
        window=LLM_CONTEXT_WINDOW,
        reserved_system=SYSTEM_RESERVE_TOKENS,
        reserved_generation=GENERATION_RESERVE_TOKENS,
        query_tokens=query_tokens,
        available=budget,
        used=used,
        utilization=round(used / budget, 4) if budget else 0.0,
        chunks_kept=len(kept),
        chunks_dropped=max(0, len(chunks) - len(kept)),
    )
