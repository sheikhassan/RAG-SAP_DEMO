"""Retrieval and query-time metrics for the production RAG stack.

Per-query: latency breakdown, context-window utilization, hit scores.
Offline eval: Hit Rate, Precision@k, Recall@k, MRR, nDCG@k across
hybrid / semantic / keyword so we can prove hybrid search earns its keep.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

from .config import (
    HNSW_EF_SEARCH,
    HNSW_M,
    LLM_CONTEXT_WINDOW,
    TOP_K,
)
from .eval_set import GOLDEN_CASES
from .tokens import ContextBudget
from .vector_store import RetrievedChunk, query as store_query, store_info


@dataclass
class QueryMetrics:
    search_mode: str
    ann_algorithm: str
    hnsw_m: int
    hnsw_ef_search: int
    top_k: int
    n_retrieved: int
    n_after_pack: int
    max_score: float
    mean_score: float
    embed_retrieve_ms: float
    generate_ms: float
    total_ms: float
    context_window: int
    context_tokens_used: int
    context_tokens_available: int
    context_utilization: float
    chunks_dropped: int
    source_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


_HISTORY: deque[QueryMetrics] = deque(maxlen=50)


def record(metrics: QueryMetrics) -> QueryMetrics:
    _HISTORY.append(metrics)
    return metrics


def recent_metrics(n: int = 10) -> list[QueryMetrics]:
    items = list(_HISTORY)
    return items[-n:]


def build_query_metrics(
    *,
    search_mode: str,
    chunks: Sequence[RetrievedChunk],
    packed: Sequence[RetrievedChunk],
    budget: ContextBudget,
    retrieve_ms: float,
    generate_ms: float,
    total_ms: float,
) -> QueryMetrics:
    scores = [c.score for c in chunks]
    return QueryMetrics(
        search_mode=search_mode,
        ann_algorithm="HNSW",
        hnsw_m=HNSW_M,
        hnsw_ef_search=HNSW_EF_SEARCH,
        top_k=len(chunks) or TOP_K,
        n_retrieved=len(chunks),
        n_after_pack=len(packed),
        max_score=round(max(scores), 4) if scores else 0.0,
        mean_score=round(sum(scores) / len(scores), 4) if scores else 0.0,
        embed_retrieve_ms=round(retrieve_ms, 2),
        generate_ms=round(generate_ms, 2),
        total_ms=round(total_ms, 2),
        context_window=budget.window or LLM_CONTEXT_WINDOW,
        context_tokens_used=budget.used,
        context_tokens_available=budget.available,
        context_utilization=budget.utilization,
        chunks_dropped=budget.chunks_dropped,
        source_ids=[str(c.metadata.get("source_id", "")) for c in packed],
    )


def _unique(ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in ids:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _hit_rate(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    return 1.0 if set(retrieved) & set(relevant) else 0.0


def _precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    top = _unique(retrieved)[:k]
    if not top:
        return 0.0
    return len(set(top) & set(relevant)) / len(top)


def _recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def _mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    relevant_set = set(relevant)
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            return 1.0 / i
    return 0.0


def _dcg(relevances: Sequence[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def _ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    # Document-level: collapse duplicate chunk source_ids so nDCG stays in [0, 1].
    uniq = _unique(retrieved)[:k]
    rels = [1.0 if doc in set(relevant) else 0.0 for doc in uniq]
    ideal = [1.0] * min(k, len(relevant))
    idcg = _dcg(ideal)
    if not idcg:
        return 0.0
    return min(1.0, _dcg(rels) / idcg)


def evaluate_retrieval(
    search_mode: str = "hybrid",
    top_k: int = TOP_K,
    cases: Iterable[dict] | None = None,
) -> dict:
    """Run the golden set and return per-case + aggregate retrieval metrics."""
    rows: list[dict] = []
    started = time.perf_counter()
    for case in cases or GOLDEN_CASES:
        if search_mode == "llama-index":
            try:
                from .llama_rag import retrieve_nodes_llama
                l_hits = retrieve_nodes_llama(
                    case["query"],
                    allowed_roles=case["allowed_roles"],
                    system=case.get("system", "both"),
                    top_k=top_k,
                )
                retrieved_ids = [str(h.metadata.get("source_id", "")) for h in l_hits]
            except Exception:
                retrieved_ids = []
        else:
            hits = store_query(
                case["query"],
                allowed_roles=case["allowed_roles"],
                system=case.get("system", "both"),
                top_k=top_k,
                search_mode=search_mode,
            )
            retrieved_ids = [str(h.metadata.get("source_id", "")) for h in hits]

        expected = list(case["expected_source_ids"])
        rows.append(
            {
                "query": case["query"],
                "expected": ", ".join(expected),
                "retrieved": ", ".join(retrieved_ids),
                "hit": _hit_rate(retrieved_ids, expected),
                "precision@k": round(_precision_at_k(retrieved_ids, expected, top_k), 4),
                "recall@k": round(_recall_at_k(retrieved_ids, expected, top_k), 4),
                "mrr": round(_mrr(retrieved_ids, expected), 4),
                "ndcg@k": round(_ndcg_at_k(retrieved_ids, expected, top_k), 4),
                "note": case.get("note", ""),
            }
        )

    n = len(rows) or 1
    current_store = store_info()
    if search_mode == "llama-index":
        try:
            from .llama_rag import get_backend_name
            current_store = dict(current_store)
            current_store["backend"] = f"LlamaIndex ({get_backend_name()})"
        except Exception:
            pass

    summary = {
        "search_mode": search_mode,
        "top_k": top_k,
        "n_cases": len(rows),
        "hit_rate": round(sum(r["hit"] for r in rows) / n, 4),
        "precision@k": round(sum(r["precision@k"] for r in rows) / n, 4),
        "recall@k": round(sum(r["recall@k"] for r in rows) / n, 4),
        "mrr": round(sum(r["mrr"] for r in rows) / n, 4),
        "ndcg@k": round(sum(r["ndcg@k"] for r in rows) / n, 4),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "store": current_store,
        "cases": rows,
    }
    return summary


def compare_search_modes(top_k: int = TOP_K) -> dict:
    """Evaluate LlamaIndex vs hybrid vs semantic vs keyword on the same golden set."""
    modes = ["llama-index", "hybrid", "semantic", "keyword"]
    by_mode = {mode: evaluate_retrieval(search_mode=mode, top_k=top_k) for mode in modes}
    leaderboard = [
        {
            "mode": mode,
            "hit_rate": by_mode[mode]["hit_rate"],
            "mrr": by_mode[mode]["mrr"],
            "ndcg@k": by_mode[mode]["ndcg@k"],
            "precision@k": by_mode[mode]["precision@k"],
            "recall@k": by_mode[mode]["recall@k"],
            "elapsed_ms": by_mode[mode]["elapsed_ms"],
        }
        for mode in modes
    ]
    return {"leaderboard": leaderboard, "by_mode": by_mode}
