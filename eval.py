"""CLI: retrieval metrics for hybrid vs semantic vs keyword search.

Usage:
    python eval.py
    python eval.py --mode hybrid --top-k 5
    python eval.py --compare
"""
from __future__ import annotations

import argparse
import json
import sys

from src.metrics import compare_search_modes, evaluate_retrieval
from src.vector_store import collection_count, store_info


def _print_summary(summary: dict) -> None:
    print(f"\nMode: {summary['search_mode']}   top_k={summary['top_k']}   cases={summary['n_cases']}")
    print(f"  Hit Rate     {summary['hit_rate']:.3f}")
    print(f"  Precision@k  {summary['precision@k']:.3f}")
    print(f"  Recall@k     {summary['recall@k']:.3f}")
    print(f"  MRR          {summary['mrr']:.3f}")
    print(f"  nDCG@k       {summary['ndcg@k']:.3f}")
    print(f"  Elapsed      {summary['elapsed_ms']:.0f} ms")
    print("\nPer-case:")
    for row in summary["cases"]:
        mark = "OK" if row["hit"] else "MISS"
        print(f"  [{mark:<4}] {row['query'][:64]}")
        print(f"         expected={row['expected']}  retrieved={row['retrieved']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality.")
    parser.add_argument("--mode", choices=["hybrid", "semantic", "keyword"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--compare", action="store_true", help="Run all three search modes.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of a table.")
    args = parser.parse_args()

    n = collection_count()
    if n == 0:
        print("Knowledge base is empty. Run `python ingest.py` first.")
        return 1

    info = store_info()
    print(
        f"Qdrant {info['mode']} | {info['ann_algorithm']} "
        f"M={info['hnsw_m']} ef={info['hnsw_ef_search']} | {n} chunks"
    )

    if args.compare:
        result = compare_search_modes(top_k=args.top_k)
        if args.json:
            print(json.dumps(result, indent=2))
            return 0
        print("\nLeaderboard (higher is better except elapsed):")
        print(f"{'mode':<12} {'hit':>6} {'mrr':>6} {'ndcg':>6} {'p@k':>6} {'r@k':>6} {'ms':>8}")
        for row in result["leaderboard"]:
            print(
                f"{row['mode']:<12} {row['hit_rate']:6.3f} {row['mrr']:6.3f} "
                f"{row['ndcg@k']:6.3f} {row['precision@k']:6.3f} "
                f"{row['recall@k']:6.3f} {row['elapsed_ms']:8.0f}"
            )
        for mode, summary in result["by_mode"].items():
            _print_summary(summary)
        return 0

    summary = evaluate_retrieval(search_mode=args.mode, top_k=args.top_k)
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
