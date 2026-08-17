"""CLI to ingest the data/ directory into Qdrant (hybrid dense + BM25).

Usage:
    python ingest.py            # incremental upsert
    python ingest.py --reset    # drop the collection first, then full re-ingest
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from src.config import DATA_DIR
from src.document_loader import load_documents
from src.vector_store import collection_count, reset_collection, store_info, upsert_chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest SAP docs into Qdrant (hybrid search).")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the collection before ingesting (full re-ingest).",
    )
    args = parser.parse_args()

    print(f"Reading documents from: {DATA_DIR}")
    chunks = list(load_documents(DATA_DIR))

    if not chunks:
        print("No documents found. Add markdown files under data/<role>/ and retry.")
        return 1

    if args.reset:
        print("Resetting Qdrant collection ...")
        reset_collection()

    print(f"Loaded {len(chunks)} chunks. Embedding (dense + BM25) and upserting ...")
    n = upsert_chunks(chunks)

    by_role = Counter(c.metadata["role"] for c in chunks)
    by_system = Counter(c.metadata["system"] for c in chunks)
    by_source = Counter(c.metadata["source_id"] for c in chunks)
    info = store_info()

    print(f"\nUpserted {n} chunks.")
    print(f"Collection now contains {collection_count()} chunks.")
    print(
        f"Index: {info['ann_algorithm']} (M={info['hnsw_m']}, "
        f"ef_construct={info['hnsw_ef_construct']}) + BM25 sparse"
    )
    print(f"Backend: {info['backend']} ({info['mode']}) @ {info['url']}\n")
    print("Chunks per role:")
    for role, count in sorted(by_role.items()):
        print(f"  {role:<14} {count}")
    print("\nChunks per system:")
    for system, count in sorted(by_system.items()):
        print(f"  {system:<14} {count}")
    print(f"\nDistinct source documents: {len(by_source)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
