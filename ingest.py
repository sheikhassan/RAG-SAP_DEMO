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

    n = 0
    try:
        n = upsert_chunks(chunks)
        print(f"Qdrant: upserted {n} chunks.")
    except Exception as exc:
        print(f"Qdrant server note (proceeding with LlamaIndex): {exc}")

    print("Indexing into LlamaIndex VectorStoreIndex ...")
    try:
        from src.llama_rag import get_backend_name, ingest_chunks_llama
        n_llama = ingest_chunks_llama(chunks)
        print(f"LlamaIndex [{get_backend_name()}]: successfully indexed {n_llama} nodes.")
    except Exception as exc:
        print(f"LlamaIndex indexing error: {exc}")

    by_role = Counter(c.metadata["role"] for c in chunks)
    by_system = Counter(c.metadata["system"] for c in chunks)
    by_source = Counter(c.metadata["source_id"] for c in chunks)

    print("\nChunks per role:")
    for role, count in sorted(by_role.items()):
        print(f"  {role:<14} {count}")
    print("\nChunks per system:")
    for system, count in sorted(by_system.items()):
        print(f"  {system:<14} {count}")
    print(f"\nDistinct source documents: {len(by_source)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
