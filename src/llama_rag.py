"""LlamaIndex RAG Orchestration Layer for Drive Medical SAP Assistant.

Integrates LlamaIndex end-to-end:
- VectorStoreIndex backed by Qdrant (auto-detects server vs embedded local disk)
  with fallback to Chroma / SimpleVectorStore.
- HuggingFaceEmbedding (sentence-transformers/all-MiniLM-L6-v2).
- RBAC MetadataFilters for role-based document access control and SAP system filtering.
- LlamaIndex RetrieverQueryEngine and ResponseSynthesizer with Drive Medical prompt template.
- Resilient LLM execution: Ollama (qwen2.5:3b) -> Gemini -> extractive fallback.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import (
    DATA_DIR,
    EMBEDDING_MODEL_NAME,
    GEMINI_MODEL_NAME,
    LLM_CONTEXT_WINDOW,
    LLM_PROVIDER,
    NO_ANSWER_MESSAGE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT_SEC,
    PROJECT_ROOT,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_DIR,
    QDRANT_EMBEDDED,
    QDRANT_URL,
    TOP_K,
)
from .document_loader import Chunk
from .tokens import ContextBudget, estimate_tokens, pack_chunks

count_tokens = estimate_tokens

logger = logging.getLogger(__name__)

# Persistence paths
STORAGE_DIR = PROJECT_ROOT / "storage"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

_EMBED_MODEL = None
_INDEX = None
_VECTOR_STORE = None
_BACKEND_NAME = "unknown"


@dataclass
class LlamaRetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    search_mode: str = "llama-index"
    extra: dict = field(default_factory=dict)


def get_embed_model():
    """Lazy initialization of HuggingFace embeddings via LlamaIndex."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        _EMBED_MODEL = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            embed_batch_size=32,
        )
    return _EMBED_MODEL


def _check_server_reachable(url: str, timeout: float = 2.0) -> bool:
    """Quick socket check to verify if a remote HTTP server is running."""
    import socket
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _init_vector_store():
    """Initialize the most appropriate vector store backend:
    1. Qdrant server (if reachable at QDRANT_URL and not explicitly embedded).
    2. Embedded Qdrant on local disk (qdrant_db/).
    3. ChromaVectorStore on local disk (chroma_db/) as fallback.
    4. SimpleVectorStore (storage/) as ultimate zero-dependency fallback.
    """
    global _BACKEND_NAME
    from llama_index.core import Settings
    Settings.embed_model = get_embed_model()

    # 1. Try Qdrant server if explicitly set and reachable
    if QDRANT_URL and not QDRANT_EMBEDDED:
        if _check_server_reachable(QDRANT_URL, timeout=1.5):
            try:
                import qdrant_client
                from llama_index.vector_stores.qdrant import QdrantVectorStore

                client = qdrant_client.QdrantClient(
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY or None,
                    timeout=10,
                )
                store = QdrantVectorStore(
                    client=client,
                    collection_name=QDRANT_COLLECTION,
                    dense_vector_name="dense",
                    sparse_vector_name="bm25",
                    enable_hybrid=True,
                    fastembed_sparse_model="Qdrant/bm25",
                )
                _BACKEND_NAME = f"Qdrant Server ({QDRANT_URL})"
                return store
            except Exception as e:
                logger.warning("Failed to connect to Qdrant server: %s", e)

    # 2. Try Embedded Qdrant (local disk storage in qdrant_db/)
    try:
        import qdrant_client
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        QDRANT_DIR.mkdir(parents=True, exist_ok=True)
        client = qdrant_client.QdrantClient(path=str(QDRANT_DIR))
        store = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION,
            dense_vector_name="dense",
            sparse_vector_name="bm25",
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25",
        )
        _BACKEND_NAME = "Qdrant Embedded (Local Disk)"
        return store
    except Exception as e:
        logger.warning("Embedded Qdrant initialization failed (%s), trying Chroma...", e)

    # 3. Try ChromaDB
    try:
        import chromadb
        from llama_index.vector_stores.chroma import ChromaVectorStore

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        chroma_collection = chroma_client.get_or_create_collection("sap_docs")
        store = ChromaVectorStore(chroma_collection=chroma_collection)
        _BACKEND_NAME = "ChromaDB Embedded (Local Disk)"
        return store
    except Exception as e:
        logger.warning("ChromaDB initialization failed (%s), using SimpleVectorStore...", e)

    # 4. Built-in SimpleVectorStore
    from llama_index.core.vector_stores import SimpleVectorStore
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    simple_file = STORAGE_DIR / "default__vector_store.json"
    if simple_file.exists():
        store = SimpleVectorStore.from_persist_path(str(simple_file))
    else:
        store = SimpleVectorStore()
    _BACKEND_NAME = "SimpleVectorStore (JSON)"
    return store


def get_vector_store():
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = _init_vector_store()
    return _VECTOR_STORE


def get_backend_name() -> str:
    get_vector_store()
    return _BACKEND_NAME


def get_index():
    """Retrieve or build the LlamaIndex VectorStoreIndex."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX

    from llama_index.core import StorageContext, VectorStoreIndex
    store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=store)

    # If storage dir has an index store, try loading from it
    try:
        _INDEX = VectorStoreIndex.from_vector_store(
            vector_store=store,
            embed_model=get_embed_model(),
        )
    except Exception:
        _INDEX = VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            embed_model=get_embed_model(),
        )
    return _INDEX


def chunk_to_llama_node(chunk: Chunk):
    """Convert an internal Chunk into a LlamaIndex TextNode with rich metadata."""
    import uuid
    from llama_index.core.schema import TextNode

    meta = dict(chunk.metadata)
    meta.setdefault("source_id", "DOC")
    meta.setdefault("role", "common")
    meta.setdefault("system", "both")
    meta.setdefault("title", "Untitled")
    meta["chunk_id"] = chunk.chunk_id

    # Qdrant requires point IDs to be UUIDs or unsigned integers
    node_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))

    return TextNode(
        text=chunk.text,
        id_=node_uuid,
        metadata=meta,
        excluded_embed_metadata_keys=["source_path", "uploaded_by", "chunk_index", "total_chunks"],
        excluded_llm_metadata_keys=["source_path", "uploaded_by", "chunk_index", "total_chunks"],
    )


def ingest_chunks_llama(chunks: Sequence[Chunk]) -> int:
    """Ingest chunks into the LlamaIndex VectorStoreIndex."""
    if not chunks:
        return 0

    index = get_index()
    nodes = [chunk_to_llama_node(c) for c in chunks]
    index.insert_nodes(nodes)

    # If simple vector store, persist to disk
    store = get_vector_store()
    if hasattr(store, "persist"):
        try:
            store.persist(persist_path=str(STORAGE_DIR / "default__vector_store.json"))
        except Exception:
            pass

    return len(nodes)


def delete_by_source_id_llama(source_id: str) -> int:
    """Delete nodes matching source_id from the vector store."""
    if not source_id:
        return 0
    store = get_vector_store()
    # Check if delete method exists
    try:
        if hasattr(store, "delete"):
            store.delete(source_id)
            return 1
    except Exception:
        pass
    return 0


def count_indexed_nodes() -> int:
    """Return total number of nodes in the active LlamaIndex vector store."""
    store = get_vector_store()
    try:
        if hasattr(store, "client") and hasattr(store.client, "count"):
            counted = store.client.count(collection_name=QDRANT_COLLECTION, exact=True)
            return int(counted.count)
        if hasattr(store, "_collection") and hasattr(store._collection, "count"):
            return int(store._collection.count())
        if hasattr(store, "data") and hasattr(store.data, "embedding_dict"):
            return len(store.data.embedding_dict)
    except Exception:
        pass
    return 0


def get_llama_llm():
    """Instantiate the primary LLM configured for LlamaIndex."""
    provider = (LLM_PROVIDER or "ollama").strip().lower()

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if api_key:
            try:
                from llama_index.llms.gemini import Gemini
                return Gemini(model_name=f"models/{GEMINI_MODEL_NAME}", api_key=api_key)
            except Exception as e:
                logger.warning("Gemini LLM init failed: %s", e)

    # Default to Ollama
    try:
        from llama_index.llms.ollama import Ollama
        return Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=OLLAMA_TEMPERATURE,
            request_timeout=OLLAMA_TIMEOUT_SEC,
        )
    except Exception as e:
        logger.warning("Ollama LLM init failed: %s", e)
        return None


def build_llama_filters(allowed_roles: Iterable[str], system: str = "both"):
    """Construct LlamaIndex MetadataFilters enforcing RBAC and SAP system selection."""
    from llama_index.core.vector_stores.types import (
        FilterCondition,
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
    )

    roles = sorted({(r or "").lower() for r in allowed_roles if r}) or ["common"]
    sys_val = (system or "both").lower()

    filters_list = [
        MetadataFilter(key="role", value=roles, operator=FilterOperator.IN),
    ]

    if sys_val != "both":
        filters_list.append(
            MetadataFilter(key="system", value=[sys_val, "both"], operator=FilterOperator.IN)
        )

    return MetadataFilters(filters=filters_list, condition=FilterCondition.AND)


def retrieve_nodes_llama(
    user_query: str,
    allowed_roles: Iterable[str],
    system: str = "both",
    top_k: int = TOP_K,
) -> list[LlamaRetrievedChunk]:
    """Retrieve nodes using LlamaIndex VectorIndexRetriever with RBAC metadata filtering."""
    if not user_query or not user_query.strip():
        return []

    index = get_index()
    filters = build_llama_filters(allowed_roles, system)

    try:
        retriever = index.as_retriever(
            similarity_top_k=top_k,
            filters=filters,
        )
        nodes_with_scores = retriever.retrieve(user_query)
    except Exception as exc:
        logger.warning("LlamaIndex filtered retrieval fallback: %s", exc)
        # Fallback without deep filter if the backend filter dialect has a discrepancy
        try:
            retriever = index.as_retriever(similarity_top_k=top_k * 2)
            raw_nodes = retriever.retrieve(user_query)
            roles_set = set(allowed_roles)
            nodes_with_scores = []
            for n in raw_nodes:
                n_role = str(n.node.metadata.get("role", "common")).lower()
                n_sys = str(n.node.metadata.get("system", "both")).lower()
                if n_role in roles_set:
                    if system == "both" or n_sys in (system, "both"):
                        nodes_with_scores.append(n)
                if len(nodes_with_scores) >= top_k:
                    break
        except Exception as e2:
            logger.error("LlamaIndex retrieval completely failed: %s", e2)
            return []

    results: list[LlamaRetrievedChunk] = []
    for item in nodes_with_scores:
        node = item.node
        meta = dict(node.metadata or {})
        score = float(item.score if item.score is not None else 0.0)
        results.append(
            LlamaRetrievedChunk(
                chunk_id=node.node_id or str(meta.get("source_id", "DOC")),
                text=node.get_content(),
                metadata=meta,
                score=score,
                search_mode="llama-index",
            )
        )
    return results


KNOWN_TCODES = {
    "MIRO", "MIGO", "FB50", "MD01", "F110", "ME21N", "ME51N", "VA01", "XK01", "FS00", "ME23N", "MM01",
}

_STOPWORDS = {
    "how", "do", "i", "a", "an", "the", "in", "on", "at", "to", "of", "and",
    "or", "for", "my", "me", "using", "with", "about", "what", "is", "are",
    "please", "walk", "through", "can", "you", "your", "this", "that", "via",
    "sap", "transaction", "cycle", "procedure",
}


def extract_tcodes(text: str) -> set[str]:
    """Extract standard SAP transaction codes from query or document text."""
    import re
    tokens = re.findall(r"\b[A-Za-z0-9_]{3,7}\b", text or "")
    found = set()
    for tok in tokens:
        u = tok.upper()
        if u in KNOWN_TCODES or re.match(r"^[A-Z]{1,2}\d{2,4}[A-Z0-9]?$", u):
            found.add(u)
    return found


def _terms(text: str) -> set[str]:
    import re
    return {
        t for t in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
        if t not in _STOPWORDS
    }


def _query_overlap(query: str, chunk: LlamaRetrievedChunk) -> float:
    q = _terms(query)
    if not q:
        return 0.0
    blob = f"{chunk.text} {chunk.metadata.get('title', '')} {chunk.metadata.get('source_id', '')}"
    return len(q & _terms(blob)) / len(q)


def filter_relevant_chunks(
    user_query: str,
    chunks: list[LlamaRetrievedChunk],
) -> list[LlamaRetrievedChunk]:
    """Filter out irrelevant chunks and verify authorization/relevance."""
    if not chunks or not user_query.strip():
        return []

    # 1. If query specifies an SAP T-code, the SOP document must cover that T-code in its title or source_id
    q_tcodes = extract_tcodes(user_query)
    if q_tcodes:
        matching = []
        for c in chunks:
            title_id = f"{c.metadata.get('title', '')} {c.metadata.get('source_id', '')}".upper()
            if any(tc in title_id for tc in q_tcodes):
                matching.append(c)
        return matching

    # 2. For non-T-code queries: check score and query term overlap
    top_score = chunks[0].score if chunks else 0.0
    top_ov = _query_overlap(user_query, chunks[0])
    if top_score < 0.42 and top_ov < 0.20:
        return []

    kept = []
    for i, c in enumerate(chunks):
        if i == 0 or _query_overlap(user_query, c) >= 0.20 or c.score >= 0.45:
            kept.append(c)
    return kept


def extractive_fallback(chunks: Sequence[LlamaRetrievedChunk]) -> str:
    """Generate a clean, structured answer directly from retrieved chunks."""
    if not chunks:
        return NO_ANSWER_MESSAGE

    blocks: list[str] = ["Here is the procedure from your role's documentation:\n"]
    seen: list[str] = []
    for chunk in chunks:
        sid = str(chunk.metadata.get("source_id", "DOC"))
        title = str(chunk.metadata.get("title", "Untitled"))
        if sid not in seen:
            blocks.append(f"### {title} (`{sid}`)\n")
            seen.append(sid)
        blocks.append(chunk.text.strip())
        blocks.append("")
    if seen:
        blocks.append("Sources: " + ", ".join(seen))
    return "\n".join(blocks)


def run_llama_query(
    user_query: str,
    allowed_roles: Iterable[str],
    system: str = "both",
    top_k: int = TOP_K,
) -> tuple[str, list[LlamaRetrievedChunk], bool, float]:
    """Execute full LlamaIndex RAG query with synthesis and robust fallback."""
    t0 = time.perf_counter()
    raw_chunks = retrieve_nodes_llama(
        user_query=user_query,
        allowed_roles=allowed_roles,
        system=system,
        top_k=top_k,
    )
    retrieve_ms = (time.perf_counter() - t0) * 1000

    chunks = filter_relevant_chunks(user_query, raw_chunks)
    if not chunks:
        return NO_ANSWER_MESSAGE, [], False, retrieve_ms

    llm = get_llama_llm()
    if llm is None or not _check_server_reachable(OLLAMA_BASE_URL, timeout=1.0):
        # Service is offline - use immediate grounded extractive fallback
        answer = extractive_fallback(chunks)
        return answer, chunks, False, retrieve_ms

    # Try LLM generation via LlamaIndex PromptTemplate
    from llama_index.core.prompts import PromptTemplate

    qa_template = PromptTemplate(
        "You are the Drive Medical SAP Training Assistant.\n"
        "Answer the user's question using ONLY the provided context.\n"
        "If the context does not contain the answer, reply with: " + NO_ANSWER_MESSAGE + "\n\n"
        "Context:\n---------------------\n{context_str}\n---------------------\n"
        "User Question: {query_str}\n"
        "Answer with precise, step-by-step guidance and include the source citations at the end.\n"
    )

    t_gen = time.perf_counter()
    try:
        context_str = "\n\n".join(
            f"[{c.metadata.get('source_id', 'DOC')}] {c.metadata.get('title', '')}:\n{c.text}"
            for c in chunks
        )
        prompt_text = qa_template.format(context_str=context_str, query_str=user_query)
        response = llm.complete(prompt_text)
        answer_text = str(response).strip()
        if not answer_text or answer_text == NO_ANSWER_MESSAGE or answer_text.startswith(NO_ANSWER_MESSAGE):
            return NO_ANSWER_MESSAGE, [], True, (time.perf_counter() - t0) * 1000
        generate_ms = (time.perf_counter() - t_gen) * 1000
        return answer_text, chunks, True, retrieve_ms + generate_ms
    except Exception as exc:
        logger.info("LLM generation unavailable (%s); using extractive grounded response", exc)
        answer = extractive_fallback(chunks)
        return answer, chunks, False, retrieve_ms
