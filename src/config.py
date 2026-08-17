"""Central configuration for the RAG SAP Training Chatbot.

Single source of truth for paths, models, hybrid search, HNSW, context
window, RBAC roles, and JWT settings.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths ----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
QDRANT_DIR = PROJECT_ROOT / "qdrant_db"
AUTH_DIR = PROJECT_ROOT / "auth"
USERS_CSV = AUTH_DIR / "users.csv"

# --- Vector store (Qdrant: hybrid dense + sparse, HNSW ANN) ---------------

QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "sap_docs")
# Empty URL = embedded local disk. Set QDRANT_URL for a Qdrant server.
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()

# Dense (semantic) embeddings — local, open-source.
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# Sparse (keyword) embeddings — BM25 via FastEmbed.
SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL_NAME", "Qdrant/bm25")
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

# Hierarchical Navigable Small World (HNSW) — the ANN index Qdrant uses
# for nearest-neighbor search. Higher M / ef = better recall, more RAM/CPU.
HNSW_M = int(os.getenv("HNSW_M", "16"))
HNSW_EF_CONSTRUCT = int(os.getenv("HNSW_EF_CONSTRUCT", "128"))
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "64"))
HNSW_FULL_SCAN_THRESHOLD = int(os.getenv("HNSW_FULL_SCAN_THRESHOLD", "10000"))

# hybrid | semantic | keyword
SEARCH_MODE = os.getenv("SEARCH_MODE", "hybrid").strip().lower()
HYBRID_PREFETCH_K = int(os.getenv("HYBRID_PREFETCH_K", "20"))
# Reciprocal Rank Fusion constant (standard is 60).
RRF_K = int(os.getenv("RRF_K", "60"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Drop semantic-only hits below this cosine similarity.
MIN_DENSE_SCORE = float(os.getenv("MIN_DENSE_SCORE", "0.30"))

# --- Context window / token budget ----------------------------------------
# qwen2.5:3b default context is 32k. We reserve room for the system prompt,
# the user question, and generated tokens so retrieved chunks cannot overflow.

LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "32768"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "6000"))
GENERATION_RESERVE_TOKENS = int(os.getenv("GENERATION_RESERVE_TOKENS", "1024"))
SYSTEM_RESERVE_TOKENS = int(os.getenv("SYSTEM_RESERVE_TOKENS", "800"))

# --- LLM (default: free local Ollama running qwen2.5:3b) ------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT_SEC = float(os.getenv("OLLAMA_TIMEOUT_SEC", "120"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

# Optional Gemini fallback (only used if LLM_PROVIDER=gemini and key is set).
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# --- Auth (JWT) -----------------------------------------------------------

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_TTL_HOURS = int(os.getenv("JWT_TTL_HOURS", "8"))

# --- Roles & RBAC ---------------------------------------------------------
#
# A user has exactly one role (stored in the JWT). A document has exactly one
# `role` metadata value (the access tier the document belongs to). The
# ROLE_VISIBILITY map is the single source of truth: when a user with role
# `r` queries the vector store, only chunks whose `role` is in
# ROLE_VISIBILITY[r] are retrieved.

ROLES = ["admin", "hr", "manager", "finance", "procurement", "planning"]

ROLE_LABELS = {
    "admin": "Admin (full access + uploads)",
    "hr": "HR (HR docs + uploads)",
    "manager": "Manager (cross-functional read access)",
    "finance": "Finance (FI / AP)",
    "procurement": "Procurement",
    "planning": "Planning (IBP / SIOP)",
}

# Document-side role tags. `common` is visible to everyone. `restricted` is
# admin-only. `manager` docs are visible to managers and admins.
DOC_ROLES = [
    "common",
    "finance",
    "procurement",
    "planning",
    "manager",
    "hr",
    "restricted",
]

ROLE_VISIBILITY: dict[str, list[str]] = {
    "admin":       ["common", "finance", "procurement", "planning", "manager", "hr", "restricted"],
    "hr":          ["common", "hr"],
    "manager":     ["common", "finance", "procurement", "planning", "manager"],
    "finance":     ["common", "finance"],
    "procurement": ["common", "procurement"],
    "planning":    ["common", "planning"],
}

# Which document `role` tags each user role may upload to.
UPLOAD_ALLOWED_ROLES: dict[str, list[str]] = {
    "admin": ["common", "finance", "procurement", "planning", "manager", "hr", "restricted"],
    "hr":    ["common", "hr"],
}


def can_upload(user_role: str) -> bool:
    return user_role in UPLOAD_ALLOWED_ROLES


def visible_doc_roles(user_role: str) -> list[str]:
    """Document `role` tags visible to a user with this role."""
    return ROLE_VISIBILITY.get(user_role, ["common"])


def upload_targets(user_role: str) -> list[str]:
    """Document `role` tags this user may upload to."""
    return UPLOAD_ALLOWED_ROLES.get(user_role, [])


# --- SAP system filter ----------------------------------------------------

SYSTEMS = ["both", "ecc", "s4hana"]
SYSTEM_LABELS = {
    "both": "Both ECC & S/4HANA",
    "ecc": "SAP ECC only",
    "s4hana": "SAP S/4HANA only",
}

SEARCH_MODES = ["hybrid", "semantic", "keyword"]
SEARCH_MODE_LABELS = {
    "hybrid": "Hybrid (semantic + keyword, RRF)",
    "semantic": "Semantic only (HNSW / dense)",
    "keyword": "Keyword only (BM25 / sparse)",
}

# --- Misc -----------------------------------------------------------------

NO_ANSWER_MESSAGE = (
    "I could not find this in your role's documentation. "
    "Please raise an IT ticket or check with your admin."
)
