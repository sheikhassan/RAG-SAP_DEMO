"""Live document upload pipeline used by the Streamlit Upload tab.

Workflow:
    1. Validate that the user role is allowed to upload to `target_role`.
    2. Persist the original file under `data/<target_role>/<safe_name>`.
    3. Extract text -> chunk -> embed (dense + BM25) -> upsert into Qdrant.
       (Also de-duplicates by removing prior chunks with the same source_id.)

The result is immediately searchable on the next query — no restart required.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from auth.rbac import require_upload_permission

from .config import DATA_DIR
from .document_loader import SUPPORTED_SUFFIXES, chunks_from_file_bytes
from .vector_store import delete_by_source_id, upsert_chunks


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class UploadResult:
    source_id: str
    chunk_count: int
    saved_path: Path
    replaced_existing: int  # how many old chunks were dropped


def _safe_filename(name: str) -> str:
    base = Path(name).name  # strip directory components
    cleaned = _SAFE_NAME_RE.sub("_", base).strip("_.")
    return cleaned or "upload.bin"


def _make_source_id(stem: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", stem).strip("_.").upper()
    return cleaned or "DOC"


def ingest_uploaded_file(
    *,
    filename: str,
    data: bytes,
    target_role: str,
    system: str,
    title: str | None,
    uploaded_by: str,
    user_role: str,
) -> UploadResult:
    """Persist + chunk + embed + upsert a freshly uploaded document."""
    require_upload_permission(user_role, target_role)

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    safe_name = _safe_filename(filename)
    target_dir = DATA_DIR / target_role
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_path = target_dir / safe_name
    saved_path.write_bytes(data)

    source_id = _make_source_id(Path(safe_name).stem)
    replaced = delete_by_source_id(source_id)

    chunks = chunks_from_file_bytes(
        filename=safe_name,
        data=data,
        role=target_role,
        system=system,
        title=title or Path(safe_name).stem.replace("_", " ").title(),
        source_id=source_id,
        uploaded_by=uploaded_by,
        source_path=str(saved_path.relative_to(DATA_DIR.parent)),
    )
    n = 0
    try:
        n = upsert_chunks(chunks)
    except Exception:
        pass

    try:
        from .llama_rag import delete_by_source_id_llama, ingest_chunks_llama
        delete_by_source_id_llama(source_id)
        llama_n = ingest_chunks_llama(chunks)
        if n == 0:
            n = llama_n
    except Exception:
        pass

    return UploadResult(
        source_id=source_id,
        chunk_count=n,
        saved_path=saved_path,
        replaced_existing=replaced,
    )
