"""Drive Medical SAP Training Assistant — login + RBAC-gated RAG chat.

Auth:    CSV-backed users + PBKDF2 hashing + JWT session (HS256).
Roles:   admin / hr / manager / finance / procurement / planning.
Vector:  Qdrant hybrid search (HNSW dense + BM25 sparse, RRF fusion).
Embed:   sentence-transformers/all-MiniLM-L6-v2 + FastEmbed BM25.
LLM:     Ollama with qwen2.5:3b (free, local), token-budgeted context.
Uploads: Admin + HR can upload md/txt/pdf/docx; chunks are immediately
         embedded and become retrievable on the next query (no restart).
"""
from __future__ import annotations

import sys

try:
    import sentence_transformers as _st_boot  # noqa: F401
    _st_boot_ok = True
except ModuleNotFoundError:
    _st_boot_ok = False
if not _st_boot_ok:
    sys.stderr.write(
        "\n[Import error] sentence-transformers is not installed for this Python.\n"
        f"  Interpreter: {sys.executable}\n"
        "  Use the project venv:\n"
        "    .\\.venv\\Scripts\\python.exe -m streamlit run app.py\n"
        "  Or: .\\run_app.ps1\n"
        "  If the venv is active but deps are missing: pip install -r requirements.txt\n\n"
    )
    raise SystemExit(1)

from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE importing modules that read env at import time.
_APP_DIR = Path(__file__).resolve().parent
load_dotenv(_APP_DIR / ".env", override=True)

import streamlit as st

from auth import authenticate, issue_token, verify_token
from auth.rbac import can_upload, upload_targets, visible_doc_roles
from src.config import (
    HNSW_EF_SEARCH,
    HNSW_M,
    LLM_CONTEXT_WINDOW,
    NO_ANSWER_MESSAGE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    ROLE_LABELS,
    SEARCH_MODE,
    SEARCH_MODE_LABELS,
    SEARCH_MODES,
    SYSTEM_LABELS,
    SYSTEMS,
    TOP_K,
    suggestions_for_role,
)
from src.document_loader import SUPPORTED_SUFFIXES
from src.metrics import compare_search_modes, recent_metrics
from src.rag_engine import RagResponse, generate_answer
from src.upload import ingest_uploaded_file
from src.vector_store import chunk_count_by_role, collection_count, store_info

st.set_page_config(
    page_title="Drive Medical SAP Assistant",
    page_icon="DM",
    layout="wide",
)


# --- Session helpers ------------------------------------------------------


def _init_session_state() -> None:
    st.session_state.setdefault("token", None)
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("system_filter", "both")
    default_mode = SEARCH_MODE if SEARCH_MODE in SEARCH_MODES else "hybrid"
    st.session_state.setdefault("search_mode", default_mode)
    st.session_state.setdefault("last_metrics", None)
    st.session_state.setdefault("eval_result", None)


def _current_user() -> dict | None:
    """Return the validated JWT payload, or None if no/expired/invalid token."""
    token = st.session_state.get("token")
    if not token:
        return None
    payload = verify_token(token)
    if payload is None:
        st.session_state.token = None
        st.session_state.user = None
        return None
    st.session_state.user = payload
    return payload


def _logout() -> None:
    st.session_state.token = None
    st.session_state.user = None
    st.session_state.messages = []
    st.rerun()


# --- Login page -----------------------------------------------------------


def _render_login_page() -> None:
    st.title("Drive Medical SAP Assistant")
    st.caption("Sign in to access SAP process documentation grounded in your role.")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", autocomplete="username")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if submitted:
        user = authenticate(username, password)
        if user is None:
            st.error("Invalid username or password.")
            return
        st.session_state.token = issue_token(user)
        st.session_state.user = verify_token(st.session_state.token)
        st.session_state.messages = []
        st.rerun()

    with st.expander("Demo accounts"):
        st.markdown(
            """
| Username     | Password     | Role         | Can upload? |
|--------------|--------------|--------------|-------------|
| `admin`      | `admin123`   | admin        | Yes (any category) |
| `hr`         | `hr123`      | hr           | Yes (HR + common) |
| `manager`    | `manager123` | manager      | No (read all functional) |
| `alice`      | `alice123`   | finance      | No (Finance + common) |
| `bob`        | `bob123`     | procurement  | No (Procurement + common) |
| `carol`      | `carol123`   | planning     | No (Planning + common) |
            """.strip()
        )


# --- Sidebar -------------------------------------------------------------


def _render_sidebar(user: dict) -> None:
    with st.sidebar:
        st.markdown(f"### {user.get('name') or user['sub']}")
        role = user["role"]
        st.caption(f"**Role:** {ROLE_LABELS.get(role, role)}")
        if st.button("Sign out", use_container_width=True):
            _logout()

        st.divider()
        st.markdown("#### Search mode")
        st.session_state.search_mode = st.selectbox(
            "Retrieval",
            options=SEARCH_MODES,
            index=SEARCH_MODES.index(st.session_state.search_mode)
            if st.session_state.search_mode in SEARCH_MODES
            else 0,
            format_func=lambda s: SEARCH_MODE_LABELS.get(s, s),
            help="Hybrid fuses HNSW semantic neighbors with BM25 keyword hits via Reciprocal Rank Fusion.",
        )

        st.divider()
        st.markdown("#### SAP system filter")
        st.session_state.system_filter = st.selectbox(
            "System",
            options=SYSTEMS,
            index=SYSTEMS.index(st.session_state.system_filter),
            format_func=lambda s: SYSTEM_LABELS.get(s, s),
            label_visibility="collapsed",
        )

        st.divider()
        backend_name = "Qdrant"
        try:
            from src.llama_rag import count_indexed_nodes, get_backend_name
            total = count_indexed_nodes()
            backend_name = get_backend_name()
        except Exception:
            total = 0
        if total == 0:
            try:
                total = collection_count()
            except Exception:  # noqa: BLE001
                total = 0

        st.metric("Indexed chunks", total)

        try:
            by_role = chunk_count_by_role()
        except Exception:  # noqa: BLE001
            by_role = {}
        visible = visible_doc_roles(role)
        accessible = sum(by_role.get(r, 0) for r in visible)
        if accessible == 0 and total > 0:
            accessible = total
        st.metric("Accessible to you", accessible)

        if total == 0:
            st.error(
                "Knowledge base is empty.\n\n"
                "Run `python ingest.py` once, or use the **Upload** tab "
                "(Admin / HR)."
            )

        st.success(f"🦙 **LlamaIndex Active**  \n`{backend_name}`")

        last = st.session_state.get("last_metrics")
        if last is not None:
            st.divider()
            st.markdown("#### Last query")
            st.metric("Retrieve + embed", f"{last.embed_retrieve_ms:.0f} ms")
            st.metric("Generate", f"{last.generate_ms:.0f} ms")
            st.caption(
                f"Context {last.context_tokens_used}/{last.context_tokens_available} tokens "
                f"({last.context_utilization:.0%} of budget) · "
                f"window {last.context_window}"
            )

        st.info(f"LLM: **Ollama** — `{OLLAMA_MODEL}` @ `{OLLAMA_BASE_URL}` (free / local).")
        st.caption(
            f"Framework: **LlamaIndex** · HNSW M={HNSW_M} ef={HNSW_EF_SEARCH} · "
            f"Vibeathon 2026"
        )


# --- Chat tab -------------------------------------------------------------


def _render_sources(response: RagResponse) -> None:
    if not response.sources:
        return
    with st.expander(f"Sources ({len(response.sources)})", expanded=False):
        for i, src in enumerate(response.sources, start=1):
            meta = src.metadata
            title = meta.get("title", "Untitled")
            sid = meta.get("source_id", "?")
            system = str(meta.get("system", "both")).upper()
            role = str(meta.get("role", "common")).title()
            path = meta.get("source_path", "")
            score_txt = f"{src.score:.3f}"
            mode = src.search_mode or "hybrid"

            st.markdown(
                f"**{i}. {title}**  \n"
                f"`{sid}` · role: {role} · system: {system} · "
                f"{mode} score: {score_txt} · file: `{path}`"
            )
            snippet = src.text.strip()
            if len(snippet) > 600:
                snippet = snippet[:600].rstrip() + " ..."
            st.markdown(
                f"<div style='border-left:3px solid #888; padding:8px 12px; "
                f"background:rgba(127,127,127,0.08); white-space:pre-wrap; "
                f"font-size:0.9rem;'>{snippet}</div>",
                unsafe_allow_html=True,
            )
            st.write("")


def _render_history() -> None:
    for entry in st.session_state.messages:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry["role"] == "assistant" and "response" in entry:
                if NO_ANSWER_MESSAGE not in (entry.get("content") or ""):
                    _render_sources(entry["response"])


def _handle_user_input(user: dict, prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    allowed = visible_doc_roles(user["role"])
    with st.chat_message("assistant"):
        with st.spinner(f"Searching {len(allowed)} document categories ..."):
            response = generate_answer(
                user_query=prompt,
                allowed_roles=allowed,
                system=st.session_state.system_filter,
                top_k=TOP_K,
                search_mode=st.session_state.search_mode,
            )
        st.session_state.last_metrics = response.metrics
        st.markdown(response.answer)
        is_no_answer = NO_ANSWER_MESSAGE in (response.answer or "")
        if not is_no_answer:
            _render_sources(response)
        else:
            st.info(
                "This topic is outside your role's document set. "
                f"You can see: **{', '.join(allowed)}**. "
                "Purchase-order SOPs are **procurement** — sign in as "
                "`bob` / `bob123` (or `admin`) to read ME21N."
            )

    st.session_state.messages.append(
        {"role": "assistant", "content": response.answer, "response": response}
    )


def _render_chat_tab(user: dict) -> None:
    st.subheader("Ask about your SAP processes")
    st.caption(
        "Answers are grounded ONLY in documentation visible to your role, with citations. "
        f"Visible categories: **{', '.join(visible_doc_roles(user['role']))}**. "
        f"Retrieval: **{SEARCH_MODE_LABELS.get(st.session_state.search_mode, st.session_state.search_mode)}**."
    )

    if not st.session_state.messages:
        cols = st.columns(3)
        suggestions = suggestions_for_role(user["role"])
        for col, q in zip(cols, suggestions):
            with col:
                if st.button(q, use_container_width=True):
                    _handle_user_input(user, q)
                    st.rerun()

    _render_history()

    prompt = st.chat_input("Ask a question about Drive Medical SAP processes ...")
    if prompt:
        _handle_user_input(user, prompt)
        st.rerun()


# --- Upload tab (admin + hr only) ----------------------------------------


def _render_upload_tab(user: dict) -> None:
    role = user["role"]
    if not can_upload(role):
        st.error("Your role is not permitted to upload documents.")
        st.caption("Only **admin** and **hr** roles may upload.")
        return

    st.subheader("Upload a new document")
    st.caption(
        "Uploaded files are saved to `data/<category>/`, embedded immediately, "
        "and become available in chat without restart. Re-uploading the same "
        "filename replaces existing chunks for that document."
    )

    targets = upload_targets(role)
    with st.form("upload_form", clear_on_submit=True):
        target_role = st.selectbox(
            "Document category (RBAC tier)",
            options=targets,
            format_func=lambda r: r.title(),
            help="Only users whose role can see this category will retrieve it.",
        )
        system = st.selectbox(
            "SAP system",
            options=SYSTEMS,
            format_func=lambda s: SYSTEM_LABELS.get(s, s),
        )
        title = st.text_input("Display title (optional)")
        uploaded = st.file_uploader(
            "Choose a file",
            type=[s.lstrip(".") for s in sorted(SUPPORTED_SUFFIXES)],
            accept_multiple_files=False,
        )
        submitted = st.form_submit_button("Upload + index", use_container_width=True)

    if submitted:
        if uploaded is None:
            st.error("Please choose a file.")
            return
        try:
            with st.spinner("Extracting text, embedding and indexing ..."):
                result = ingest_uploaded_file(
                    filename=uploaded.name,
                    data=uploaded.getvalue(),
                    target_role=target_role,
                    system=system,
                    title=title or None,
                    uploaded_by=user.get("sub", ""),
                    user_role=role,
                )
        except PermissionError as exc:
            st.error(f"Permission denied: {exc}")
            return
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Upload failed: {type(exc).__name__}: {exc}")
            return

        if result.chunk_count == 0:
            st.warning(
                "Upload succeeded but no text could be extracted — the document "
                "may be a scanned image PDF without OCR."
            )
        else:
            replaced_note = (
                f" (replaced {result.replaced_existing} previously indexed chunks)"
                if result.replaced_existing
                else ""
            )
            st.success(
                f"Indexed **{result.chunk_count}** chunks from "
                f"**{result.source_id}**{replaced_note}.\n\n"
                f"Saved to: `{result.saved_path}`"
            )

    st.divider()
    st.markdown("#### Currently indexed (by category)")
    by_role = chunk_count_by_role()
    if not by_role:
        st.caption("Knowledge base is empty.")
    else:
        rows = sorted(by_role.items(), key=lambda kv: -kv[1])
        st.table({"Category": [r for r, _ in rows], "Chunks": [n for _, n in rows]})


# --- Knowledge base tab ---------------------------------------------------


def _render_kb_tab(user: dict) -> None:
    st.subheader("Your access")
    role = user["role"]
    visible = visible_doc_roles(role)
    st.markdown(
        f"You are signed in as **{user.get('name') or user['sub']}** "
        f"(role: **{ROLE_LABELS.get(role, role)}**)."
    )
    st.markdown("**Visible document categories:**")
    st.code("\n".join(f"- {v}" for v in visible) or "(none)", language="markdown")

    st.divider()
    st.subheader("Indexed knowledge base")
    by_role = chunk_count_by_role()
    if not by_role:
        st.caption("Empty. Upload a document or run `python ingest.py`.")
        return
    rows = sorted(by_role.items(), key=lambda kv: -kv[1])
    st.table(
        {
            "Category": [r for r, _ in rows],
            "Chunks": [n for _, n in rows],
            "Visible to you?": ["Yes" if r in visible else "No" for r, _ in rows],
        }
    )


# --- Metrics tab ----------------------------------------------------------


def _render_metrics_tab() -> None:
    st.subheader("Retrieval quality & runtime metrics")
    st.caption(
        "HNSW (Hierarchical Navigable Small World) is the approximate nearest-neighbor "
        "index for semantic search. Keyword search uses BM25 sparse vectors. Hybrid "
        "fuses both rankings with Reciprocal Rank Fusion (RRF)."
    )

    try:
        info = store_info()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Vector store error: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ANN algorithm", info["ann_algorithm"])
    c2.metric("HNSW M / ef", f"{info['hnsw_m']} / {info['hnsw_ef_search']}")
    c3.metric("Indexed chunks", info["points"])
    c4.metric("Context window", LLM_CONTEXT_WINDOW)

    st.markdown(
        f"- **Backend:** Qdrant ({info['mode']}) @ `{info['url']}`  \n"
        f"- **Dense:** {info['dense_model']} ({info['dense_dim']}-d, cosine)  \n"
        f"- **Sparse:** {info['sparse_model']}  \n"
        f"- **Fusion:** {info['fusion']}  \n"
        f"- **Build ef_construct:** {info['hnsw_ef_construct']}"
    )

    st.divider()
    st.markdown("#### Last queries in this session")
    history = recent_metrics(10)
    if not history:
        st.caption("Ask a question in Chat to populate per-query latency and context stats.")
    else:
        st.table(
            {
                "Mode": [m.search_mode for m in history],
                "Hits": [m.n_after_pack for m in history],
                "Retrieve ms": [m.embed_retrieve_ms for m in history],
                "Generate ms": [m.generate_ms for m in history],
                "Ctx tokens": [m.context_tokens_used for m in history],
                "Ctx %": [f"{m.context_utilization:.0%}" for m in history],
                "Sources": [", ".join(s for s in m.source_ids if s) for m in history],
            }
        )

    st.divider()
    st.markdown("#### Golden-set evaluation")
    st.caption(
        "Runs 10 labeled SAP questions through hybrid, semantic, and keyword search. "
        "Reports Hit Rate, Precision@k, Recall@k, MRR, and nDCG@k."
    )
    if st.button("Run retrieval eval (all modes)", use_container_width=True):
        with st.spinner("Evaluating hybrid vs semantic vs keyword ..."):
            st.session_state.eval_result = compare_search_modes(top_k=TOP_K)

    result = st.session_state.get("eval_result")
    if not result:
        return

    st.markdown("**Leaderboard**")
    board = result["leaderboard"]
    st.table(
        {
            "Mode": [r["mode"] for r in board],
            "Hit rate": [r["hit_rate"] for r in board],
            "MRR": [r["mrr"] for r in board],
            "nDCG@k": [r["ndcg@k"] for r in board],
            "P@k": [r["precision@k"] for r in board],
            "R@k": [r["recall@k"] for r in board],
            "Elapsed ms": [r["elapsed_ms"] for r in board],
        }
    )

    for mode, summary in result["by_mode"].items():
        with st.expander(f"{mode} — per-case results", expanded=mode == "hybrid"):
            st.table(
                {
                    "Query": [c["query"] for c in summary["cases"]],
                    "Expected": [c["expected"] for c in summary["cases"]],
                    "Retrieved": [c["retrieved"] for c in summary["cases"]],
                    "Hit": ["Yes" if c["hit"] else "No" for c in summary["cases"]],
                    "MRR": [c["mrr"] for c in summary["cases"]],
                    "nDCG@k": [c["ndcg@k"] for c in summary["cases"]],
                }
            )


# --- Top-level ------------------------------------------------------------


def main() -> None:
    _init_session_state()
    user = _current_user()

    if user is None:
        _render_login_page()
        return

    _render_sidebar(user)
    st.title("Drive Medical SAP Training Assistant")
    st.caption(
        "Ask plain-English questions about Drive Medical's SAP processes. "
        "Answers are grounded in your role's documentation, with citations."
    )

    tabs = ["Chat", "Knowledge base", "Metrics"]
    if can_upload(user["role"]):
        tabs.insert(1, "Upload")

    rendered = st.tabs(tabs)
    name_to_widget = dict(zip(tabs, rendered))

    with name_to_widget["Chat"]:
        _render_chat_tab(user)
    if "Upload" in name_to_widget:
        with name_to_widget["Upload"]:
            _render_upload_tab(user)
    with name_to_widget["Knowledge base"]:
        _render_kb_tab(user)
    with name_to_widget["Metrics"]:
        _render_metrics_tab()


if __name__ == "__main__":
    main()
