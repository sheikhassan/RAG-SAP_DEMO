"""RAG orchestration: hybrid retrieve, pack to context window, generate.

Default LLM is **Ollama** running `qwen2.5:3b` locally. Optional Gemini
fallback kicks in only when LLM_PROVIDER=gemini and GOOGLE_API_KEY is set.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Iterable

from .config import (
    GEMINI_MODEL_NAME,
    LLM_CONTEXT_WINDOW,
    LLM_PROVIDER,
    MIN_DENSE_SCORE,
    NO_ANSWER_MESSAGE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT_SEC,
    SEARCH_MODE,
    TOP_K,
)
from .metrics import QueryMetrics, build_query_metrics, record
from .ollama_client import (
    OllamaError,
    OllamaHTTPError,
    OllamaRuntimeError,
    OllamaUnreachableError,
    ollama_chat,
)
from .tokens import ContextBudget, pack_chunks
from .vector_store import RetrievedChunk, query

try:
    from google.api_core import exceptions as google_api_exceptions
except ImportError:  # google-generativeai is optional
    google_api_exceptions = None


# --- Identity / smalltalk -------------------------------------------------


def _is_identity_smalltalk(user_query: str) -> bool:
    t = user_query.strip().lower()
    if not t or len(t) > 160:
        return False
    if t in {"hi", "hey", "hello", "yo", "thanks", "thank you"}:
        return True
    needles = (
        "who are you",
        "what are you",
        "what is your name",
        "your name",
        "introduce yourself",
        "hello",
        "hi there",
        "thank you",
        "thanks",
    )
    return any(n in t for n in needles)


# --- Prompt construction -------------------------------------------------


SYSTEM_PROMPT = """You are the Drive Medical SAP Training Assistant.

You answer questions about Drive Medical's SAP processes (ECC and S/4HANA).

STRICT RULES — you MUST follow these:
1. Use ONLY the information in the "CONTEXT" section below. Do NOT use any
   outside SAP knowledge, even if you know the answer.
2. If the context does not contain enough information to answer, reply with
   exactly this sentence and nothing else:
   "{no_answer}"
3. When you do answer, be concrete and step-by-step. Preserve transaction
   codes (e.g. ME21N, MIRO), Fiori app names, field names, and any Drive
   Medical-specific values (company codes, plant numbers, thresholds).
4. After your answer, include a "Sources:" line listing the source_id values
   you actually used, comma-separated. Do not invent source ids.
5. Keep the tone professional and concise. Use markdown headings, numbered
   lists, and bold for field names where it helps clarity.
"""


@dataclass
class RagResponse:
    answer: str
    sources: list[RetrievedChunk]
    used_llm: bool
    metrics: QueryMetrics | None = None


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        meta = c.metadata
        title = meta.get("title", "Untitled")
        sid = meta.get("source_id", "UNKNOWN")
        system = str(meta.get("system", "both")).upper()
        role = str(meta.get("role", "common")).title()
        blocks.append(
            f"--- Source {i} ---\n"
            f"source_id: {sid}\n"
            f"title: {title}\n"
            f"system: {system}\n"
            f"role: {role}\n"
            f"content:\n{c.text}\n"
        )
    return "\n".join(blocks)


def _build_user_prompt(user_query: str, context: str) -> str:
    return (
        f"CONTEXT:\n{context}\n\n"
        f"USER QUESTION:\n{user_query}\n\n"
        "Answer following the STRICT RULES above."
    )


# --- Retrieval ------------------------------------------------------------


def retrieve(
    user_query: str,
    allowed_roles: Iterable[str],
    system: str = "both",
    top_k: int = TOP_K,
    search_mode: str | None = None,
) -> list[RetrievedChunk]:
    return query(
        user_query,
        allowed_roles=allowed_roles,
        system=system,
        top_k=top_k,
        search_mode=search_mode,
    )


def _filter_relevant(chunks: list[RetrievedChunk], search_mode: str) -> list[RetrievedChunk]:
    """Semantic-only uses a cosine floor; hybrid/keyword already rank via RRF/BM25."""
    if search_mode == "semantic":
        return [c for c in chunks if c.score >= MIN_DENSE_SCORE]
    return list(chunks)


# --- Ollama / Gemini glue ------------------------------------------------


def _format_ollama_unreachable_help() -> str:
    return (
        f"**Cannot reach Ollama.** Answers use a free local open model "
        f"(`{OLLAMA_MODEL}`). Retrieval still worked — expand **Sources** below.\n\n"
        "**Setup:**\n"
        "1. Install [Ollama](https://ollama.com) and leave it running.\n"
        f"2. Pull the model once: **`ollama pull {OLLAMA_MODEL}`** (or set "
        "`OLLAMA_MODEL` in `.env` to whatever you have pulled).\n"
        f"3. Ensure `OLLAMA_BASE_URL` matches Ollama (default `{OLLAMA_BASE_URL}`)."
    )


def _safe_gemini_text(response: object) -> tuple[str, str | None]:
    pf = getattr(response, "prompt_feedback", None)
    if pf is not None:
        br = getattr(pf, "block_reason", None)
        if br is not None and str(br) not in ("BLOCK_REASON_UNSPECIFIED", "0", ""):
            return (
                "The model blocked this request at the prompt stage (safety / policy). "
                "Try a shorter SAP question.",
                f"prompt_block_reason={br}",
            )
    try:
        raw_text = response.text  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - SDK raises generic exceptions
        raw_text = None
        note = f"response.text_exception:{type(exc).__name__}:{str(exc)[:240]}"
    else:
        note = "response.text_empty"
    t = (raw_text or "").strip() if raw_text else ""
    if t:
        return t, None

    parts_out: list[str] = []
    try:
        for cand in getattr(response, "candidates", None) or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", None) or []:
                txt = getattr(part, "text", None)
                if txt:
                    parts_out.append(txt)
    except Exception as exc:  # noqa: BLE001
        return NO_ANSWER_MESSAGE, f"parts_walk_failed:{type(exc).__name__}:{str(exc)[:200]}"

    joined = "\n".join(parts_out).strip()
    return (joined, note) if joined else (NO_ANSWER_MESSAGE, note)


def _configure_gemini():
    import google.generativeai as genai

    api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        GEMINI_MODEL_NAME,
        system_instruction=SYSTEM_PROMPT.format(no_answer=NO_ANSWER_MESSAGE),
    )


def _empty_budget() -> ContextBudget:
    return ContextBudget(
        window=LLM_CONTEXT_WINDOW,
        reserved_system=0,
        reserved_generation=0,
        query_tokens=0,
        available=0,
        used=0,
        utilization=0.0,
        chunks_kept=0,
        chunks_dropped=0,
    )


# --- Top-level entry ------------------------------------------------------


def generate_answer(
    user_query: str,
    allowed_roles: Iterable[str],
    system: str = "both",
    top_k: int = TOP_K,
    search_mode: str | None = None,
) -> RagResponse:
    """Retrieve role-filtered chunks, pack to the context budget, generate."""
    t0 = time.perf_counter()
    mode = (search_mode or SEARCH_MODE or "hybrid").strip().lower()
    system_instruction = SYSTEM_PROMPT.format(no_answer=NO_ANSWER_MESSAGE)

    if _is_identity_smalltalk(user_query):
        metrics = build_query_metrics(
            search_mode=mode,
            chunks=[],
            packed=[],
            budget=_empty_budget(),
            retrieve_ms=0.0,
            generate_ms=0.0,
            total_ms=(time.perf_counter() - t0) * 1000,
        )
        return RagResponse(
            answer=(
                "I am the **Drive Medical SAP Training Assistant** — a retrieval-"
                "augmented chatbot for this demo. I answer SAP process questions "
                "using only your organization's documentation (with role-based "
                "access and source citations). I am not a general web assistant."
            ),
            sources=[],
            used_llm=False,
            metrics=record(metrics),
        )

    t_ret = time.perf_counter()
    chunks = retrieve(
        user_query,
        allowed_roles=allowed_roles,
        system=system,
        top_k=top_k,
        search_mode=mode,
    )
    relevant = _filter_relevant(chunks, mode)
    packed, budget = pack_chunks(relevant, user_query, system_instruction)
    retrieve_ms = (time.perf_counter() - t_ret) * 1000

    if not packed:
        metrics = build_query_metrics(
            search_mode=mode,
            chunks=chunks,
            packed=[],
            budget=budget,
            retrieve_ms=retrieve_ms,
            generate_ms=0.0,
            total_ms=(time.perf_counter() - t0) * 1000,
        )
        return RagResponse(
            answer=NO_ANSWER_MESSAGE,
            sources=[],
            used_llm=False,
            metrics=record(metrics),
        )

    context = _format_context(packed)
    prompt = _build_user_prompt(user_query, context)
    provider = LLM_PROVIDER if LLM_PROVIDER in ("ollama", "gemini") else "ollama"

    def _finish(answer: str, used_llm: bool, generate_ms: float) -> RagResponse:
        metrics = build_query_metrics(
            search_mode=mode,
            chunks=relevant,
            packed=packed,
            budget=budget,
            retrieve_ms=retrieve_ms,
            generate_ms=generate_ms,
            total_ms=(time.perf_counter() - t0) * 1000,
        )
        return RagResponse(
            answer=answer,
            sources=packed,
            used_llm=used_llm,
            metrics=record(metrics),
        )

    if provider == "ollama":
        t_gen = time.perf_counter()
        try:
            answer_text = ollama_chat(
                OLLAMA_BASE_URL,
                OLLAMA_MODEL,
                system_instruction,
                prompt,
                timeout_sec=OLLAMA_TIMEOUT_SEC,
                num_ctx=LLM_CONTEXT_WINDOW,
                temperature=OLLAMA_TEMPERATURE,
            )
            if not answer_text.strip():
                answer_text = NO_ANSWER_MESSAGE
        except OllamaUnreachableError:
            return _finish(_format_ollama_unreachable_help(), False, 0.0)
        except (OllamaHTTPError, OllamaRuntimeError, OllamaError) as exc:
            answer_text = (
                "**Ollama error.** The retrieved sources below still contain "
                "relevant text.\n\n"
                f"*Detail: {type(exc).__name__}: {str(exc)[:400]}*\n\n"
                f"Try: **`ollama pull {OLLAMA_MODEL}`**"
            )
            return _finish(answer_text, False, 0.0)
        return _finish(answer_text, True, (time.perf_counter() - t_gen) * 1000)

    model = _configure_gemini()
    if model is None:
        return _finish(
            "`LLM_PROVIDER=gemini` but `GOOGLE_API_KEY` is not set. "
            "Add it to `.env`, or use the free local model "
            "(`LLM_PROVIDER=ollama`).",
            False,
            0.0,
        )
    t_gen = time.perf_counter()
    try:
        response = model.generate_content(prompt)
        answer_text, _note = _safe_gemini_text(response)
        if not answer_text.strip():
            answer_text = NO_ANSWER_MESSAGE
    except Exception as exc:  # noqa: BLE001 - SDK raises generic exceptions
        return _finish(_format_gemini_user_error(exc), False, 0.0)
    return _finish(answer_text, True, (time.perf_counter() - t_gen) * 1000)


# --- Gemini error formatting (kept minimal; Ollama is primary) ----------


def _is_quota_or_rate_limit_error(exc: BaseException) -> bool:
    if google_api_exceptions and isinstance(exc, google_api_exceptions.ResourceExhausted):
        return True
    msg = str(exc).lower()
    return "429" in msg or "quota" in msg or "rate limit" in msg or "resource exhausted" in msg


_RETRY_AFTER_RE = re.compile(r"Please retry in ([0-9.]+)\s*s", re.IGNORECASE)


def _format_gemini_user_error(exc: BaseException) -> str:
    if _is_quota_or_rate_limit_error(exc):
        retry = _RETRY_AFTER_RE.search(str(exc))
        wait_hint = (
            f" The API suggested waiting about **{float(retry.group(1)):.0f}s**."
            if retry
            else ""
        )
        return (
            "**Gemini quota / rate limit (HTTP 429).** Wait a minute and retry, "
            "or switch `LLM_PROVIDER=ollama`." + wait_hint
        )
    return (
        "An error occurred while contacting the language model. "
        "The retrieved sources are still shown below.\n\n"
        f"*Detail: {type(exc).__name__}: {str(exc)[:400]}*"
    )
