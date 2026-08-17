"""Call a local Ollama server (free, open-weight models — no cloud API key)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def ollama_chat(
    base_url: str,
    model: str,
    system_prompt: str,
    user_content: str,
    timeout_sec: float = 120.0,
    *,
    num_ctx: int | None = None,
    temperature: float = 0.1,
) -> str:
    """POST /api/chat (non-streaming). Returns assistant text or raises OllamaError."""
    options: dict[str, Any] = {"temperature": temperature}
    if num_ctx:
        options["num_ctx"] = int(num_ctx)

    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": options,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise OllamaHTTPError(exc.code, detail) from exc
    except urllib.error.URLError as exc:
        raise OllamaUnreachableError(str(exc.reason or exc)) from exc

    body = json.loads(raw)
    message = body.get("message") or {}
    content = (message.get("content") or "").strip()
    if not content and body.get("error"):
        raise OllamaRuntimeError(str(body["error"]))
    return content


class OllamaError(Exception):
    pass


class OllamaUnreachableError(OllamaError):
    """Cannot connect to Ollama (not running or wrong host/port)."""


class OllamaHTTPError(OllamaError):
    def __init__(self, code: int, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"HTTP {code}: {detail}")


class OllamaRuntimeError(OllamaError):
    """Ollama returned an error field (e.g. model not found)."""
