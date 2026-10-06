from __future__ import annotations

from typing import Any

import httpx


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def resolve_vllm_model_id(base_url: str, requested: str) -> str:
    """
    vLLM's OpenAI-compatible server often exposes a short model id (e.g. 'local-model')
    while reporting the actual checkpoint name under 'root'.
    """
    base_url = _normalize_base_url(base_url)
    url = f"{base_url}/models"
    try:
        r = httpx.get(url, timeout=10.0)
        r.raise_for_status()
        payload = r.json()
        data = payload.get("data", []) if isinstance(payload, dict) else []
        for m in data:
            if isinstance(m, dict) and m.get("id") == requested:
                return requested
        for m in data:
            if isinstance(m, dict) and m.get("root") == requested:
                mid = m.get("id")
                return str(mid) if mid else requested
    except Exception:
        pass
    return requested
