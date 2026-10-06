"""Small, dependency-light utilities shared across modules."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any


_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_whitespace(text: str) -> str:
    """Collapse all runs of whitespace into single spaces; trim ends."""
    return _WS_RE.sub(" ", text or "").strip()


def stable_id(*parts: Any, length: int = 12) -> str:
    """Deterministic short id from parts (used for steps/nodes)."""
    payload = "||".join("" if p is None else str(p) for p in parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:length]


def ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: Any) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
