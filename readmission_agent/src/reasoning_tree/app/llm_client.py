"""LLM-assisted canonicalization via an OpenAI-compatible endpoint (e.g. vLLM).

Talks to a chat-completions endpoint exposed by vLLM at, by default,
``http://localhost:8000/v1``. Used as the implementation of
``canonicalization`` mode ``llm``.

Features:

* Disk cache (one JSON file per (base_url, model)) so reruns are free.
* Thread-pool concurrency for batch canonicalization.
* Model-id resolution (vLLM often advertises a short id while reporting the
  HF checkpoint under ``root`` — we accept either).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import httpx
from openai import OpenAI


_PROMPT_VERSION = "v1"

_SYSTEM_PROMPT = (
    "You normalize reasoning steps into a single concise canonical proposition.\n\n"
    "Rules:\n"
    "- Output exactly ONE short proposition. No bullets, no JSON, no preamble.\n"
    "- Use lowercase. No surrounding quotes.\n"
    "- Subject-verb-object form when possible.\n"
    "- Expand clinical acronyms (HF -> heart failure, MI -> myocardial infarction, "
    "CHF -> heart failure, COPD -> chronic obstructive pulmonary disease, "
    "CKD -> chronic kidney disease, AKI -> acute kidney injury, "
    "AFib -> atrial fibrillation, EF -> ejection fraction, BNP -> brain natriuretic peptide, "
    "DNR -> do not resuscitate, CMO -> comfort measures only, ICU -> intensive care unit, "
    "SNF -> skilled nursing facility).\n"
    "- Remove filler words (\"I think\", \"of course\", \"importantly\"), markdown emphasis, "
    "and parentheticals.\n"
    "- Drop generic subjects (\"the patient\", \"this patient\", \"he\", \"she\").\n"
    "- Drop pure section headers (e.g. \"Synthesis:\", \"Confidence:\", \"Step 1:\"). "
    "If the input is only such a header, respond with: <NONE>\n"
    "- Preserve clinically meaningful content (specific medications, conditions, "
    "anatomical sites, time references that change meaning).\n"
    "- Keep the output under 25 words.\n"
)

_USER_TEMPLATE = (
    "Reasoning step:\n"
    "{text}\n\n"
    "Canonical proposition:"
)


def _normalize_base_url(base_url: str) -> str:
    return (base_url or "http://localhost:8000/v1").rstrip("/")


def resolve_vllm_model_id(base_url: str, requested: str) -> str:
    """Resolve a friendly model name to whatever the vLLM server reports.

    vLLM commonly exposes a short id like ``local-model`` while reporting the
    real checkpoint under ``root`` — this lets the caller pass either.
    """
    base_url = _normalize_base_url(base_url)
    try:
        r = httpx.get(f"{base_url}/models", timeout=10.0)
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


_OUTPUT_CLEANUP_RE = re.compile(r"^[\"'`\*]+|[\"'`\*]+$")
_LEADING_LABEL_RE = re.compile(
    r"^(?:canonical\s*proposition\s*:|proposition\s*:|output\s*:|answer\s*:)\s*",
    flags=re.IGNORECASE,
)


class LLMCanonicalizer:
    """Canonicalize reasoning steps via an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "local-model",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout_s: float = 60.0,
        cache_path: Optional[str] = None,
        workers: int = 8,
        disable_thinking: bool = True,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.requested_model = model
        self.api_key = api_key
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_s = float(timeout_s)
        self.cache_path = cache_path
        self.workers = max(1, int(workers))
        self.disable_thinking = bool(disable_thinking)

        self._model_id: Optional[str] = None
        self._client: Optional[OpenAI] = None
        self._cache_lock = threading.Lock()
        self._cache: Dict[str, str] = self._load_cache()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def model_id(self) -> str:
        if self._model_id is None:
            self._model_id = resolve_vllm_model_id(self.base_url, self.requested_model)
        return self._model_id

    def canonicalize(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        key = self._cache_key(text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        rewritten = self._call(text)
        with self._cache_lock:
            self._cache[key] = rewritten
        self._save_cache()
        return rewritten

    def canonicalize_batch(self, texts: List[str]) -> List[str]:
        """Concurrent canonicalization with the configured worker count."""
        results: List[str] = ["" for _ in texts]
        todo: List[int] = []
        for i, t in enumerate(texts):
            if not t or not t.strip():
                continue
            key = self._cache_key(t)
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                todo.append(i)

        if not todo:
            return results

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            future_to_idx = {ex.submit(self._call, texts[i]): i for i in todo}
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                try:
                    out = fut.result()
                except Exception:
                    out = ""
                results[i] = out
                with self._cache_lock:
                    self._cache[self._cache_key(texts[i])] = out
        self._save_cache()
        return results

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_s)
        return self._client

    def _call(self, text: str) -> str:
        client = self._ensure_client()
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(text=text.strip())},
        ]
        extra: Dict[str, object] = {}
        if self.disable_thinking:
            # vLLM/Qwen3 chat template knob; silently ignored by models that don't support it.
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            resp = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body=extra or None,
            )
            msg = resp.choices[0].message
            content = (getattr(msg, "content", None) or "").strip()
            # If we still got nothing (e.g. thinking model burned the budget),
            # try to pull the model's final line out of the `reasoning` field.
            if not content:
                reasoning = getattr(msg, "reasoning", None)
                if reasoning:
                    last_line = reasoning.strip().splitlines()[-1].strip()
                    if last_line:
                        content = last_line
        except Exception:
            return ""
        return self._postprocess(content)

    @staticmethod
    def _postprocess(text: str) -> str:
        if not text:
            return ""
        # The model occasionally returns "<NONE>" for pure headings.
        if text.strip().upper() in {"<NONE>", "NONE"}:
            return ""
        line = text.splitlines()[0].strip()
        line = _LEADING_LABEL_RE.sub("", line)
        line = _OUTPUT_CLEANUP_RE.sub("", line)
        return line.lower().strip()

    def _cache_key(self, text: str) -> str:
        material = f"{_PROMPT_VERSION}|{self.base_url}|{self.requested_model}|{text}".encode("utf-8")
        return hashlib.sha1(material).hexdigest()

    def _load_cache(self) -> Dict[str, str]:
        if not self.cache_path or not os.path.exists(self.cache_path):
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)) or ".", exist_ok=True)
        tmp = self.cache_path + ".tmp"
        with self._cache_lock:
            data = dict(self._cache)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.cache_path)
