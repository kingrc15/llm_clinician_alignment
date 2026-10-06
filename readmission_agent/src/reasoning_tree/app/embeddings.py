"""Phase 3 — Embeddings.

Default model is ``Qwen/Qwen3-Embedding-8B`` (4096-dim, last-token-pooled
Qwen3 backbone trained for retrieval/STS). Embeddings are L2-normalized
so cosine similarity becomes a dot product downstream.

The 8B model is large; runs that re-embed the same canonical strings hit
the disk cache instead of recomputing. Cache key is
``(model_name, instruction, truncate_dim, canonical_text)``.

Optional features:

* ``instruction``: Qwen3-Embedding supports a per-text natural-language
  instruction prefix (e.g. "Represent this sentence for clustering."). For
  symmetric clustering we default to *no* instruction (treat all texts
  equivalently); set one explicitly to enable.
* ``truncate_dim``: Matryoshka truncation — keep only the leading ``d``
  dimensions of each embedding. Common choices: 256, 512, 1024, 2048,
  4096 (default = full).
* ``device``, ``model_kwargs``: forwarded to ``sentence-transformers``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from .models import ReasoningStep


_DEFAULT_MODEL = "Qwen/Qwen3-Embedding-8B"


def _cache_key(
    model_name: str,
    instruction: Optional[str],
    truncate_dim: Optional[int],
    text: str,
) -> str:
    payload = json.dumps(
        {
            "model": model_name,
            "instruction": instruction or "",
            "truncate_dim": truncate_dim or 0,
            "text": text or "",
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _DiskCache:
    """Lightweight JSONL-style on-disk dict[str, list[float]] cache."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._mem: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._mem = json.load(f)
            except Exception:
                self._mem = {}

    def get(self, key: str) -> Optional[List[float]]:
        return self._mem.get(key)

    def put(self, key: str, vec: List[float]) -> None:
        with self._lock:
            self._mem[key] = vec

    def flush(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._mem, f)
        os.replace(tmp, self.path)


class StepEmbedder:
    """Sentence-transformers wrapper with disk cache + optional truncation."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        instruction: Optional[str] = None,
        truncate_dim: Optional[int] = None,
        cache_path: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 4,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.instruction = instruction or None
        self.truncate_dim = int(truncate_dim) if truncate_dim else None
        self.cache_path = cache_path or None
        self.model_kwargs = dict(model_kwargs or {})
        self.encode_kwargs = dict(encode_kwargs or {})
        self.batch_size = max(1, int(batch_size))
        self._model = None  # lazy
        self._cache = _DiskCache(self.cache_path) if self.cache_path else None

    # ---- model lifecycle -------------------------------------------------
    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            kwargs: Dict[str, Any] = {}
            if self.device:
                kwargs["device"] = self.device
            if self.truncate_dim:
                kwargs["truncate_dim"] = self.truncate_dim
            if self.model_kwargs:
                kwargs["model_kwargs"] = self.model_kwargs
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    # ---- public API ------------------------------------------------------
    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        keys = [
            _cache_key(self.model_name, self.instruction, self.truncate_dim, t)
            for t in texts
        ]
        out: List[Optional[np.ndarray]] = [None] * len(texts)

        # Resolve cache hits first.
        miss_idx: List[int] = []
        miss_texts: List[str] = []
        if self._cache is not None:
            for i, k in enumerate(keys):
                hit = self._cache.get(k)
                if hit is not None:
                    out[i] = np.asarray(hit, dtype=np.float32)
                else:
                    miss_idx.append(i)
                    miss_texts.append(texts[i])
        else:
            miss_idx = list(range(len(texts)))
            miss_texts = list(texts)

        if miss_texts:
            model = self._ensure_model()
            encode_kwargs: Dict[str, Any] = dict(self.encode_kwargs)
            encode_kwargs.setdefault("convert_to_numpy", True)
            encode_kwargs.setdefault("normalize_embeddings", normalize)
            encode_kwargs.setdefault("show_progress_bar", False)
            encode_kwargs.setdefault("batch_size", self.batch_size)
            if self.instruction:
                # SentenceTransformer >= 3 accepts a `prompt` kwarg that is
                # prepended to every text before tokenisation.
                encode_kwargs.setdefault("prompt", self.instruction)
            embs = model.encode(miss_texts, **encode_kwargs)
            embs = np.asarray(embs, dtype=np.float32)
            for j, i in enumerate(miss_idx):
                vec = embs[j]
                out[i] = vec
                if self._cache is not None:
                    self._cache.put(keys[i], vec.tolist())
            if self._cache is not None:
                self._cache.flush()

        # Stack in original order (every slot is filled by now).
        arr = np.stack([np.asarray(v, dtype=np.float32) for v in out], axis=0)
        return arr


def embed_steps(
    steps: List[ReasoningStep],
    embedder: Optional[StepEmbedder] = None,
    use_canonical: bool = True,
) -> np.ndarray:
    """Embed each step and write the result back to ``step.embedding``.

    Embeddings are L2-normalised so cosine similarity collapses to a dot
    product, which is what the clustering phase assumes.
    """
    if not steps:
        return np.zeros((0, 1), dtype=np.float32)

    embedder = embedder or StepEmbedder()
    texts = [
        (s.canonical_text if use_canonical and s.canonical_text else s.raw_text)
        for s in steps
    ]
    embs = embedder.encode(texts, normalize=True)
    for s, vec in zip(steps, embs):
        s.embedding = vec.astype(np.float32).tolist()
    return embs
