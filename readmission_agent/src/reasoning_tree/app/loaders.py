"""Loaders / adapters for input formats.

Supports two ingestion shapes:

1. The canonical CLI format described in the spec (``responses.json``):

   .. code-block:: json

       {"prompt": "...", "responses": [{"trajectory_id": "r1", "text": "..."}, ...]}

2. The principal-reasons JSONL produced by this repo. Each line has fields
   ``row_id``, ``thinking`` (long-form reasoning), and ``analysis`` (the
   structured prediction). The adapter selects one ``row_id`` and builds a
   trajectory from each JSONL file's ``thinking`` (preferred) or ``analysis``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .models import ResponsesPayload, TrajectoryInput


def load_responses_json(path: str) -> ResponsesPayload:
    with open(path, "r", encoding="utf-8") as f:
        return ResponsesPayload.model_validate(json.load(f))


def _record_to_text(rec: Dict[str, Any], field: str) -> str:
    """Extract the trajectory text from a principal-reasons JSONL record.

    `field` chooses what to use:
      - "thinking": long-form chain of thought (default)
      - "analysis": structured principal_reason + contributing_reasons
      - "content": the raw model content string
    """
    field = (field or "thinking").lower()
    if field == "thinking":
        for k in ("thinking", "raw"):
            v = rec.get(k)
            if isinstance(v, dict):
                v = v.get("thinking")
            if isinstance(v, str) and v.strip():
                return v
        return ""
    if field == "content":
        v = rec.get("raw")
        if isinstance(v, dict):
            v = v.get("content")
        if isinstance(v, str):
            return v
        return ""
    if field == "analysis":
        analysis = rec.get("analysis") or {}
        parts: List[str] = []
        principal = analysis.get("principal_reason")
        if isinstance(principal, str) and principal.strip():
            parts.append(f"Principal reason: {principal.strip()}")
        contributing = analysis.get("contributing_reasons") or []
        for i, c in enumerate(contributing, start=1):
            if isinstance(c, str) and c.strip():
                parts.append(f"Contributing reason {i}: {c.strip()}")
        return "\n".join(parts)
    raise ValueError(f"Unknown field {field!r}; expected one of: thinking, content, analysis")


def _record_to_conclusion(rec: Dict[str, Any], conclusion_field: str) -> Optional[str]:
    """Extract an explicit "answer" / "conclusion" text for one trajectory.

    Supported values for `conclusion_field`:
      * "none"               — do not append a conclusion step.
      * "principal_reason"   — pull ``analysis.principal_reason`` (default).
      * "principal+contributing" — combine principal + contributing reasons
                                   into one trailing step.
    """
    cf = (conclusion_field or "principal_reason").lower()
    if cf == "none":
        return None
    analysis = rec.get("analysis") or {}
    principal = analysis.get("principal_reason")
    if cf == "principal_reason":
        if isinstance(principal, str) and principal.strip():
            return principal.strip()
        return None
    if cf in ("principal+contributing", "principal_plus_contributing"):
        parts: List[str] = []
        if isinstance(principal, str) and principal.strip():
            parts.append(principal.strip())
        contributing = analysis.get("contributing_reasons") or []
        for c in contributing:
            if isinstance(c, str) and c.strip():
                parts.append(c.strip())
        return " | ".join(parts) if parts else None
    raise ValueError(
        f"Unknown conclusion_field {conclusion_field!r}; "
        f"expected: none, principal_reason, principal+contributing"
    )


def load_principal_reasons_jsonl_pair(
    jsonl_paths: List[str],
    row_id: int,
    field: str = "thinking",
    trajectory_labels: Optional[List[str]] = None,
    conclusion_field: str = "principal_reason",
) -> ResponsesPayload:
    """Build a ResponsesPayload by selecting `row_id` from each JSONL file.

    Parameters
    ----------
    jsonl_paths : list[str]
        Paths to principal-reasons JSONL files. Each will contribute one trajectory.
    row_id : int
        Row to extract from each file (must be present in all of them).
    field : str
        Which field within the JSONL record to treat as the trajectory text.
    trajectory_labels : optional list[str]
        Friendly labels (default: basename of each file). Length must match `jsonl_paths`.
    """
    if trajectory_labels is not None and len(trajectory_labels) != len(jsonl_paths):
        raise ValueError("trajectory_labels length must match jsonl_paths length")

    labels = trajectory_labels or [os.path.basename(p).replace(".jsonl", "") for p in jsonl_paths]
    prompt: Optional[str] = None
    responses: List[TrajectoryInput] = []

    for path, label in zip(jsonl_paths, labels):
        found = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("row_id") == row_id:
                    found = rec
                    break
        if found is None:
            raise ValueError(f"row_id {row_id} not found in {path}")
        text = _record_to_text(found, field)
        if not text.strip():
            raise ValueError(f"empty text for field={field!r} in row_id {row_id} of {path}")
        if prompt is None:
            pid = found.get("patient_identifier")
            if isinstance(pid, str):
                prompt = f"row_id={row_id} ({pid})"
        conclusion = _record_to_conclusion(found, conclusion_field)
        responses.append(TrajectoryInput(
            trajectory_id=label, text=text, conclusion_text=conclusion,
        ))

    return ResponsesPayload(prompt=prompt or f"row_id={row_id}", responses=responses)
