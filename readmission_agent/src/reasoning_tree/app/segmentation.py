"""Phase 1 — Segmentation.

Convert a raw LLM response into an ordered list of atomic reasoning steps
using sentence + newline + bullet splitting. Designed to be swappable: any
function with the signature `segment_response(text, trajectory_id)` may
replace this one (e.g. an LLM-assisted decomposer).
"""

from __future__ import annotations

import re
from typing import List

from .models import ReasoningStep
from .utils import normalize_whitespace, stable_id

# Bullet leaders we strip from segment starts so equal-meaning steps merge.
_BULLET_RE = re.compile(
    r"""^(
        [-*\u2022\u25E6\u25CF\u2023\u2043]          # - * • ◦ ● ‣ ⁃
        | \(?\d+[\.\)]                              # 1. 1) (1)
        | [a-zA-Z][\.\)]                            # a. a)
        | \(?[ivxlcdmIVXLCDM]+[\.\)]                # roman numerals
        | \#+                                       # markdown headers
    )\s*""",
    flags=re.VERBOSE | re.UNICODE,
)

_THINKING_TAG_RE = re.compile(r"</?\s*thinking\s*>", flags=re.IGNORECASE)
_ANSWER_BLOCK_RE = re.compile(r"<\s*answer\s*>.*?<\s*/\s*answer\s*>", flags=re.IGNORECASE | re.DOTALL)
_JSON_BLOCK_RE = re.compile(r"^\s*\{[\s\S]*\}\s*$", flags=re.MULTILINE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+(?=[A-Z\u00C0-\u017F\(\[\u201C\"\d])")

# A floor on segment length — anything shorter is fused into the next segment.
_MIN_CHARS = 8


def _strip_meta(text: str) -> str:
    """Remove obvious framing tokens that aren't reasoning steps."""
    text = _THINKING_TAG_RE.sub("\n", text or "")
    text = _ANSWER_BLOCK_RE.sub("\n", text)
    # Drop trailing JSON object that some prompts produce alongside <thinking>.
    text = _JSON_BLOCK_RE.sub("\n", text)
    return text


def _strip_bullet(line: str) -> str:
    return _BULLET_RE.sub("", line, count=1).strip()


def _split_sentences(block: str) -> List[str]:
    parts = _SENTENCE_SPLIT_RE.split(block.strip())
    return [p.strip() for p in parts if p.strip()]


def _coalesce_short(steps: List[str]) -> List[str]:
    """Glue together tiny fragments (e.g. a stray '1.') with the next step."""
    out: List[str] = []
    buf = ""
    for s in steps:
        if len(s) < _MIN_CHARS and not buf:
            buf = s
            continue
        if buf:
            out.append(normalize_whitespace(f"{buf} {s}"))
            buf = ""
        else:
            out.append(s)
    if buf:
        out.append(buf)
    return out


def segment_text(text: str) -> List[str]:
    """Pure string-in/list-out segmentation (no IDs)."""
    cleaned = _strip_meta(text or "")
    raw_lines = [ln for ln in cleaned.splitlines()]

    chunks: List[str] = []
    for line in raw_lines:
        line = _strip_bullet(line)
        line = normalize_whitespace(line)
        if not line:
            continue
        for sent in _split_sentences(line):
            sent = normalize_whitespace(sent)
            if sent:
                chunks.append(sent)

    return _coalesce_short(chunks)


def segment_response(text: str, trajectory_id: str) -> List[ReasoningStep]:
    """Public API. Returns ordered ReasoningStep objects with stable ids."""
    steps: List[ReasoningStep] = []
    for i, raw in enumerate(segment_text(text)):
        step_id = stable_id("step", trajectory_id, i, raw)
        steps.append(
            ReasoningStep(
                id=step_id,
                trajectory_id=trajectory_id,
                step_index=i,
                raw_text=raw,
            )
        )
    return steps
