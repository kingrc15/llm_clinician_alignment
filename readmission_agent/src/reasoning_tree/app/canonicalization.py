"""Phase 2 — Canonicalization.

Rule-based normalization. The contract is intentionally narrow so a
Mode B (LLM-assisted) implementation can drop in later.

Two modes:

* ``simple``     — conservative: lowercase, drop filler/markdown, normalize space.
* ``aggressive`` — everything ``simple`` does, plus:
    - drop trailing "Synthesis:", "Confidence:", "Step N:" headings
    - drop generic clinical subjects ("the patient", "this patient", etc.)
    - drop temporal hedges ("recently", "currently", "previously", "now")
    - expand common cardiology acronyms (HF -> heart failure, CHF -> heart failure,
      MI -> myocardial infarction, COPD -> chronic obstructive pulmonary disease, etc.)
    - strip parenthetical asides
    - collapse double spaces / punctuation runs
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from .utils import normalize_whitespace

# Conservative list of "filler" / hedging phrases to drop. We deliberately
# keep clinical content words (e.g. "suggests", "indicates") because they
# carry meaning — only true filler is removed.
_FILLER_PATTERNS = [
    r"\bnote(?:d)? that\b",
    r"\bplease note\b",
    r"\blet'?s\b",
    r"\bok(?:ay)?,?\b",
    r"\bso,?\b",
    r"\bnow,?\b",
    r"\bin (?:my|our) (?:opinion|view)\b",
    r"\bi think\b",
    r"\bi believe\b",
    r"\bobviously\b",
    r"\bbasically\b",
    r"\bessentially\b",
    r"\bof course\b",
    r"\bas (?:you can )?see\b",
    r"\bclearly\b",
    r"\bsimply put\b",
    r"\bto be clear\b",
    r"\bfor (?:the )?reader\b",
]
_FILLER_RE = re.compile("|".join(_FILLER_PATTERNS), flags=re.IGNORECASE)

# Markdown emphasis & inline code (drop the formatting, keep the words).
_MD_EMPHASIS_RE = re.compile(r"[*_`]{1,3}")

# Section header colons we keep for readability but normalize whitespace.
_HEADING_PREFIX_RE = re.compile(r"^\s*[A-Za-z][A-Za-z0-9 \-/]{0,50}:\s+")

# Trailing punctuation we strip so "X." and "X" cluster together.
_TRAILING_PUNCT_RE = re.compile(r"[\.\!\?;,:]+\s*$")


def canonicalize_step(step_text: str) -> str:
    """Mode A canonicalizer. Returns a lowercased, de-fillered string."""
    if not step_text:
        return ""
    s = _MD_EMPHASIS_RE.sub("", step_text)
    s = _HEADING_PREFIX_RE.sub("", s)
    s = _FILLER_RE.sub("", s)
    s = normalize_whitespace(s)
    s = _TRAILING_PUNCT_RE.sub("", s)
    return s.lower().strip()


# --- Aggressive mode ------------------------------------------------------- #

# Standalone heading-like steps to drop entirely (e.g. "Synthesis:", "Confidence").
_HEADING_DROP_PATTERNS = [
    r"^synthesis$",
    r"^confidence$",
    r"^confidence calibration$",
    r"^clinical picture$",
    r"^analysis$",
    r"^assessment$",
    r"^plan$",
    r"^summary$",
    r"^reasoning$",
    r"^step\s*\d+$",
    r"^reason(?:ing)?\s*\d*$",
    r"^contributing factor[s]?$",
    r"^principal reason$",
    r"^contributing reason[s]?$",
    r"^root cause[s]?(?:\s*&?\s*contributing factor[s]?)?$",
]
_HEADING_DROP_RE = re.compile("|".join(_HEADING_DROP_PATTERNS), flags=re.IGNORECASE)

# Subject phrases that add no semantic content for our purpose. We drop them
# from the start of canonical text so that "the patient has X" merges with
# "patient has X" merges with "this patient has X".
_SUBJECT_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"the\s+patient(?:'s)?"
    r"|this\s+patient(?:'s)?"
    r"|the\s+pt\b"
    r"|pt\b"
    r"|patient(?:'s)?"
    r"|she|he|they|it"
    r")\s+",
    flags=re.IGNORECASE,
)

# Temporal/hedging words that don't change meaning meaningfully for clustering.
_TEMPORAL_HEDGE_RE = re.compile(
    r"\b(?:recently|previously|currently|now|then|"
    r"likely|presumably|apparently|reportedly|"
    r"importantly|notably|of note|essentially)\b",
    flags=re.IGNORECASE,
)

# Strip parenthetical asides ("(see below)", "(BNP 6240)", etc).
_PARENTHETICAL_RE = re.compile(r"\s*\([^()]{1,200}\)")

# Expand common cardiology acronyms so abbreviation/full-form variants cluster.
_ACRONYM_MAP = {
    r"\bchf\b": "heart failure",
    r"\bhf\b": "heart failure",
    r"\bhfref\b": "heart failure reduced ejection fraction",
    r"\bhfpef\b": "heart failure preserved ejection fraction",
    r"\bmi\b": "myocardial infarction",
    r"\bcad\b": "coronary artery disease",
    r"\bcabg\b": "coronary artery bypass graft",
    r"\bcopd\b": "chronic obstructive pulmonary disease",
    r"\bckd\b": "chronic kidney disease",
    r"\baki\b": "acute kidney injury",
    r"\barf\b": "acute renal failure",
    r"\bafib\b": "atrial fibrillation",
    r"\baf\b": "atrial fibrillation",
    r"\brvr\b": "rapid ventricular response",
    r"\bdm\b": "diabetes mellitus",
    r"\bt2dm\b": "type 2 diabetes mellitus",
    r"\bsob\b": "shortness of breath",
    r"\bdoe\b": "dyspnea on exertion",
    r"\baki\b": "acute kidney injury",
    r"\bcva\b": "cerebrovascular accident",
    r"\bdvt\b": "deep vein thrombosis",
    r"\bpe\b": "pulmonary embolism",
    r"\bvte\b": "venous thromboembolism",
    r"\biv\b": "intravenous",
    r"\bnyha\b": "new york heart association",
    r"\blvef\b": "left ventricular ejection fraction",
    r"\bef\b": "ejection fraction",
    r"\bbnp\b": "brain natriuretic peptide",
    r"\bvap\b": "ventilator associated pneumonia",
    r"\bvte\b": "venous thromboembolism",
    r"\bicu\b": "intensive care unit",
    r"\bsnf\b": "skilled nursing facility",
    r"\baicd\b": "automatic implantable cardioverter defibrillator",
    r"\bicd\b": "implantable cardioverter defibrillator",
    r"\bppm\b": "permanent pacemaker",
    r"\baf\b": "atrial fibrillation",
    r"\bpvd\b": "peripheral vascular disease",
    r"\bpad\b": "peripheral artery disease",
    r"\bgoc\b": "goals of care",
    r"\bdnr\b": "do not resuscitate",
    r"\bdni\b": "do not intubate",
    r"\bcmo\b": "comfort measures only",
}
_ACRONYM_RES = [(re.compile(pat, flags=re.IGNORECASE), repl) for pat, repl in _ACRONYM_MAP.items()]

# Generic determiners / articles to drop after the other rewrites have run.
_LEADING_DET_RE = re.compile(r"^(?:a|an|the|this|that|these|those)\s+", flags=re.IGNORECASE)


def canonicalize_step_aggressive(step_text: str) -> str:
    """More aggressive canonicalization. Returns empty string for heading-only steps.

    The empty-string convention lets callers easily filter trivial steps before
    embedding/clustering.
    """
    if not step_text:
        return ""
    simple = canonicalize_step(step_text)
    if not simple:
        return ""

    if _HEADING_DROP_RE.match(simple):
        return ""

    s = _PARENTHETICAL_RE.sub("", simple)
    s = _SUBJECT_PREFIX_RE.sub("", s)
    s = _TEMPORAL_HEDGE_RE.sub("", s)

    for pat, repl in _ACRONYM_RES:
        s = pat.sub(repl, s)

    s = _LEADING_DET_RE.sub("", s)
    s = normalize_whitespace(s)
    s = _TRAILING_PUNCT_RE.sub("", s)
    return s.strip()


CanonicalizerFn = Callable[[str], str]


def make_llm_canonicalizer(
    base_url: str = "http://localhost:8000/v1",
    model: str = "local-model",
    temperature: float = 0.0,
    max_tokens: int = 256,
    cache_path: Optional[str] = None,
    workers: int = 8,
    disable_thinking: bool = True,
) -> "LLMCanonicalizer":
    """Construct an LLM-backed canonicalizer (Mode B). Lazy import to keep
    the rule-based path free of the openai/httpx dependency surface."""
    from .llm_client import LLMCanonicalizer  # noqa: WPS433
    return LLMCanonicalizer(
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        cache_path=cache_path,
        workers=workers,
        disable_thinking=disable_thinking,
    )


def get_canonicalizer(mode: str = "simple", **kwargs) -> CanonicalizerFn:
    """Factory so callers can swap modes by name.

    For ``mode='llm'`` you may pass ``base_url``, ``model``, ``temperature``,
    ``max_tokens``, ``cache_path``, and ``workers`` as kwargs. The returned
    callable runs one step per call; for many steps prefer
    :func:`make_llm_canonicalizer` directly so you can batch via
    ``LLMCanonicalizer.canonicalize_batch``.
    """
    mode = (mode or "simple").lower()
    if mode in {"simple", "a", "rule"}:
        return canonicalize_step
    if mode in {"aggressive", "agg", "strong"}:
        return canonicalize_step_aggressive
    if mode in {"llm", "b", "vllm"}:
        client = make_llm_canonicalizer(**kwargs)
        return client.canonicalize
    raise ValueError(
        f"Unknown canonicalization mode {mode!r}. "
        "Available: 'simple', 'aggressive', 'llm'."
    )
