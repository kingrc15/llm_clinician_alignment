#!/usr/bin/env bash
# Run the three-pass HF readmission review from two plain-text discharge summaries.
#
# Prerequisites:
#   - Python env with this repo installed: pip install -e ".[graph]"  (graph extra optional for review only)
#   - vLLM (or any OpenAI-compatible server) reachable at BASE_URL
#
# Usage (positional); discharge paths may be absolute or relative to this repo
# or to the directory from which you invoke the script:
#   ./scripts/run_hf_readmit_review.sh INDEX_DISCHARGE.txt READMIT_DISCHARGE.txt MODEL_ID
#
# Usage (environment variables — same names work):
#   INDEX_DISCHARGE=... READMIT_DISCHARGE=... MODEL=local-model ./scripts/run_hf_readmit_review.sh
#
# Optional files / variables:
#   READMIT_ADMISSION_NOTE  — path to optional admission/H&P text
#   OUT_DIR                 — output directory (default: ./outputs/review_run)
#   BASE_URL                — default http://localhost:8000/v1
#   NO_THINK_FIRST=1        — JSON-only mode (no <thinking> before JSON)
#   OVERWRITE=1             — replace prior JSONL outputs in OUT_DIR
#   MAX_CHARS_EACH=N        — truncate each note to N characters (0 = full text)
#   MAX_TOKENS=N            — forward to all three LLM passes

set -euo pipefail

# Directory you ran the script from (before we cd to the repo).
CALLER_PWD="$(pwd)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PY="${REPO_ROOT}/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

# Resolve a note path: as given, then under caller cwd, then under repo root.
_resolve_note_file() {
  local label="$1" path="$2"
  if [[ -f "$path" ]]; then
    printf '%s' "$(cd "$(dirname "$path")" && pwd)/$(basename "$path")"
    return 0
  fi
  if [[ -f "${CALLER_PWD}/${path}" ]]; then
    printf '%s' "$(cd "$(dirname "${CALLER_PWD}/${path}")" && pwd)/$(basename "${CALLER_PWD}/${path}")"
    return 0
  fi
  if [[ -f "${REPO_ROOT}/${path}" ]]; then
    printf '%s' "$(cd "$(dirname "${REPO_ROOT}/${path}")" && pwd)/$(basename "${REPO_ROOT}/${path}")"
    return 0
  fi
  echo "ERROR: ${label} file not found: ${path}" >&2
  echo "  Tried: ${path}" >&2
  echo "  Tried: ${CALLER_PWD}/${path}" >&2
  echo "  Tried: ${REPO_ROOT}/${path}" >&2
  exit 1
}

if [[ $# -ge 3 ]]; then
  INDEX_DISCHARGE="${1:?}"
  READMIT_DISCHARGE="${2:?}"
  MODEL="${3:?}"
  shift 3
fi

: "${INDEX_DISCHARGE:?Set INDEX_DISCHARGE to the index discharge summary .txt path (arg 1)}"
: "${READMIT_DISCHARGE:?Set READMIT_DISCHARGE to the readmit discharge summary .txt path (arg 2)}"
: "${MODEL:?Set MODEL to the served model id (arg 3 or env)}"

INDEX_DISCHARGE="$(_resolve_note_file "INDEX_DISCHARGE" "${INDEX_DISCHARGE}")"
READMIT_DISCHARGE="$(_resolve_note_file "READMIT_DISCHARGE" "${READMIT_DISCHARGE}")"
if [[ -n "${READMIT_ADMISSION_NOTE:-}" ]]; then
  READMIT_ADMISSION_NOTE="$(_resolve_note_file "READMIT_ADMISSION_NOTE" "${READMIT_ADMISSION_NOTE}")"
fi

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/review_run}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"

extra=()
if [[ -n "${READMIT_ADMISSION_NOTE:-}" ]]; then
  extra+=(--readmit-admission-note "${READMIT_ADMISSION_NOTE}")
fi
if [[ "${NO_THINK_FIRST:-0}" == "1" ]]; then
  extra+=(--no-think-first)
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  extra+=(--overwrite)
fi
if [[ -n "${MAX_CHARS_EACH:-}" && "${MAX_CHARS_EACH}" != "0" ]]; then
  extra+=(--max-chars-each "${MAX_CHARS_EACH}")
fi
if [[ -n "${MAX_TOKENS:-}" ]]; then
  extra+=(--max-tokens "${MAX_TOKENS}")
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "==> Repo:       $REPO_ROOT"
echo "==> Python:     $PY"
echo "==> Out dir:    $OUT_DIR"
echo "==> Model:      $MODEL"
echo "==> Base URL:   $BASE_URL"
echo

exec "$PY" -m hf_readmission_review.run_pair_review \
  --index-discharge "$INDEX_DISCHARGE" \
  --readmit-discharge "$READMIT_DISCHARGE" \
  "${extra[@]}" \
  --out-dir "$OUT_DIR" \
  --model "$MODEL" \
  --base-url "$BASE_URL"
