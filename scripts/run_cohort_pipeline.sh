#!/usr/bin/env bash
# Run the three-pass HF readmission review pipeline on a cohort parquet.
#
# Usage:
#   ./scripts/run_cohort_pipeline.sh COHORT_PARQUET MODEL [OPTIONS]
#
# Examples:
#   ./scripts/run_cohort_pipeline.sh mimiciv_hf_30d_hf_readmit_extract.parquet Qwen/Qwen3-32B
#   COHORT=... MODEL=... BASE_URL=http://gpu:8000/v1 LIMIT=5 ./scripts/run_cohort_pipeline.sh
#
# Environment variables:
#   COHORT          - path to cohort parquet (or pass as arg 1)
#   MODEL           - vLLM served model id (or pass as arg 2)
#   BASE_URL        - OpenAI-compatible endpoint (default http://localhost:8000/v1)
#   OUT_ROOT        - output root (default AgenticHF/outputs)
#   COHORT_TAG      - output subdir tag (default derived from parquet basename)
#   WORKERS         - thread pool size (default 4)
#   LIMIT           - max rows to process, 0 = all (default 0)
#   ROW_IDS_FILE    - optional file of row_ids to process
#   MAX_CHARS_EACH  - truncate each note to N chars (0 = full text)
#   MAX_TOKENS      - completion token budget
#   NO_THINK_FIRST  - set to 1 for JSON-only mode
#   OVERWRITE       - set to 1 to wipe prior JSONL outputs
#   RETRY_ERRORS    - set to 1 to re-run failed rows
#   SKIP_PASS1/2/3  - set to 1 to skip a pass (resume workflow)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
READMIT_AGENT="${READMIT_AGENT:-/home/hmrirxk/readmission_agent}"

if [[ -x "${READMIT_AGENT}/.venv/bin/python" ]]; then
  PY="${READMIT_AGENT}/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

COHORT="${COHORT:-${1:-}}"
MODEL="${MODEL:-${2:-}}"
shift 2 2>/dev/null || true

: "${COHORT:?Set COHORT to cohort parquet path (arg 1)}"
: "${MODEL:?Set MODEL to served model id (arg 2)}"

if [[ ! -f "$COHORT" ]]; then
  if [[ -f "${REPO_ROOT}/${COHORT}" ]]; then
    COHORT="${REPO_ROOT}/${COHORT}"
  else
    echo "ERROR: cohort parquet not found: $COHORT" >&2
    exit 1
  fi
fi

BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs}"
WORKERS="${WORKERS:-4}"
LIMIT="${LIMIT:-0}"
ROW_IDS_FILE="${ROW_IDS_FILE:-}"
MAX_CHARS_EACH="${MAX_CHARS_EACH:-0}"
MAX_TOKENS="${MAX_TOKENS:-}"
NO_THINK_FIRST="${NO_THINK_FIRST:-0}"
OVERWRITE="${OVERWRITE:-0}"
RETRY_ERRORS="${RETRY_ERRORS:-0}"
SKIP_PASS1="${SKIP_PASS1:-0}"
SKIP_PASS2="${SKIP_PASS2:-0}"
SKIP_PASS3="${SKIP_PASS3:-0}"

if [[ -z "${COHORT_TAG:-}" ]]; then
  base="$(basename "$COHORT" .parquet)"
  if [[ "$base" == *30d* ]]; then
    COHORT_TAG="hf_30d"
  elif [[ "$base" == *180d* ]]; then
    COHORT_TAG="hf_180d"
  else
    COHORT_TAG="$base"
  fi
fi

# Filesystem-safe model slug ('-' kept at end of set so tr does not read it as a range)
MODEL_SLUG="$(echo "$MODEL" | tr '/:' '__' | tr -cd '[:alnum:]_.-')"
OUT_DIR="${OUT_ROOT}/${COHORT_TAG}/${MODEL_SLUG}"
mkdir -p "$OUT_DIR"

PASS1="${OUT_DIR}/pass1.jsonl"
PASS2="${OUT_DIR}/pass2.jsonl"
PASS3="${OUT_DIR}/pass3.jsonl"
MANIFEST="${OUT_DIR}/run_manifest.json"

export PYTHONPATH="${READMIT_AGENT}/src${PYTHONPATH:+:${PYTHONPATH}}"

common_args=()
if [[ -n "$ROW_IDS_FILE" ]]; then
  common_args+=(--row-ids-file "$ROW_IDS_FILE")
fi
if [[ "$LIMIT" != "0" ]]; then
  common_args+=(--limit "$LIMIT")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  common_args+=(--overwrite)
fi
if [[ "$RETRY_ERRORS" == "1" ]]; then
  common_args+=(--retry-errors)
fi

pass1_extra=()
pass2_extra=()
pass3_extra=()
if [[ "$NO_THINK_FIRST" == "1" ]]; then
  pass1_extra+=(--no-think-first)
  pass2_extra+=(--no-think-first)
  pass3_extra+=(--no-think-first)
fi
if [[ "$MAX_CHARS_EACH" != "0" ]]; then
  pass1_extra+=(--max-chars-each "$MAX_CHARS_EACH")
  pass3_extra+=(--max-chars-each "$MAX_CHARS_EACH")
fi
if [[ -n "$MAX_TOKENS" ]]; then
  pass1_extra+=(--max-tokens "$MAX_TOKENS")
  pass2_extra+=(--max-tokens "$MAX_TOKENS")
  pass3_extra+=(--max-tokens "$MAX_TOKENS")
fi

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "==> Cohort:     $COHORT"
echo "==> Model:      $MODEL"
echo "==> Base URL:   $BASE_URL"
echo "==> Out dir:    $OUT_DIR"
echo "==> Workers:    $WORKERS"
echo "==> Limit:      $LIMIT"
echo

_run_pass() {
  local name="$1"
  shift
  echo "==> Running $name ..."
  "$PY" -m "hf_readmission_review.$name" "$@"
}

if [[ "$SKIP_PASS1" != "1" ]]; then
  _run_pass run_hf_readmit_llm \
    --cohort-parquet "$COHORT" \
    --out-jsonl "$PASS1" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --workers "$WORKERS" \
    --temperature 0.2 \
    "${common_args[@]}" \
    "${pass1_extra[@]}"
fi

if [[ "$SKIP_PASS2" != "1" ]]; then
  _run_pass run_hf_readmit_bin_llm \
    --in-jsonl "$PASS1" \
    --out-jsonl "$PASS2" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --workers "$WORKERS" \
    --temperature 0.1 \
    "${common_args[@]}" \
    "${pass2_extra[@]}"
fi

if [[ "$SKIP_PASS3" != "1" ]]; then
  _run_pass run_hf_readmit_bin_verify_llm \
    --binned-jsonl "$PASS2" \
    --cohort-parquet "$COHORT" \
    --out-jsonl "$PASS3" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --workers "$WORKERS" \
    --temperature 0.1 \
    "${common_args[@]}" \
    "${pass3_extra[@]}"
fi

finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Write run manifest
"$PY" - <<'PY' "$MANIFEST" "$COHORT" "$MODEL" "$BASE_URL" "$OUT_DIR" "$started_at" "$finished_at" "$LIMIT"
import json, sys, os
from pathlib import Path

manifest_path, cohort, model, base_url, out_dir, started_at, finished_at, limit = sys.argv[1:9]

def count_jsonl(path):
    if not os.path.exists(path):
        return {"exists": False, "lines": 0, "errors": 0, "ok": 0}
    lines = errors = ok = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines += 1
            rec = json.loads(line)
            if rec.get("error"):
                errors += 1
            else:
                ok += 1
    return {"exists": True, "lines": lines, "errors": errors, "ok": ok}

manifest = {
    "cohort_parquet": cohort,
    "model": model,
    "base_url": base_url,
    "out_dir": out_dir,
    "started_at": started_at,
    "finished_at": finished_at,
    "limit": int(limit),
    "passes": {
        "pass1": count_jsonl(os.path.join(out_dir, "pass1.jsonl")),
        "pass2": count_jsonl(os.path.join(out_dir, "pass2.jsonl")),
        "pass3": count_jsonl(os.path.join(out_dir, "pass3.jsonl")),
    },
}
Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
PY

echo
echo "==> Done. Manifest: $MANIFEST"
