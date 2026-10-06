#!/usr/bin/env bash
# Build an interactive reasoning graph (HTML + JSON) from pass-1 analysis JSONL.
#
# Typical use after run_hf_readmit_review.sh (single row_id 0):
#   PASS1_JSONL=outputs/review_run/review_pass1_analysis.jsonl \
#   OUT_DIR=outputs/review_graph \
#   ./scripts/build_reasoning_graph.sh
#
# Environment variables:
#   PASS1_JSONL   — required: path to *_pass1_analysis.jsonl
#   OUT_DIR       — default: ./outputs/reasoning_graph
#   ROW_ID        — default 0 (must exist in JSONL)
#   FIELD         — thinking | content | analysis (default: thinking)
#   SIM_THRESHOLD — default 0.83
#   EMBED_MODEL   — default Qwen/Qwen3-Embedding-8B (needs GPU RAM or pick a smaller model)
#
# Install graph dependencies: pip install -e ".[graph]"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PY="${REPO_ROOT}/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

: "${PASS1_JSONL:?Set PASS1_JSONL to review_pass1_analysis.jsonl from the review step}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/reasoning_graph}"
ROW_ID="${ROW_ID:-0}"
FIELD="${FIELD:-thinking}"
SIM_THRESHOLD="${SIM_THRESHOLD:-0.83}"
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "==> Pass1 JSONL: $PASS1_JSONL"
echo "==> Out dir:     $OUT_DIR"
echo "==> row-id:      $ROW_ID  field: $FIELD"
echo

exec "$PY" -m reasoning_tree.app.main \
  --from-jsonl "$PASS1_JSONL" \
  --row-id "$ROW_ID" \
  --field "$FIELD" \
  --out-dir "$OUT_DIR" \
  --similarity-threshold "$SIM_THRESHOLD" \
  --embedding-model "$EMBED_MODEL"
