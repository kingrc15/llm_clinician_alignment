#!/usr/bin/env bash
# Run the HF readmission pipeline for all default medical LLMs on both cohorts.
#
# Usage:
#   BASE_URL=http://gpu-host:8000/v1 ./scripts/run_all_models.sh
#   LIMIT=5 ./scripts/run_all_models.sh   # smoke test
#
# Models are run sequentially (one vLLM server at a time). Start vLLM with each
# model before invoking, or use a multi-model router that honors --model.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE="${REPO_ROOT}/scripts/run_cohort_pipeline.sh"

MODELS=(
  "google/medgemma-27b-text-it"
  "openai/gpt-oss-120b"
  "mistralai/Mistral-Large-3-675B-Instruct-2512"
  "deepseek-ai/DeepSeek-V3.1"
)

COHORTS=(
  "${REPO_ROOT}/readmit_30d.parquet"
)

BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
LIMIT="${LIMIT:-0}"
WORKERS="${WORKERS:-4}"

for model in "${MODELS[@]}"; do
  for cohort in "${COHORTS[@]}"; do
    echo "========================================"
    echo "Model:  $model"
    echo "Cohort: $cohort"
    echo "========================================"
    COHORT="$cohort" MODEL="$model" BASE_URL="$BASE_URL" LIMIT="$LIMIT" WORKERS="$WORKERS" \
      bash "$PIPELINE"
  done
done

echo "All model × cohort runs complete."
