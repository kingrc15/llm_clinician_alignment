#!/usr/bin/env bash
# Submit quickgpuq pipeline jobs WITHOUT cancelling existing gpuq submissions.
# Safe to run alongside gpuq jobs for the same model: whichever starts first
# writes outputs; the other resumes/skips completed rows (JSONL append + skip).
#
# Usage:
#   ./scripts/submit_quickgpu_jobs.sh
#   ./scripts/submit_quickgpu_jobs.sh BioMistral/BioMistral-7B   # pass3 only

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=model_catalog.sh
source "${REPO}/scripts/model_catalog.sh"

# Default: models pending on gpuq that also fit 1×H100 — quickgpuq may dequeue faster.
QUICKGPU_MODELS=(
  "google/medgemma-27b-text-it"
  "openai/gpt-oss-120b"
)

if [[ $# -gt 0 ]]; then
  QUICKGPU_MODELS=("$@")
fi

COHORT="${COHORT:-30d}"
ROW_IDS_FILE="${ROW_IDS_FILE:-}"
MANIFEST="${REPO}/outputs/pipeline_jobs_quickgpu.json"
mkdir -p "${REPO}/outputs"
SUBMITTED=()

HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/notes.txt}"
if [[ -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(grep -m1 -oE 'hf_[A-Za-z0-9]+' "$HF_TOKEN_FILE" || true)"
  export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

for model in "${QUICKGPU_MODELS[@]}"; do
  resolve_model_plan "$model"
  workers="${WORKERS:-2}"

  extra_export="RETRY_ERRORS=1"
  # BioMistral: pass1 ok; pass2 had validation errors — retry pass2 then pass3
  if [[ "$model" == "BioMistral/BioMistral-7B" ]]; then
    extra_export="RETRY_ERRORS=1,SKIP_PASS1=1"
  fi

  row_ids_export=""
  if [[ -n "$ROW_IDS_FILE" ]]; then
    row_ids_export=",ROW_IDS_FILE=${ROW_IDS_FILE}"
  fi

  echo "Submitting quickgpuq: $model (cohort=$COHORT workers=$workers mem=${MEM_GB}G row_ids=${ROW_IDS_FILE:-all})"
  out="$(sbatch \
    --partition=quickgpuq \
    --gres=gpu:${NUM_GPUS} \
    --cpus-per-task=${CPUS} \
    --mem=${MEM_GB}G \
    --time=${TIME_LIMIT} \
    --job-name="qgpu_$(echo "$model" | awk -F/ '{print $NF}' | cut -c1-16)" \
    --export="ALL,MODEL=${model},COHORT=${COHORT},LIMIT=${LIMIT:-0},WORKERS=${workers},${extra_export}${row_ids_export}" \
    "${REPO}/scripts/slurm_full_run.sh")"
  echo "  $out"
  job_id="$(echo "$out" | awk '{print $NF}')"
  SUBMITTED+=("{\"job_id\":${job_id},\"model\":\"${model}\",\"partition\":\"quickgpuq\"}")
done

{
  echo "{"
  echo "  \"submitted_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"note\": \"Does not cancel existing gpuq jobs. Parallel jobs for same model resume via JSONL skip.\","
  echo "  \"jobs\": ["
  for i in "${!SUBMITTED[@]}"; do
    sep=","
    [[ $i -eq $((${#SUBMITTED[@]} - 1)) ]] && sep=""
    echo "    ${SUBMITTED[$i]}${sep}"
  done
  echo "  ]"
  echo "}"
} > "$MANIFEST"

echo
echo "Quickgpuq jobs submitted. Manifest: $MANIFEST"
echo "Existing gpuq jobs left unchanged. Monitor: squeue -u \$USER"
