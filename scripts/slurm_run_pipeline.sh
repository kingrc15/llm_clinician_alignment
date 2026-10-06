#!/usr/bin/env bash
#SBATCH --job-name=hf_readmit
#SBATCH --output=/home/hmrirxk/AgenticHF/outputs/slurm_%j.out
#SBATCH --error=/home/hmrirxk/AgenticHF/outputs/slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#
# Submit example:
#   MODEL=Qwen/Qwen3-32B COHORT=30d LIMIT=0 sbatch scripts/slurm_run_pipeline.sh
#   MODEL=Qwen/Qwen3-32B COHORT=30d LIMIT=5 sbatch scripts/slurm_run_pipeline.sh  # smoke test
#
# Requires vLLM running on the allocated node (or set BASE_URL to a shared endpoint).

set -euo pipefail

REPO="/home/hmrirxk/AgenticHF"
AGENT="/home/hmrirxk/readmission_agent"
MODEL="${MODEL:-Qwen/Qwen3-32B}"
COHORT="${COHORT:-30d}"
LIMIT="${LIMIT:-0}"
WORKERS="${WORKERS:-4}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
START_VLLM="${START_VLLM:-0}"

case "$COHORT" in
  30d) COHORT_PARQUET="${REPO}/readmit_30d.parquet" ;;
  180d) COHORT_PARQUET="${REPO}/mimiciv_hf_180d_hf_readmit_extract.parquet" ;;
  *) COHORT_PARQUET="$COHORT" ;;
esac

source "${AGENT}/.venv/bin/activate"

export HF_HOME=/condo/alkindilab/hmrirxk/hf_cache

# HF token (first hf_ line in ~/notes.txt) for gated model downloads
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/notes.txt}"
if [[ -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(grep -m1 -oE 'hf_[A-Za-z0-9]+' "$HF_TOKEN_FILE" || true)"
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

if [[ "$START_VLLM" == "1" ]]; then
  echo "Starting vLLM for $MODEL ..."
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --host 0.0.0.0 --port 8000 &
  VLLM_PID=$!
  sleep 60
  trap "kill $VLLM_PID 2>/dev/null || true" EXIT
fi

COHORT="$COHORT_PARQUET" MODEL="$MODEL" BASE_URL="$BASE_URL" LIMIT="$LIMIT" WORKERS="$WORKERS" \
  bash "${REPO}/scripts/run_cohort_pipeline.sh"
