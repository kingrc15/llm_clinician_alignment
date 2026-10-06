# LLM–clinician alignment (HF readmission)

Public code for comparing open-weight LLM heart-failure readmission reviews with clinician annotations.

**This repository does not contain MIMIC notes, clinician review JSON, parquet extracts, model outputs, API tokens, or other PHI.** Those stay on credentialed local storage. Obtain MIMIC-III/IV via [PhysioNet](https://physionet.org/) and keep reviews off git.

Layout:

| Path | Purpose |
|------|---------|
| `eval/` | Load clinician review exports and comparison helpers |
| `scripts/` | Slurm / local vLLM pipeline launchers |
| `readmission_agent/` | Three-pass OpenAI-compatible LLM review + optional reasoning graph |

See `readmission_agent/README.md` for install details.

## Quick start

### 1. Pipeline (requires GPU + vLLM)

```bash
# Smoke test (5 rows, BioMistral-7B on quickgpuq)
sbatch scripts/slurm_smoke_test.sh

# Full run: all 4 models × both cohorts
./scripts/submit_all_pipeline_jobs.sh

# Or single run locally (vLLM must be running)
COHORT=mimiciv_hf_30d_hf_readmit_extract.parquet \
MODEL=BioMistral/BioMistral-7B LIMIT=5 \
./scripts/run_cohort_pipeline.sh
```

Outputs land in `outputs/{hf_30d|hf_180d}/{model_slug}/pass{1,2,3}.jsonl`.

### 2. Evaluation (after pipeline outputs exist)

```bash
pip install -r eval/requirements.txt

# Build crosswalk (validates review selectedText against parquet notes)
python -m eval.validate_crosswalk --method text_fingerprint

# Build comparison table (dry-run works without crosswalk matches)
python -m eval.build_comparison_table --cohort-tag hf_30d
```

### 3. Cohort alignment note

Clinician reviews in `20260707T144323Z/` reference a **different patient cohort** than the current parquet extracts (0% `selectedText` validation). A crosswalk file at `eval/data/crosswalk.csv` is generated but requires the original review cohort parquet to achieve valid joins. Until then, pipeline runs proceed on current parquet; comparison metrics will populate once alignment is fixed.

## Model comparison table columns

| Column | Description |
|--------|-------------|
| `usmle_medqa_accuracy` | Published MedQA-USMLE-4 score from `eval/data/usmle_benchmarks.csv` |
| `bleu_mean` / `bleu_max` | LLM `thinking` vs clinician `selectedText` |
| `bertscore_f1_mean` | Semantic similarity of thinking vs evidence spans |
| `label_top1_semantic_match` | Pass-1 ranked reason vs primary clinician factor |
| `bin_agreement_rate` | Pass-2 bin vs embedding-mapped clinician label |
