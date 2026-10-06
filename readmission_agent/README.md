# Readmission agent


## Layout

| Path | Purpose |
|------|---------|
| `src/hf_readmission_review/` | Three-pass OpenAI-compatible LLM pipeline (vLLM-ready). |
| `src/reasoning_tree/app/` | Segment → embed → cluster → DAG → `graph.html`. |
| `scripts/run_hf_readmit_review.sh` | Shell wrapper: two discharge `.txt` files + model id. |
| `scripts/build_reasoning_graph.sh` | Shell wrapper: pass-1 JSONL → graph artifacts. |
| `scripts/run_tests.sh` | `reasoning_tree` unit tests (always uses `.venv/bin/python`). |
| `examples/example_responses.json` | Tiny sample for the graph CLI in “spec” mode. |

## Install

**Python 3.10+** recommended.

### pip (minimal — review only, no graph)

```bash
cd readmission_agent_share   # or the path where you cloned this tree
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### pip (review + reasoning graph)

Graph mode pulls `torch` and `sentence-transformers` (large). Use a GPU machine
for the default `Qwen/Qwen3-Embedding-8B` embedding model, or pass a smaller
`--embedding-model` (see `reasoning_tree` CLI help).

```bash
pip install -e ".[graph]"
```

### conda (includes graph extra)

```bash
cd readmission_agent_share
conda env create -f environment.yml
conda activate readmission-agent-share
```

## vLLM (or other OpenAI-compatible server)

Point `BASE_URL` at your server (default `http://localhost:8000/v1`). The
`MODEL` value must match a model id your server exposes (often `local-model` if
you set `--served-model-name` in vLLM).

## Run the three-pass review from two text files

Prepare UTF-8 plain text:

- **Index discharge** — discharge summary from the prior (index) HF
  hospitalization.
- **Readmit discharge** — discharge summary from the readmission stay.
- **Optional** — readmission admission / H&P note (if you omit it, an empty note
  is sent, which is supported).

### Shell (recommended)

```bash
cd readmission_agent_share
./scripts/run_hf_readmit_review.sh \
  /path/to/index_discharge.txt \
  /path/to/readmit_discharge.txt \
  local-model
```

Optional environment variables:

- `READMIT_ADMISSION_NOTE=/path/to/hp.txt`
- `OUT_DIR=/path/to/outputs` (default `./outputs/review_run`)
- `BASE_URL=http://host:port/v1`
- `NO_THINK_FIRST=1` — use strict JSON mode if `<thinking>` blocks are unstable.
- `OVERWRITE=1` — delete prior `review_*.jsonl` in `OUT_DIR` before re-running.
- `MAX_CHARS_EACH=8000` — truncate each note for context-limited models.
- `MAX_TOKENS=4096` — cap completion length for all passes.

### Python module (same behavior)

```bash
export PYTHONPATH="$(pwd)/src"
python -m hf_readmission_review.run_pair_review \
  --index-discharge /path/to/index_discharge.txt \
  --readmit-discharge /path/to/readmit_discharge.txt \
  --model local-model
```

### Outputs (`OUT_DIR`)

| File | Content |
|------|---------|
| `cohort_one_row.parquet` | Single-row cohort used internally. |
| `review_pass1_analysis.jsonl` | Structured analysis + model `thinking` / raw fields. |
| `review_pass2_binned.jsonl` | Dominant readmission-reason bin + description. |
| `review_pass3_verified.jsonl` | `makes_sense` check vs full discharge texts. |

## Build the reasoning graph

After pass 1 exists, point the graph script at that JSONL (default expects
`row_id=0` from `run_pair_review`):

```bash
cd readmission_agent_share
PASS1_JSONL=outputs/review_run/review_pass1_analysis.jsonl \
OUT_DIR=outputs/reasoning_graph \
./scripts/build_reasoning_graph.sh
```

Artifacts in `OUT_DIR`:

- `graph.html` — open in a browser.
- `graph.json`, `graph_dump.json`, `steps.json`, `clusters.json`.

Tuning: set `SIM_THRESHOLD` (default `0.83`), `FIELD` (`thinking` vs `content`),
or `EMBED_MODEL` for a lighter embedding model on CPU.

### Spec-style graph (no HF JSONL)

```bash
python -m reasoning_tree.app.main \
  --input examples/example_responses.json \
  --out-dir outputs/example_graph
```

Discharge paths can be **absolute** or **relative**. The script `cd`s into the
repo root before calling Python; relative paths are resolved in this order:
the path as given (under repo root), then **the directory you invoked the
script from**, then the repo root again. Prefer absolute paths if unsure.

## Tests and environment hygiene

Reasoning-graph tests import **scikit-learn** (and thus **SciPy**). Many conda
installations ship **NumPy 2.x** with SciPy wheels that were built against
**NumPy 1.x**, which triggers import errors like
`numpy.core.multiarray failed to import`.

**Do this:**

1. Use this repo’s venv and reinstall so the `numpy<2` pin in `pyproject.toml`
   applies: `pip install -e ".[graph]" --upgrade`
2. Run tests **only** with that interpreter:

```bash
cd readmission_agent_share
./scripts/run_tests.sh
```

Do **not** run `python -m unittest discover …` with a random conda `python`
unless that environment has matching NumPy / SciPy / scikit-learn builds.

## Relationship to the full `readmission_agent` repo

The full research worktree (same machine layout as your original project)
includes MIMIC cohort builders, the 30-day binary classifier with optional EHR
tooling, ACC guideline tools, and experiment write-ups. **This `*_share` tree
is intentionally smaller:** ship only this directory to collaborators who
only need the HF paired-summary review and optional reasoning graph.

None of the Python modules here hard-code your home directory; paths in shell
examples use the repo root relative to where you run commands.

## License / data

Discharge text is **your responsibility**—do not commit PHI. The prompts
implement a fixed clinical reasoning template; validate outputs in your own
QA process before any operational use.
