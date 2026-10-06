"""
Run the HF readmission three-pass review from two discharge summary files.

Pass 1: structured analysis (JSON + optional <thinking>).
Pass 2: single dominant readmission-reason bin (five canonical buckets).
Pass 3: sanity-check the bin against the full discharge texts.

This entry point writes a temporary one-row cohort parquet and invokes the same
modules as batch processing. No EHR tools, no ACC guideline PDF tools, and no
experiment logging.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--index-discharge",
        required=True,
        type=Path,
        help="Path to UTF-8 text: discharge summary from the index (prior) HF hospitalization.",
    )
    ap.add_argument(
        "--readmit-discharge",
        required=True,
        type=Path,
        help="Path to UTF-8 text: discharge summary from the readmission stay.",
    )
    ap.add_argument(
        "--readmit-admission-note",
        type=Path,
        default=None,
        help="Optional UTF-8 text: readmission admission / H&P note (empty if omitted).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/review_run"),
        help="Directory for cohort parquet + pass1/2/3 JSONL outputs.",
    )
    ap.add_argument("--row-id", type=int, default=0, help="row_id stored in outputs (default 0).")
    ap.add_argument(
        "--patient-id",
        default="shared_case",
        help="String identifier echoed in JSONL records.",
    )
    ap.add_argument("--model", required=True, help="Model id as served by vLLM (e.g. local-model).")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible API base URL.")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=None, help="Completion token budget for all passes.")
    ap.add_argument(
        "--max-chars-each",
        type=int,
        default=0,
        help="Truncate each long note to N characters (0 = full text).",
    )
    ap.add_argument(
        "--no-think-first",
        action="store_true",
        help="JSON-only mode for models that do not emit <thinking> blocks reliably.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete previous JSONL outputs in out-dir for this run prefix before starting.",
    )
    args = ap.parse_args()

    idx = _read_text(args.index_discharge)
    rd = _read_text(args.readmit_discharge)
    adm = _read_text(args.readmit_admission_note) if args.readmit_admission_note else ""

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = out_dir / "cohort_one_row.parquet"
    prefix = "review"
    p1 = out_dir / f"{prefix}_pass1_analysis.jsonl"
    p2 = out_dir / f"{prefix}_pass2_binned.jsonl"
    p3 = out_dir / f"{prefix}_pass3_verified.jsonl"

    from .cohort_from_text import write_minimal_cohort_parquet

    write_minimal_cohort_parquet(
        out_path=cohort_path,
        row_id=args.row_id,
        patient_identifier=args.patient_id,
        index_discharge_summary=idx,
        readmit_discharge_summary=rd,
        readmit_admission_note=adm,
    )

    py = sys.executable
    common_core: list[str] = [
        f"--model={args.model}",
        f"--base-url={args.base_url}",
        f"--workers={args.workers}",
    ]
    if args.max_tokens is not None:
        common_core.append(f"--max-tokens={args.max_tokens}")
    if args.no_think_first:
        common_core.append("--no-think-first")
    if args.overwrite:
        common_core.append("--overwrite")

    trunc: list[str] = []
    if args.max_chars_each and args.max_chars_each > 0:
        trunc.append(f"--max-chars-each={args.max_chars_each}")

    env = os.environ.copy()
    # Ensure local package is importable when invoked without installation.
    repo_src = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = f"{repo_src}{os.pathsep}{env.get('PYTHONPATH', '')}"

    def run(mod: str, extra: list[str], *, note_trunc: bool) -> None:
        tail = list(common_core)
        if note_trunc:
            tail.extend(trunc)
        cmd = [py, "-m", mod, *extra, *tail]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=env)

    run(
        "hf_readmission_review.run_hf_readmit_llm",
        [
            f"--cohort-parquet={cohort_path}",
            f"--out-jsonl={p1}",
            "--temperature=0.2",
            "--limit=1",
        ],
        note_trunc=True,
    )
    run(
        "hf_readmission_review.run_hf_readmit_bin_llm",
        [
            f"--in-jsonl={p1}",
            f"--out-jsonl={p2}",
            "--temperature=0.1",
            "--limit=1",
        ],
        note_trunc=False,
    )
    run(
        "hf_readmission_review.run_hf_readmit_bin_verify_llm",
        [
            f"--binned-jsonl={p2}",
            f"--cohort-parquet={cohort_path}",
            f"--out-jsonl={p3}",
            "--temperature=0.1",
            "--limit=1",
        ],
        note_trunc=True,
    )

    print(
        f"Done.\n  Cohort: {cohort_path}\n  Pass 1: {p1}\n  Pass 2: {p2}\n  Pass 3: {p3}\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
