from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Dict, Iterator, Set

import pandas as pd
from openai import BadRequestError
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential_jitter
from tqdm import tqdm

from .hf_readmit_llm import HfReadmitVllmBackend
from .io import ensure_parent_dir, read_jsonl
from .cli_utils import _load_row_ids_file
from .vllm_resolve import resolve_vllm_model_id


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=15),
    retry=retry_if_not_exception_type(BadRequestError),
    reraise=True,
)
def _analyze_with_retry(
    backend: HfReadmitVllmBackend,
    patient_identifier: str,
    index_text: str,
    readmit_text: str,
):
    return backend.analyze(patient_identifier, index_text, readmit_text)


def _load_done_row_ids(path: str) -> Set[int]:
    if not os.path.exists(path):
        return set()
    done: Set[int] = set()
    for rec in read_jsonl(path):
        rid = rec.get("row_id")
        if isinstance(rid, int):
            done.add(rid)
    return done


def _is_analysis_ok(rec: Dict[str, Any]) -> bool:
    if rec.get("error"):
        return False
    a = rec.get("analysis")
    if not isinstance(a, dict):
        return False
    need = ("diagnosis", "symptoms", "confidence", "factors")
    if not all(k in a for k in need):
        return False
    factors = a.get("factors")
    if not isinstance(factors, list) or not factors:
        return False
    for f in factors:
        if not isinstance(f, dict):
            return False
        if not str(f.get("label", "")).strip():
            return False
        if f.get("role") not in ("primary", "contributing"):
            return False
    return True


def _load_latest_by_row_id(path: str) -> Dict[int, Dict[str, Any]]:
    latest: Dict[int, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return latest
    for rec in read_jsonl(path):
        if not isinstance(rec, dict):
            continue
        rid = rec.get("row_id")
        if isinstance(rid, int):
            latest[rid] = rec
    return latest


def _iter_tasks(df: pd.DataFrame) -> Iterator[Dict[str, Any]]:
    for rec in df.to_dict("records"):
        yield rec


def main() -> None:
    ap = argparse.ArgumentParser(
        description="HF primary index admission → first hospital readmit within 30d: structured LLM analysis"
    )
    ap.add_argument("--cohort-parquet", required=True, help="Parquet cohort: row_id, patient_identifier, index + readmit discharge summaries")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Completion budget (default 3072). Reasoning models (e.g. Qwen3) need enough headroom for "
            "`reasoning` + JSON in `content`. Keep prompt+max_tokens within the server's max_model_len."
        ),
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--row-ids-file", default="")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--retry-errors",
        action="store_true",
        help=(
            "Rewrite --out-jsonl to keep only rows with a valid analysis, then re-run cohort rows that had "
            "errors or missing analysis fields (append new results)."
        ),
    )
    ap.add_argument("--max-chars-each", type=int, default=0, help="Truncate each discharge summary to N chars (0=no truncate)")
    ap.add_argument(
        "--no-think-first",
        action="store_true",
        help="Disable <thinking>...</thinking> before JSON; use JSON-only mode (response_format).",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.cohort_parquet)
    required = {
        "row_id",
        "patient_identifier",
        "index_discharge_summary",
        "readmit_discharge_summary",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cohort parquet missing columns: {sorted(missing)}")

    # readmit_admission_note column ignored if present in parquet
    if "readmit_admission_note" not in df.columns:
        df["readmit_admission_note"] = ""

    if args.row_ids_file:
        ids = _load_row_ids_file(args.row_ids_file)
        df = df[df["row_id"].astype(int).isin(ids)].copy()

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    ensure_parent_dir(args.out_jsonl)
    if args.overwrite and os.path.exists(args.out_jsonl):
        os.remove(args.out_jsonl)

    if args.retry_errors and os.path.exists(args.out_jsonl):
        latest = _load_latest_by_row_id(args.out_jsonl)
        retry_ids = {rid for rid, rec in latest.items() if not _is_analysis_ok(rec)}
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for rid, rec in sorted(latest.items(), key=lambda kv: kv[0]):
                if rid in retry_ids:
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    done = _load_done_row_ids(args.out_jsonl)
    if done:
        df = df[~df["row_id"].astype(int).isin(done)].copy()

    resolved_model = resolve_vllm_model_id(args.base_url, args.model)
    max_tokens = args.max_tokens if args.max_tokens is not None else 8192
    backend = HfReadmitVllmBackend(
        base_url=args.base_url.rstrip("/"),
        model=resolved_model,
        temperature=args.temperature,
        max_tokens=max_tokens,
        think_first=not args.no_think_first,
    )

    total = len(df)
    if total == 0:
        print(json.dumps({"rows": 0, "out": args.out_jsonl, "note": "nothing to do"}))
        return

    error_n = 0
    max_chars = int(args.max_chars_each) if args.max_chars_each and args.max_chars_each > 0 else 0

    with open(args.out_jsonl, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            task_iter = _iter_tasks(df)
            in_flight: Dict[Future, Dict[str, Any]] = {}
            target_in_flight = max(1, int(args.workers)) * 2

            def submit_one() -> bool:
                try:
                    row = next(task_iter)
                except StopIteration:
                    return False
                row_id = int(row["row_id"])
                pid = str(row["patient_identifier"])
                idx_t = str(row["index_discharge_summary"])
                rd_t = str(row["readmit_discharge_summary"])
                if max_chars:
                    idx_t = idx_t[:max_chars]
                    rd_t = rd_t[:max_chars]
                fut = ex.submit(_analyze_with_retry, backend, pid, idx_t, rd_t)
                in_flight[fut] = row
                return True

            for _ in range(target_in_flight):
                if not submit_one():
                    break

            pbar = tqdm(total=total)
            while in_flight:
                done_futs, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for fut in done_futs:
                    row = in_flight.pop(fut)
                    row_id = int(row["row_id"])
                    base_rec = {
                        "row_id": row_id,
                        "subject_id": row.get("subject_id"),
                        "index_hadm_id": row.get("index_hadm_id"),
                        "readmit_hadm_id": row.get("readmit_hadm_id"),
                        "patient_identifier": row.get("patient_identifier"),
                        "index_primary_icd_code": row.get("index_primary_icd_code"),
                        "days_to_readmit": row.get("days_to_readmit"),
                    }
                    rh_icu = row.get("readmit_has_icu")
                    if rh_icu is not None and pd.notna(rh_icu):
                        base_rec["readmit_has_icu"] = bool(rh_icu)
                    try:
                        pred, raw = fut.result()
                        rec = {
                            **base_rec,
                            "thinking": raw.get("thinking"),
                            "analysis": pred.model_dump(),
                            "raw": raw,
                        }
                    except Exception as e:
                        error_n += 1
                        rec = {
                            **base_rec,
                            "thinking": None,
                            "analysis": None,
                            "error": {"type": type(e).__name__, "message": str(e)},
                        }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    pbar.update(1)
                    submit_one()
            pbar.close()

    print(
        json.dumps(
            {"rows": total, "out": args.out_jsonl, "resumed_skipped": len(done), "errors": error_n},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
