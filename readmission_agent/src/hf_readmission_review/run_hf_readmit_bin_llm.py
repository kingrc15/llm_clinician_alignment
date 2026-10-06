from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Dict, Iterator, Optional, Set

from openai import BadRequestError
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential_jitter
from tqdm import tqdm

from .hf_readmit_bin_llm import HfReadmitBinVllmBackend
from .hf_readmit_bin_prompts import format_first_pass_context_from_analysis
from .io import ensure_parent_dir, read_jsonl
from .cli_utils import _load_row_ids_file
from .vllm_resolve import resolve_vllm_model_id


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=15),
    retry=retry_if_not_exception_type(BadRequestError),
    reraise=True,
)
def _bin_with_retry(backend: HfReadmitBinVllmBackend, structured_json: str):
    return backend.bin_readmit_reason(structured_json)


def _load_done_row_ids(path: str) -> Set[int]:
    if not os.path.exists(path):
        return set()
    done: Set[int] = set()
    for rec in read_jsonl(path):
        rid = rec.get("row_id")
        if isinstance(rid, int):
            done.add(rid)
    return done


def _extract_analysis(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    a = rec.get("analysis")
    if not isinstance(a, dict):
        return None
    need = ("diagnosis", "symptoms", "confidence", "factors")
    if any(k not in a for k in need):
        return None
    factors = a.get("factors")
    if not isinstance(factors, list) or not factors:
        return None
    return a


def _iter_records(path: str) -> Iterator[Dict[str, Any]]:
    yield from read_jsonl(path)


def _is_binned_ok(rec: Dict[str, Any]) -> bool:
    if rec.get("error"):
        return False
    b = rec.get("reason_for_readmission_binned")
    d = rec.get("reason_for_readmission_binned_description")
    return isinstance(b, str) and b.strip() and isinstance(d, str) and d.strip()


def _load_latest_by_row_id(path: str) -> Dict[int, Dict[str, Any]]:
    """
    JSONL may contain multiple entries per row_id (e.g., retries). Keep the last one.
    """
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


def _pass1_row_ids_with_valid_analysis(in_jsonl: str) -> Set[int]:
    latest = _load_latest_by_row_id(in_jsonl)
    return {
        rid
        for rid, rec in latest.items()
        if not rec.get("error") and _extract_analysis(rec) is not None
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Second pass: bin readmission reason from first-pass HF readmit JSONL analysis fields."
    )
    ap.add_argument("--in-jsonl", required=True, help="JSONL from run_hf_readmit_llm.py")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--row-ids-file", default="")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--retry-errors",
        action="store_true",
        help=(
            "Retry rows that previously had an error or missing bin in --out-jsonl. "
            "This rewrites --out-jsonl to keep only successful rows, then re-runs the failed ones."
        ),
    )
    ap.add_argument(
        "--no-think-first",
        action="store_true",
        help="Disable <thinking>...</thinking> before JSON; use JSON-only mode (response_format).",
    )
    args = ap.parse_args()

    ensure_parent_dir(args.out_jsonl)
    # Overwrite takes precedence.
    if args.overwrite and os.path.exists(args.out_jsonl):
        os.remove(args.out_jsonl)

    # If retrying, rewrite output to keep only successful rows,
    # and compute a retry_id set from the last-seen record per row_id.
    retry_ids: Optional[Set[int]] = None
    if args.retry_errors and os.path.exists(args.out_jsonl):
        latest_out = _load_latest_by_row_id(args.out_jsonl)
        retry_ids = {rid for rid, rec in latest_out.items() if not _is_binned_ok(rec)}
        # Rows fixed in pass-1 after an error may never appear in pass-2; bin them too.
        pass1_ok = _pass1_row_ids_with_valid_analysis(args.in_jsonl)
        retry_ids |= pass1_ok - set(latest_out.keys())
        # Rewrite output: keep only successful rows.
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for rid, rec in sorted(latest_out.items(), key=lambda kv: kv[0]):
                if rid in retry_ids:
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    done = _load_done_row_ids(args.out_jsonl)

    row_filter: Optional[Set[int]] = None
    if args.row_ids_file:
        row_filter = _load_row_ids_file(args.row_ids_file)

    records: list[Dict[str, Any]] = []
    if retry_ids is not None:
        pass1_latest = _load_latest_by_row_id(args.in_jsonl)
        for rid in sorted(retry_ids):
            rec = pass1_latest.get(rid)
            if rec is None:
                continue
            if row_filter is not None and rid not in row_filter:
                continue
            if rid in done:
                continue
            if rec.get("error"):
                continue
            if _extract_analysis(rec) is None:
                continue
            records.append(rec)
    else:
        for rec in _iter_records(args.in_jsonl):
            rid = rec.get("row_id")
            if not isinstance(rid, int):
                continue
            if row_filter is not None and rid not in row_filter:
                continue
            if rid in done:
                continue
            if rec.get("error"):
                continue
            analysis = _extract_analysis(rec)
            if analysis is None:
                continue
            records.append(rec)

    if args.limit and args.limit > 0:
        records = records[: int(args.limit)]

    resolved_model = resolve_vllm_model_id(args.base_url, args.model)
    max_tokens = args.max_tokens if args.max_tokens is not None else 3072
    backend = HfReadmitBinVllmBackend(
        base_url=args.base_url.rstrip("/"),
        model=resolved_model,
        temperature=args.temperature,
        max_tokens=max_tokens,
        think_first=not args.no_think_first,
    )

    total = len(records)
    if total == 0:
        # Do not wipe an existing pass-2 file when --retry-errors found nothing new to bin.
        if not args.retry_errors or not os.path.exists(args.out_jsonl):
            with open(args.out_jsonl, "w", encoding="utf-8"):
                pass
        print(json.dumps({"rows": 0, "out": args.out_jsonl, "note": "nothing to do"}))
        return

    error_n = 0

    with open(args.out_jsonl, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            task_iter = iter(records)
            in_flight: Dict[Future, Dict[str, Any]] = {}
            target_in_flight = max(1, int(args.workers)) * 2

            def submit_one() -> bool:
                try:
                    rec = next(task_iter)
                except StopIteration:
                    return False
                analysis = _extract_analysis(rec)
                assert analysis is not None
                ctx = format_first_pass_context_from_analysis(analysis)
                fut = ex.submit(_bin_with_retry, backend, ctx)
                in_flight[fut] = rec
                return True

            for _ in range(target_in_flight):
                if not submit_one():
                    break

            pbar = tqdm(total=total)
            while in_flight:
                done_futs, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for fut in done_futs:
                    rec = in_flight.pop(fut)
                    row_id = int(rec["row_id"])
                    base = {
                        "row_id": row_id,
                        "patient_identifier": rec.get("patient_identifier"),
                        "subject_id": rec.get("subject_id"),
                        "index_hadm_id": rec.get("index_hadm_id"),
                        "readmit_hadm_id": rec.get("readmit_hadm_id"),
                        "index_primary_icd_code": rec.get("index_primary_icd_code"),
                        "days_to_readmit": rec.get("days_to_readmit"),
                        "first_pass_analysis": rec.get("analysis"),
                    }
                    rh_icu = rec.get("readmit_has_icu")
                    if rh_icu is not None:
                        base["readmit_has_icu"] = bool(rh_icu)
                    try:
                        pred, raw = fut.result()
                        out_rec = {
                            **base,
                            "reason_for_readmission_binned": pred.reason_for_readmission_binned,
                            "reason_for_readmission_binned_description": pred.reason_for_readmission_binned_description,
                            "raw": raw,
                        }
                    except Exception as e:
                        error_n += 1
                        out_rec = {
                            **base,
                            "reason_for_readmission_binned": None,
                            "reason_for_readmission_binned_description": None,
                            "error": {"type": type(e).__name__, "message": str(e)},
                        }
                    f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
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
