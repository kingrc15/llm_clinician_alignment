from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Dict, Iterator, Optional, Set, Tuple

import pandas as pd
from openai import BadRequestError
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential_jitter
from tqdm import tqdm

from .hf_readmit_verify_llm import HfReadmitVerifyVllmBackend
from .io import ensure_parent_dir, read_jsonl
from .cli_utils import _load_row_ids_file
from .vllm_resolve import resolve_vllm_model_id


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=15),
    retry=retry_if_not_exception_type(BadRequestError),
    reraise=True,
)
def _verify_with_retry(
    backend: HfReadmitVerifyVllmBackend,
    bin_label: str,
    bin_desc: str,
    idx_text: str,
    rd_text: str,
):
    return backend.verify_binning(bin_label, bin_desc, idx_text, rd_text)


def _load_done_row_ids(path: str) -> Set[int]:
    if not os.path.exists(path):
        return set()
    done: Set[int] = set()
    for rec in read_jsonl(path):
        rid = rec.get("row_id")
        if isinstance(rid, int):
            done.add(rid)
    return done


def _is_verify_ok(rec: Dict[str, Any]) -> bool:
    if rec.get("error"):
        return False
    return rec.get("makes_sense") is not None


def _binned_row_verifiable(rec: Dict[str, Any]) -> bool:
    if rec.get("error"):
        return False
    b = rec.get("reason_for_readmission_binned")
    d = rec.get("reason_for_readmission_binned_description")
    return isinstance(b, str) and b.strip() and isinstance(d, str) and d.strip()


def _pass2_row_ids_verifiable(binned_jsonl: str) -> Set[int]:
    latest = _load_latest_by_row_id(binned_jsonl)
    return {rid for rid, rec in latest.items() if _binned_row_verifiable(rec)}


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


def _iter_binned(path: str) -> Iterator[Dict[str, Any]]:
    yield from read_jsonl(path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Third pass: given binned readmit JSONL + cohort parquet (discharge texts), "
            "ask the LLM if the bin and explanation make sense."
        )
    )
    ap.add_argument("--binned-jsonl", required=True, help="Output from run_hf_readmit_bin_llm.py")
    ap.add_argument(
        "--cohort-parquet",
        required=True,
        help="Same cohort parquet as run_hf_readmit_llm (must include row_id and both discharge summaries).",
    )
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
        "--max-chars-each",
        type=int,
        default=0,
        help="Truncate each discharge summary to N chars (0 = no truncate).",
    )
    ap.add_argument(
        "--no-think-first",
        action="store_true",
        help="Disable <thinking>...</thinking> before JSON; use JSON-only mode (response_format).",
    )
    ap.add_argument(
        "--retry-errors",
        action="store_true",
        help=(
            "Rewrite --out-jsonl to keep only rows with a successful verification (makes_sense set, no error), "
            "then re-run verification for failed rows using the current --binned-jsonl."
        ),
    )
    args = ap.parse_args()

    cohort = pd.read_parquet(args.cohort_parquet)
    need = {"row_id", "index_discharge_summary", "readmit_discharge_summary"}
    missing = need - set(cohort.columns)
    if missing:
        raise ValueError(f"Cohort parquet missing columns: {sorted(missing)}")
    cohort = cohort.drop_duplicates(subset=["row_id"], keep="first")
    cohort = cohort.set_index("row_id", drop=False)

    ensure_parent_dir(args.out_jsonl)
    if args.overwrite and os.path.exists(args.out_jsonl):
        os.remove(args.out_jsonl)

    retry_ids: Optional[Set[int]] = None
    if args.retry_errors and os.path.exists(args.out_jsonl):
        latest_out = _load_latest_by_row_id(args.out_jsonl)
        retry_ids = {rid for rid, rec in latest_out.items() if not _is_verify_ok(rec)}
        if os.path.isfile(args.binned_jsonl):
            pass2_ok = _pass2_row_ids_verifiable(args.binned_jsonl)
            retry_ids |= pass2_ok - set(latest_out.keys())
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for rid, rec in sorted(latest_out.items(), key=lambda kv: kv[0]):
                if rid in retry_ids:
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    done = _load_done_row_ids(args.out_jsonl)

    row_filter: Optional[Set[int]] = None
    if args.row_ids_file:
        row_filter = _load_row_ids_file(args.row_ids_file)

    def _append_verify_task(rec: Dict[str, Any], rid: int) -> None:
        if row_filter is not None and rid not in row_filter:
            return
        if rid in done:
            return
        if rec.get("error"):
            return
        b = rec.get("reason_for_readmission_binned")
        d = rec.get("reason_for_readmission_binned_description")
        if not isinstance(b, str) or not b.strip():
            return
        if not isinstance(d, str):
            d = ""
        if not d.strip():
            return
        if rid not in cohort.index:
            return
        row = cohort.loc[rid]
        idx_t = str(row["index_discharge_summary"] or "")
        rd_t = str(row["readmit_discharge_summary"] or "")
        if not idx_t.strip() or not rd_t.strip():
            return
        max_c = int(args.max_chars_each) if args.max_chars_each and args.max_chars_each > 0 else 0
        if max_c:
            idx_t = idx_t[:max_c]
            rd_t = rd_t[:max_c]
        tasks.append((rec, b.strip(), d.strip(), idx_t, rd_t))

    tasks: list[Tuple[Dict[str, Any], str, str, str, str]] = []
    if os.path.isfile(args.binned_jsonl):
        if retry_ids is not None:
            binned_latest = _load_latest_by_row_id(args.binned_jsonl)
            for rid in sorted(retry_ids):
                rec = binned_latest.get(rid)
                if rec is None:
                    continue
                _append_verify_task(rec, rid)
        else:
            for rec in _iter_binned(args.binned_jsonl):
                rid = rec.get("row_id")
                if not isinstance(rid, int):
                    continue
                _append_verify_task(rec, rid)

    if args.limit and args.limit > 0:
        tasks = tasks[: int(args.limit)]

    resolved_model = resolve_vllm_model_id(args.base_url, args.model)
    max_tokens = args.max_tokens if args.max_tokens is not None else 3072
    backend = HfReadmitVerifyVllmBackend(
        base_url=args.base_url.rstrip("/"),
        model=resolved_model,
        temperature=args.temperature,
        max_tokens=max_tokens,
        think_first=not args.no_think_first,
    )

    total = len(tasks)
    if total == 0:
        if not args.retry_errors or not os.path.exists(args.out_jsonl):
            with open(args.out_jsonl, "w", encoding="utf-8"):
                pass
        print(json.dumps({"rows": 0, "out": args.out_jsonl, "note": "nothing to do"}))
        return

    error_n = 0

    with open(args.out_jsonl, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            it = iter(tasks)
            in_flight: Dict[Future, Dict[str, Any]] = {}
            target_in_flight = max(1, int(args.workers)) * 2

            def submit_one() -> bool:
                try:
                    rec, bl, bd, ix, rx = next(it)
                except StopIteration:
                    return False
                fut = ex.submit(_verify_with_retry, backend, bl, bd, ix, rx)
                in_flight[fut] = {
                    "rec": rec,
                    "reason_for_readmission_binned": bl,
                    "reason_for_readmission_binned_description": bd,
                }
                return True

            for _ in range(target_in_flight):
                if not submit_one():
                    break

            pbar = tqdm(total=total)
            while in_flight:
                done_futs, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for fut in done_futs:
                    meta = in_flight.pop(fut)
                    rec = meta["rec"]
                    row_id = int(rec["row_id"])
                    base = {
                        "row_id": row_id,
                        "patient_identifier": rec.get("patient_identifier"),
                        "subject_id": rec.get("subject_id"),
                        "index_hadm_id": rec.get("index_hadm_id"),
                        "readmit_hadm_id": rec.get("readmit_hadm_id"),
                        "reason_for_readmission_binned": meta["reason_for_readmission_binned"],
                        "reason_for_readmission_binned_description": meta["reason_for_readmission_binned_description"],
                    }
                    rh_icu = rec.get("readmit_has_icu")
                    if rh_icu is not None:
                        base["readmit_has_icu"] = bool(rh_icu)
                    try:
                        pred, raw = fut.result()
                        out_rec = {
                            **base,
                            "makes_sense": pred.makes_sense,
                            "verification_rationale": pred.rationale,
                            "verification_confidence": pred.confidence,
                            "verification_qualifier": pred.qualifier,
                            "raw": raw,
                        }
                    except Exception as e:
                        error_n += 1
                        out_rec = {
                            **base,
                            "makes_sense": None,
                            "verification_rationale": None,
                            "verification_confidence": None,
                            "verification_qualifier": None,
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
