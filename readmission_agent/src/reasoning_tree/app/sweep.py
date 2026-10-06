"""Threshold / setting sweep helper.

Runs the reasoning-DAG pipeline multiple times with different settings and
prints a compact comparison table so you can pick the merging level you want.

Example:

.. code-block:: bash

    PYTHONPATH=. python -m reasoning_tree.app.sweep \
        --from-jsonl runs/hf_principal_reasons_manual_review_20260508_155205.jsonl \
        --from-jsonl runs/hf_principal_reasons_manual_review_20260509_085739.jsonl \
        --from-jsonl runs/hf_principal_reasons_manual_review_20260511_142040.jsonl \
        --row-id 0 \
        --thresholds 0.85 0.80 0.75 0.70 \
        --canonicalizations simple aggressive \
        --post-merge-jaccards 0.0 0.6
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from .canonicalization import get_canonicalizer, make_llm_canonicalizer
from .clustering import cluster_steps
from .embeddings import StepEmbedder, embed_steps
from .graph_builder import build_reasoning_graph
from .loaders import load_principal_reasons_jsonl_pair, load_responses_json
from .metrics import compute_metrics
from .segmentation import segment_response


def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="Path to responses.json.")
    src.add_argument("--from-jsonl", action="append", help="Principal-reasons JSONL (repeatable).")
    ap.add_argument("--row-id", type=int, default=None)
    ap.add_argument("--field", default="thinking", choices=["thinking", "analysis", "content"])
    ap.add_argument("--trajectory-label", action="append", default=None)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.85, 0.80, 0.75, 0.70])
    ap.add_argument("--canonicalizations", nargs="+", default=["simple", "aggressive"],
                    choices=["simple", "aggressive", "llm"])
    ap.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--llm-model", default="local-model")
    ap.add_argument("--llm-temperature", type=float, default=0.0)
    ap.add_argument("--llm-max-tokens", type=int, default=128)
    ap.add_argument("--llm-workers", type=int, default=8)
    ap.add_argument("--llm-cache", default="reasoning_tree/outputs/.llm_canonicalize_cache.json")
    ap.add_argument("--linkages", nargs="+", default=["average"],
                    choices=["average", "single", "complete", "ward"])
    ap.add_argument("--post-merge-jaccards", type=float, nargs="+", default=[0.0])
    ap.add_argument("--min-step-chars", type=int, default=0)
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-json", default="")
    return ap.parse_args(argv)


def _load_payload(args: argparse.Namespace):
    if args.input:
        return load_responses_json(args.input)
    if not args.from_jsonl or args.row_id is None:
        raise SystemExit("--from-jsonl requires --row-id")
    return load_principal_reasons_jsonl_pair(
        args.from_jsonl, row_id=int(args.row_id), field=args.field,
        trajectory_labels=args.trajectory_label,
    )


def _segment_canon(payload, canon_mode: str, min_chars: int, llm_kwargs=None):
    # 1) segment
    pre = [(r.trajectory_id, segment_response(r.text, trajectory_id=r.trajectory_id))
           for r in payload.responses]
    # 2) canonicalize (batched for llm mode)
    if canon_mode == "llm":
        client = make_llm_canonicalizer(**(llm_kwargs or {}))
        flat = [s.raw_text for _, ss in pre for s in ss]
        flat_canon = client.canonicalize_batch(flat)
        it = iter(flat_canon)
        for _, ss in pre:
            for s in ss:
                s.canonical_text = next(it)
    else:
        canon_fn = get_canonicalizer(canon_mode)
        for _, ss in pre:
            for s in ss:
                s.canonical_text = canon_fn(s.raw_text)
    # 3) filter
    all_steps = []
    by_traj: Dict[str, list] = {}
    for trajectory_id, steps in pre:
        kept = []
        for s in steps:
            ref = s.canonical_text or s.raw_text
            if not ref or len(ref) < min_chars:
                continue
            kept.append(s)
        for i, s in enumerate(kept):
            s.step_index = i
        by_traj[trajectory_id] = kept
        all_steps.extend(kept)
    return all_steps, by_traj


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    payload = _load_payload(args)

    embedder = StepEmbedder(model_name=args.embedding_model, device=args.device)

    # We re-segment + re-embed per canonicalization mode (since text changes),
    # but reuse embeddings across threshold/linkage/post-merge variants.
    cache: Dict[str, Any] = {}
    results: List[Dict[str, Any]] = []

    llm_kwargs = {
        "base_url": args.llm_base_url,
        "model": args.llm_model,
        "temperature": float(args.llm_temperature),
        "max_tokens": int(args.llm_max_tokens),
        "cache_path": (args.llm_cache or None),
        "workers": int(args.llm_workers),
    }

    for canon in args.canonicalizations:
        steps, by_traj = _segment_canon(payload, canon, int(args.min_step_chars), llm_kwargs=llm_kwargs)
        if not steps:
            print(f"[skip] canonicalization={canon} produced 0 steps", file=sys.stderr)
            continue
        embed_steps(steps, embedder=embedder, use_canonical=True)
        cache[canon] = (steps, by_traj)

        for thr in args.thresholds:
            for linkage in args.linkages:
                for pmj in args.post_merge_jaccards:
                    clustered = cluster_steps(
                        steps,
                        similarity_threshold=float(thr),
                        linkage=linkage,
                        post_merge_jaccard=float(pmj),
                    )
                    g = build_reasoning_graph(clustered, list(by_traj.values()))
                    m = compute_metrics(g, by_traj)
                    results.append({
                        "canonicalization": canon,
                        "similarity_threshold": float(thr),
                        "linkage": linkage,
                        "post_merge_jaccard": float(pmj),
                        "n_steps": m["n_steps_total"],
                        "n_nodes": m["n_nodes"],
                        "n_edges": m["n_edges"],
                        "compression": round(m["node_compression_ratio"], 3),
                        "shared_nodes": m["shared_node_count"],
                        "divergence": m["divergence_count"],
                        "convergence": m["convergence_count"],
                        "dropped_back_edges": m["dropped_back_edges"],
                    })

    if not results:
        print("no results", file=sys.stderr)
        sys.exit(1)

    # Pretty table.
    headers = [
        "canon", "thr", "linkage", "pmj",
        "steps", "nodes", "edges", "compr",
        "shared", "div", "conv", "back",
    ]
    fmt = "{:>11} {:>5} {:>8} {:>5}  {:>5} {:>5} {:>5} {:>6}  {:>6} {:>4} {:>4} {:>4}"
    print(fmt.format(*headers))
    print("-" * len(fmt.format(*headers)))
    for r in results:
        print(fmt.format(
            r["canonicalization"],
            f"{r['similarity_threshold']:.2f}",
            r["linkage"],
            f"{r['post_merge_jaccard']:.2f}",
            r["n_steps"], r["n_nodes"], r["n_edges"], f"{r['compression']:.3f}",
            r["shared_nodes"], r["divergence"], r["convergence"], r["dropped_back_edges"],
        ))

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"prompt": payload.prompt, "results": results}, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
