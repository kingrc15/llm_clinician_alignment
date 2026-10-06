"""Top-level CLI entry point.

Two input modes are supported:

1. ``--input responses.json``: the canonical spec shape:

   .. code-block:: json

       {"prompt": "...", "responses": [{"trajectory_id": "r1", "text": "..."}, ...]}

2. ``--from-jsonl <file>`` (repeatable) + ``--row-id <int>``: each JSONL is a
   principal-reasons run; one row is pulled from each to compare different
   reasoning trajectories on the same input.

Outputs written to ``--out-dir`` (default ``reasoning_tree/outputs``):

* ``steps.json``         — every segmented step with embedding ids
* ``clusters.json``      — cluster_id -> [step ids]
* ``graph.json``         — node/edge dump compatible with d3/cytoscape
* ``graph_dump.json``    — pydantic ``GraphDump`` (richer, with metrics + config)
* ``graph.html``         — interactive pyvis visualization
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from .canonicalization import get_canonicalizer, make_llm_canonicalizer
from .clustering import cluster_steps
from .embeddings import StepEmbedder, embed_steps
from .graph_builder import (
    build_reasoning_graph,
    graph_to_dump,
    prune_from_source,
    prune_to_conclusion,
    restore_trajectory_flow,
)
from .loaders import load_principal_reasons_jsonl_pair, load_responses_json
from .metrics import compute_metrics
from .models import ResponsesPayload
from .segmentation import segment_response
from .utils import ensure_dir, stable_id, write_json
from .visualization import graph_to_json_dict, visualize_graph


def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="Path to responses.json (spec format).")
    src.add_argument(
        "--from-jsonl",
        action="append",
        default=None,
        help=(
            "Path to a principal-reasons JSONL. Pass repeatedly to compare multiple "
            "runs. Requires --row-id."
        ),
    )
    ap.add_argument("--row-id", type=int, default=None, help="row_id to extract when using --from-jsonl.")
    ap.add_argument(
        "--field",
        default="thinking",
        choices=["thinking", "analysis", "content"],
        help="Field to use when ingesting principal-reasons JSONL.",
    )
    ap.add_argument("--trajectory-label", action="append", default=None,
                    help="Label for each --from-jsonl trajectory (repeatable; default = filename).")
    ap.add_argument("--out-dir", default="reasoning_tree/outputs",
                    help="Output directory for all generated artifacts.")
    ap.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.83,
        help="Cosine similarity threshold for clustering (recommended 0.70–0.90; "
             "lower = more aggressive merging).",
    )
    ap.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-8B",
        help="HF model id for sentence-transformers. Default: Qwen/Qwen3-Embedding-8B "
             "(4096-dim, last-token-pooled). For a smaller/CPU-friendly choice try "
             "Qwen/Qwen3-Embedding-0.6B or sentence-transformers/all-MiniLM-L6-v2.",
    )
    ap.add_argument(
        "--embedding-dim",
        type=int,
        default=0,
        help="Optional Matryoshka truncation dimension (256/512/1024/2048/4096). "
             "0 = use the model's full output dimension.",
    )
    ap.add_argument(
        "--embedding-instruction",
        default=None,
        help="Optional natural-language instruction prefix to send with every "
             "text (Qwen3-Embedding supports this). Leave unset for symmetric "
             "clustering.",
    )
    ap.add_argument(
        "--embedding-cache",
        default="reasoning_tree/outputs/.embedding_cache.json",
        help="Disk cache for computed embeddings (keyed by model+instruction+dim+text). "
             "Set to '' to disable. Strongly recommended with the 8B default.",
    )
    ap.add_argument(
        "--embedding-batch-size",
        type=int,
        default=4,
        help="Batch size for the embedding model. Default 4 (safe for Qwen3-Embedding-8B "
             "on most GPUs). Raise it for smaller models or larger GPU RAM.",
    )
    ap.add_argument(
        "--embedding-dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Torch dtype to load the embedding model with. 'auto' lets HF decide; "
             "use 'bfloat16' or 'float16' to fit the 8B model on a single GPU.",
    )
    ap.add_argument("--device", default=None, help="Torch device for sentence-transformers (e.g. 'cuda', 'cpu').")
    ap.add_argument(
        "--canonicalization",
        default="simple",
        choices=["simple", "aggressive", "llm"],
        help="Canonicalizer to apply before embedding. "
             "'aggressive' drops headings, generic subjects, hedges, parentheticals, "
             "and expands clinical acronyms (rule-based). "
             "'llm' uses an OpenAI-compatible chat endpoint (e.g. vLLM) to rewrite "
             "each step into a single canonical proposition (see --llm-*).",
    )
    ap.add_argument(
        "--llm-base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible base URL for --canonicalization llm (default: vLLM on localhost:8000).",
    )
    ap.add_argument(
        "--llm-model",
        default="local-model",
        help="Model id to request from the LLM server (accepts the short id or HF root name).",
    )
    ap.add_argument("--llm-temperature", type=float, default=0.0)
    ap.add_argument("--llm-max-tokens", type=int, default=256)
    ap.add_argument(
        "--llm-keep-thinking",
        action="store_true",
        help="Don't pass chat_template_kwargs.enable_thinking=False (only useful "
             "for models that need their reasoning step kept on).",
    )
    ap.add_argument(
        "--llm-workers",
        type=int,
        default=8,
        help="Concurrent canonicalization requests (only used with --canonicalization llm).",
    )
    ap.add_argument(
        "--llm-cache",
        default="reasoning_tree/outputs/.llm_canonicalize_cache.json",
        help="Disk cache for LLM canonicalizations. Set to '' to disable.",
    )
    ap.add_argument(
        "--linkage",
        default="average",
        choices=["average", "single", "complete", "ward"],
        help="Linkage for agglomerative clustering. 'single' merges chained "
             "near-neighbors more aggressively (use with caution).",
    )
    ap.add_argument(
        "--post-merge-jaccard",
        type=float,
        default=0.0,
        help="If >0, run a second pass that merges clusters whose canonical-text "
             "tokens have Jaccard overlap >= this threshold (0.55–0.7 is a good range). "
             "Catches semantic dupes the embedding missed.",
    )
    ap.add_argument(
        "--min-step-chars",
        type=int,
        default=0,
        help="Drop steps whose canonical text is shorter than this many characters "
             "(useful for filtering 'Synthesis:'/'Confidence:' fragments).",
    )
    ap.add_argument("--no-canonicalize", action="store_true",
                    help="Skip canonicalization and embed raw step text.")
    ap.add_argument(
        "--allow-cycles",
        action="store_true",
        help="Do NOT enforce DAG (keep all sequential edges, including cycle-inducing ones).",
    )
    ap.add_argument(
        "--keep-self-loops",
        action="store_true",
        help="Keep self-edges (transitions to the same cluster) instead of dropping them.",
    )
    ap.add_argument(
        "--no-virtual-endpoints",
        action="store_true",
        help="Disable the synthetic START/END nodes (which by default give the graph "
             "exactly one source and one leaf, with edge counts equal to the number "
             "of trajectories entering/exiting each real cluster).",
    )
    ap.add_argument(
        "--conclusion-field",
        default="principal_reason",
        choices=["none", "principal_reason", "principal+contributing"],
        help="What to use as each trajectory's explicit conclusion / answer step. "
             "'principal_reason' (default) appends the JSONL record's "
             "analysis.principal_reason as the final reasoning step, so the cluster "
             "containing it is identified as the trajectory's conclusion. "
             "'principal+contributing' joins contributing reasons too. "
             "'none' disables the feature (no conclusion step appended).",
    )
    ap.add_argument(
        "--prune-to-conclusion",
        action="store_true",
        help="After building the graph, remove every node that does NOT lie on a "
             "directed path to a conclusion (★ ANSWER) cluster. Useful for "
             "removing dead-end branches and DAG-enforcement orphans, leaving only "
             "the reasoning that actually leads to the answer.",
    )
    ap.add_argument(
        "--mix-conclusions-with-reasoning",
        action="store_true",
        help="Allow conclusion (★ ANSWER) steps to cluster with semantically similar "
             "reasoning steps. By default they are kept in a separate clustering pool, "
             "which guarantees the answer cluster's in-degree equals the number of "
             "trajectories that ended there (one edge per trajectory).",
    )
    ap.add_argument(
        "--prune-from-source",
        action="store_true",
        help="After building the graph, remove every node that is NOT reachable "
             "from the source (virtual START if enabled, otherwise the union of "
             "all trajectory first clusters). Eliminates orphan-source nodes "
             "created when DAG enforcement severs an interior node's only "
             "incoming edge. Combine with --prune-to-conclusion to keep only "
             "nodes on a START → ANSWER path.",
    )
    ap.add_argument(
        "--no-restore-trajectory-flow",
        action="store_true",
        help="By default, after any DAG enforcement / pruning surgery, the "
             "pipeline ensures flow conservation by chaining each "
             "trajectory's transits through ONLY the clusters still present "
             "in the graph (pruned intermediates are skipped over with a "
             "direct edge between the surrounding kept clusters). This "
             "guarantees trajectory_in_degree(v) == trajectory_out_degree(v) "
             "for every interior node. Pass this flag to disable the "
             "restore step entirely.",
    )
    ap.add_argument(
        "--restore-re-add-pruned",
        action="store_true",
        help="When restoring trajectory flow, also re-add any cluster that "
             "pruning removed if it lies on a trajectory's transit. This "
             "effectively un-prunes (the default chain-over-pruned mode "
             "respects pruning instead).",
    )
    ap.add_argument("--title", default=None, help="Title to display in the HTML visualization.")
    ap.add_argument(
        "--layout",
        default="auto",
        choices=["auto", "hierarchical", "physics"],
        help="Graph layout. 'auto' (default) uses hierarchical for DAGs and "
             "force-directed (physics) when the graph has cycles (e.g. after "
             "trajectory-flow restore re-adds back-edges). 'hierarchical' and "
             "'physics' force a specific layout.",
    )
    ap.add_argument("--node-spacing", type=int, default=360,
                    help="Horizontal spacing between sibling nodes (hierarchical).")
    ap.add_argument("--level-separation", type=int, default=260,
                    help="Vertical spacing between layers (hierarchical).")
    ap.add_argument("--tree-spacing", type=int, default=360,
                    help="Spacing between separate tree components (hierarchical).")
    ap.add_argument("--label-width", type=int, default=260,
                    help="Maximum label width in px before text wraps in node boxes.")
    ap.add_argument("--label-chars", type=int, default=110,
                    help="Max characters of canonical text in each node label "
                         "before truncating with ellipsis.")
    ap.add_argument("--canvas-height", default="1000px",
                    help="HTML canvas height (e.g. '1200px', '90vh').")
    return ap.parse_args(argv)


def _load_payload(args: argparse.Namespace) -> ResponsesPayload:
    if args.input:
        return load_responses_json(args.input)
    if not args.from_jsonl or args.row_id is None:
        raise SystemExit("--from-jsonl requires --row-id")
    return load_principal_reasons_jsonl_pair(
        args.from_jsonl, row_id=int(args.row_id), field=args.field,
        trajectory_labels=args.trajectory_label,
        conclusion_field=args.conclusion_field,
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    payload = _load_payload(args)
    ensure_dir(args.out_dir)

    min_chars = max(0, int(args.min_step_chars))

    # 1) Segment first so we can batch the LLM canonicalizer. If the trajectory
    #    has an explicit conclusion_text (e.g. principal_reason), append it as
    #    a final, flagged reasoning step so that the cluster containing it can
    #    be identified as the trajectory's answer.
    from .models import ReasoningStep  # local import to avoid circular hint

    all_steps_pre = []
    for resp in payload.responses:
        steps = segment_response(resp.text, trajectory_id=resp.trajectory_id)
        if resp.conclusion_text and resp.conclusion_text.strip():
            concl_text = resp.conclusion_text.strip()
            concl_id = stable_id("conclusion", resp.trajectory_id, len(steps), concl_text)
            steps.append(ReasoningStep(
                id=concl_id,
                trajectory_id=resp.trajectory_id,
                step_index=len(steps),
                raw_text=concl_text,
                is_conclusion=True,
            ))
        all_steps_pre.append((resp.trajectory_id, steps))

    # 2) Canonicalize. LLM mode uses batched concurrent calls; rule-based modes
    #    run inline.
    llm_client = None
    if not args.no_canonicalize and args.canonicalization == "llm":
        llm_client = make_llm_canonicalizer(
            base_url=args.llm_base_url,
            model=args.llm_model,
            temperature=float(args.llm_temperature),
            max_tokens=int(args.llm_max_tokens),
            cache_path=(args.llm_cache or None),
            workers=int(args.llm_workers),
            disable_thinking=not args.llm_keep_thinking,
        )
        flat_raw = [s.raw_text for _, steps in all_steps_pre for s in steps]
        flat_canon = llm_client.canonicalize_batch(flat_raw)
        it = iter(flat_canon)
        for _, steps in all_steps_pre:
            for s in steps:
                s.canonical_text = next(it)
    else:
        canon_fn = (lambda x: x) if args.no_canonicalize else get_canonicalizer(args.canonicalization)
        for _, steps in all_steps_pre:
            for s in steps:
                s.canonical_text = canon_fn(s.raw_text)

    # 3) Filter empty/short (but never drop a marked conclusion step), reindex.
    all_steps = []
    steps_by_trajectory: Dict[str, list] = {}
    n_filtered = 0
    for trajectory_id, steps in all_steps_pre:
        kept = []
        for s in steps:
            ref = s.canonical_text if s.canonical_text is not None else s.raw_text
            if (not ref or len(ref) < min_chars) and not s.is_conclusion:
                n_filtered += 1
                continue
            kept.append(s)
        for i, s in enumerate(kept):
            s.step_index = i
        steps_by_trajectory[trajectory_id] = kept
        all_steps.extend(kept)

    if not all_steps:
        raise SystemExit("No reasoning steps survived filtering — try lowering --min-step-chars.")

    model_kwargs: Dict[str, Any] = {}
    if args.embedding_dtype and args.embedding_dtype != "auto":
        model_kwargs["torch_dtype"] = args.embedding_dtype

    embedder = StepEmbedder(
        model_name=args.embedding_model,
        device=args.device,
        instruction=args.embedding_instruction,
        truncate_dim=(int(args.embedding_dim) if args.embedding_dim else None),
        cache_path=(args.embedding_cache or None),
        batch_size=int(args.embedding_batch_size),
        model_kwargs=model_kwargs or None,
    )
    embed_steps(all_steps, embedder=embedder, use_canonical=not args.no_canonicalize)

    clustered = cluster_steps(
        all_steps,
        similarity_threshold=float(args.similarity_threshold),
        linkage=args.linkage,
        post_merge_jaccard=float(args.post_merge_jaccard),
        partition_conclusions=not args.mix_conclusions_with_reasoning,
    )

    graph = build_reasoning_graph(
        clustered_steps=clustered,
        trajectories=list(steps_by_trajectory.values()),
        drop_self_loops=not args.keep_self_loops,
        enforce_dag=not args.allow_cycles,
        add_virtual_endpoints=not args.no_virtual_endpoints,
    )

    prune_audits = []
    if args.prune_to_conclusion:
        prune_audits.append(("to_conclusion", prune_to_conclusion(graph)))
    if args.prune_from_source:
        prune_audits.append(("from_source", prune_from_source(graph)))

    restore_audit = None
    if not args.no_restore_trajectory_flow:
        restore_audit = restore_trajectory_flow(
            graph,
            clustered_steps=clustered,
            trajectories=list(steps_by_trajectory.values()),
            re_add_pruned_nodes=bool(args.restore_re_add_pruned),
        )

    metrics = compute_metrics(graph, steps_by_trajectory)
    if prune_audits:
        metrics["prune_audit"] = {name: audit for name, audit in prune_audits}
    if restore_audit is not None:
        metrics["trajectory_flow_restore"] = restore_audit

    config = {
        "similarity_threshold": float(args.similarity_threshold),
        "embedding_model": args.embedding_model,
        "embedding_dim": int(args.embedding_dim) or None,
        "embedding_instruction": args.embedding_instruction,
        "embedding_cache_path": (args.embedding_cache or None),
        "embedding_batch_size": int(args.embedding_batch_size),
        "embedding_dtype": args.embedding_dtype,
        "device": args.device,
        "canonicalization": "none" if args.no_canonicalize else args.canonicalization,
        "linkage": args.linkage,
        "post_merge_jaccard": float(args.post_merge_jaccard),
        "min_step_chars": int(args.min_step_chars),
        "filtered_steps": int(n_filtered),
        "drop_self_loops": not args.keep_self_loops,
        "enforce_dag": not args.allow_cycles,
        "field": args.field if not args.input else None,
        "row_id": args.row_id if not args.input else None,
        "conclusion_field": args.conclusion_field if not args.input else None,
        "prune_to_conclusion": bool(args.prune_to_conclusion),
        "prune_from_source": bool(args.prune_from_source),
        "sources": args.from_jsonl or [args.input],
    }
    if llm_client is not None:
        config["llm"] = {
            "base_url": llm_client.base_url,
            "requested_model": llm_client.requested_model,
            "resolved_model": llm_client.model_id,
            "temperature": llm_client.temperature,
            "max_tokens": llm_client.max_tokens,
            "workers": llm_client.workers,
            "cache_path": llm_client.cache_path,
        }

    dump = graph_to_dump(
        graph=graph,
        trajectories=[r.trajectory_id for r in payload.responses],
        prompt=payload.prompt,
        metrics=metrics,
        config=config,
    )

    steps_payload = {
        "trajectories": {
            tid: [s.model_dump() for s in steps]
            for tid, steps in steps_by_trajectory.items()
        },
        "config": config,
    }
    clusters_payload = {
        "clusters": {
            str(cid): [s.id for s in steps]
            for cid, steps in clustered.items()
        }
    }

    graph_html = os.path.join(args.out_dir, "graph.html")
    graph_json = os.path.join(args.out_dir, "graph.json")
    graph_dump_json = os.path.join(args.out_dir, "graph_dump.json")
    steps_json = os.path.join(args.out_dir, "steps.json")
    clusters_json = os.path.join(args.out_dir, "clusters.json")

    write_json(steps_json, steps_payload)
    write_json(clusters_json, clusters_payload)
    write_json(graph_json, graph_to_json_dict(graph))
    write_json(graph_dump_json, json.loads(dump.model_dump_json()))
    visualize_graph(
        graph,
        graph_html,
        title=args.title or payload.prompt,
        height=args.canvas_height,
        layout=args.layout,
        node_spacing=int(args.node_spacing),
        level_separation=int(args.level_separation),
        tree_spacing=int(args.tree_spacing),
        label_width=int(args.label_width),
        label_chars=int(args.label_chars),
    )

    print(
        json.dumps(
            {
                "out_dir": args.out_dir,
                "graph_html": graph_html,
                "graph_json": graph_json,
                "graph_dump_json": graph_dump_json,
                "steps_json": steps_json,
                "clusters_json": clusters_json,
                "metrics": metrics,
                "config": config,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return {"metrics": metrics, "config": config}


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    run(args)


if __name__ == "__main__":
    main()
