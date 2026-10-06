"""Evaluation metrics for the reasoning DAG."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import networkx as nx

from .models import ReasoningStep


def compute_metrics(
    graph: nx.DiGraph,
    steps_by_trajectory: Dict[str, List[ReasoningStep]],
) -> Dict[str, Any]:
    """Compute structural metrics for the assembled DAG.

    Non-trivial fields:
      * trajectory_distinct_starts / _ends — how many distinct clusters the
        trajectories' first / last reasoning steps map to. With virtual
        endpoints enabled this is exactly the out-/in-degree of START/END.
      * sources / leaves — graph nodes with in_degree=0 / out_degree=0
        (excluding virtual START/END).
      * orphan_sources / orphan_leaves — sources/leaves that are NOT actually
        trajectory entry/exit clusters. These exist when DAG enforcement
        removes the only incoming/outgoing edge to/from an interior node.
      * transitions_retained / transitions_dropped — bookkeeping of how many
        sequential transitions made it into the graph vs. were removed (by
        self-loop merging or DAG enforcement).
    """
    total_steps = sum(len(ss) for ss in steps_by_trajectory.values())
    n_trajectories = len(steps_by_trajectory) or 1

    # Separate virtual START/END from real nodes for accounting.
    virtual_nodes = {n for n, d in graph.nodes(data=True) if d.get("virtual")}
    real_nodes = [n for n in graph.nodes if n not in virtual_nodes]
    real_edges = [
        (u, v, d) for u, v, d in graph.edges(data=True)
        if u not in virtual_nodes and v not in virtual_nodes
    ]

    n_nodes = len(real_nodes)
    n_edges = len(real_edges)

    node_compression_ratio = (1.0 - n_nodes / total_steps) if total_steps else 0.0

    out_degrees = []
    in_degrees = []
    for n in real_nodes:
        out_degrees.append(sum(1 for _, v in graph.out_edges(n) if v not in virtual_nodes))
        in_degrees.append(sum(1 for u, _ in graph.in_edges(n) if u not in virtual_nodes))

    non_leaf_out = [d for d in out_degrees if d > 0]
    average_branch_factor = (sum(non_leaf_out) / len(non_leaf_out)) if non_leaf_out else 0.0

    density = nx.density(graph.subgraph(real_nodes)) if real_nodes else 0.0

    trajectories_covered = set()
    for _, _, data in real_edges:
        for t in data.get("trajectories") or []:
            trajectories_covered.add(t)
    trajectory_coverage = (len(trajectories_covered) / n_trajectories) if n_trajectories else 0.0

    shared_nodes = sum(1 for n in real_nodes if int(graph.nodes[n].get("support_count", 0)) >= 2)
    divergence_count = sum(1 for n, d in zip(real_nodes, out_degrees) if d >= 2)
    convergence_count = sum(1 for n, d in zip(real_nodes, in_degrees) if d >= 2)

    sources = {n for n, d in zip(real_nodes, in_degrees) if d == 0}
    leaves = {n for n, d in zip(real_nodes, out_degrees) if d == 0}

    real_starts = set(graph.graph.get("trajectory_first_clusters", {}).values())
    real_ends = set(graph.graph.get("trajectory_last_clusters", {}).values())
    orphan_sources = sources - real_starts
    orphan_leaves = leaves - real_ends

    transitions_per_traj = graph.graph.get("transitions_per_trajectory") or {}
    expected_transitions = sum(transitions_per_traj.values()) or max(0, total_steps - n_trajectories)
    retained_transitions = sum(int(d.get("count", 0)) for _, _, d in real_edges)
    dropped_transitions = max(0, expected_transitions - retained_transitions)

    # Conclusion accounting: which clusters are flagged as answers, and how
    # many trajectories' answers landed in the same cluster.
    conclusion_clusters = [
        n for n in real_nodes if bool(graph.nodes[n].get("is_conclusion", False))
    ]
    conclusion_trajectories_total = sum(
        len(graph.nodes[n].get("conclusion_trajectories") or [])
        for n in conclusion_clusters
    )
    shared_conclusion_clusters = sum(
        1 for n in conclusion_clusters
        if len(graph.nodes[n].get("conclusion_trajectories") or []) >= 2
    )

    return {
        "n_trajectories": len(steps_by_trajectory),
        "n_steps_total": total_steps,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "node_compression_ratio": node_compression_ratio,
        "semantic_redundancy_reduction": node_compression_ratio,
        "average_branch_factor": average_branch_factor,
        "graph_density": density,
        "trajectory_coverage": trajectory_coverage,
        "shared_node_count": shared_nodes,
        "divergence_count": divergence_count,
        "convergence_count": convergence_count,
        "trajectory_distinct_starts": len(real_starts),
        "trajectory_distinct_ends": len(real_ends),
        "n_sources": len(sources),
        "n_leaves": len(leaves),
        "n_orphan_sources": len(orphan_sources),
        "n_orphan_leaves": len(orphan_leaves),
        "transitions_expected": expected_transitions,
        "transitions_retained": retained_transitions,
        "transitions_dropped": dropped_transitions,
        "dropped_back_edges": len(graph.graph.get("dropped_back_edges", []) or []),
        "virtual_endpoints": bool(graph.graph.get("virtual_endpoints", False)),
        "n_conclusion_clusters": len(conclusion_clusters),
        "n_conclusion_trajectories": conclusion_trajectories_total,
        "n_shared_conclusion_clusters": shared_conclusion_clusters,
    }
