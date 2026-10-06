"""Phase 5 — Graph construction.

Build a directed graph (preferring a DAG with reconvergence) where:

* every semantic cluster of steps is one node
* edges represent sequential reasoning transitions within trajectories
* shared prefixes/suffixes naturally merge because they map to the same node
* self-loops (a cluster transitioning to itself) are dropped — they're an
  artifact of two adjacent steps being semantic duplicates.

The implementation guarantees the result is a DAG: if a transition would
introduce a cycle (because two trajectories order the same pair of clusters
in opposite directions), the conflicting edge is dropped and recorded under
graph.graph["dropped_back_edges"] for transparency.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx

from .models import ReasoningEdge, ReasoningNode, ReasoningStep
from .utils import stable_id


def _pick_representative_step(steps: List[ReasoningStep]) -> ReasoningStep:
    """Choose the cluster representative: shortest non-empty canonical text."""
    def key(s: ReasoningStep) -> Tuple[int, int]:
        text = s.canonical_text or s.raw_text or ""
        return (len(text) if text else 10**9, s.step_index)

    return sorted(steps, key=key)[0]


def _build_nodes(
    clustered_steps: Dict[int, List[ReasoningStep]],
) -> Tuple[Dict[int, ReasoningNode], Dict[str, int]]:
    """Materialize one ReasoningNode per cluster + a step_id -> cluster lookup."""
    nodes: Dict[int, ReasoningNode] = {}
    step_to_cluster: Dict[str, int] = {}

    for cid, steps in clustered_steps.items():
        # Prefer a conclusion step as the cluster representative when available,
        # so the visualization shows the canonical answer text on the node.
        conclusion_steps = [s for s in steps if getattr(s, "is_conclusion", False)]
        rep = _pick_representative_step(conclusion_steps or steps)
        trajectory_ids = sorted({s.trajectory_id for s in steps})
        conclusion_trajs = sorted({s.trajectory_id for s in conclusion_steps})
        node = ReasoningNode(
            node_id=stable_id("node", cid, rep.canonical_text or rep.raw_text),
            canonical_text=rep.canonical_text or rep.raw_text,
            supporting_steps=[s.id for s in steps],
            trajectory_ids=trajectory_ids,
            support_count=len(trajectory_ids),
            representative_raw_text=rep.raw_text,
            is_conclusion=bool(conclusion_steps),
            conclusion_trajectories=conclusion_trajs,
        )
        nodes[cid] = node
        for s in steps:
            step_to_cluster[s.id] = cid

    return nodes, step_to_cluster


def _enforce_dag(graph: nx.DiGraph) -> List[Tuple[str, str]]:
    """Remove edges that introduce cycles. Returns the dropped (u,v) pairs.

    Strategy: iterate the directed graph; whenever a cycle is detected, drop
    the lowest-support edge in the cycle. Repeat until the graph is a DAG.
    """
    dropped: List[Tuple[str, str]] = []
    safety = 0
    while not nx.is_directed_acyclic_graph(graph):
        safety += 1
        if safety > 10_000:
            break
        try:
            cycle = nx.find_cycle(graph, orientation="original")
        except nx.NetworkXNoCycle:
            break
        # find_cycle returns list of (u, v, key/dir) tuples
        cycle_edges = [(u, v) for (u, v, *_rest) in cycle]
        u, v = min(cycle_edges, key=lambda uv: graph.edges[uv].get("count", 0))
        graph.remove_edge(u, v)
        dropped.append((u, v))
    return dropped


VIRTUAL_START_ID = "__START__"
VIRTUAL_END_ID = "__END__"


def _prune_keeping(graph: nx.DiGraph, keep: set) -> Tuple[int, int, int, int]:
    """Remove every node NOT in `keep`. Returns (n_before, n_after, e_before, e_after)."""
    n_before = graph.number_of_nodes()
    e_before = graph.number_of_edges()
    drop = [n for n in list(graph.nodes()) if n not in keep]
    graph.remove_nodes_from(drop)
    # Recompute transition probabilities after pruning.
    for u in graph.nodes:
        out_data = [d for _, _, d in graph.out_edges(u, data=True)]
        total = sum(int(d.get("count", 0)) for d in out_data) or 1
        for d in out_data:
            d["probability"] = int(d.get("count", 0)) / total
    return n_before, graph.number_of_nodes(), e_before, graph.number_of_edges()


def _trajectory_paths(
    clustered_steps: Dict[int, List[ReasoningStep]],
    trajectories: Iterable[Iterable[ReasoningStep]],
) -> List[List[Tuple[int, int]]]:
    """For each trajectory, return the list of (cluster_id, cluster_id) transitions
    in its actual step order, with self-loops collapsed (consecutive identical
    clusters merged). This is the ground-truth flow each trajectory took
    through the cluster space, BEFORE any DAG/prune surgery.
    """
    step_to_cluster: Dict[str, int] = {}
    for cid, ss in clustered_steps.items():
        for s in ss:
            step_to_cluster[s.id] = cid
    paths: List[List[Tuple[int, int]]] = []
    for tsteps in trajectories:
        ts = list(tsteps)
        prev_cid = None
        seq: List[int] = []
        for s in ts:
            cid = step_to_cluster.get(s.id)
            if cid is None:
                continue
            if cid == prev_cid:
                continue
            seq.append(cid)
            prev_cid = cid
        paths.append([(seq[i], seq[i + 1]) for i in range(len(seq) - 1)])
    return paths


def restore_trajectory_flow(
    graph: nx.DiGraph,
    clustered_steps: Dict[int, List[ReasoningStep]],
    trajectories: Iterable[Iterable[ReasoningStep]],
    re_add_pruned_nodes: bool = False,
) -> Dict[str, Any]:
    """Ensure every trajectory's transit through the **current** graph is
    represented as an edge, so that flow conservation holds for every
    interior node.

    Two behaviours:

    * ``re_add_pruned_nodes=False`` (default, respects pruning): for each
      trajectory ``T``, take its actual cluster sequence and *filter it down
      to clusters that are present in the graph* (skipping any cluster that
      pruning removed). Then add a direct edge for every consecutive pair
      in the filtered sequence. This "chains over" pruned intermediates,
      so a trajectory whose original transit was ``A -> B(pruned) -> C``
      contributes an edge ``A -> C``.
    * ``re_add_pruned_nodes=True`` (the older behaviour): if a cluster was
      pruned but lies on a trajectory's transit, re-add the cluster node
      and the original edges. This *un-prunes* any cluster touched by any
      trajectory, which usually negates the pruning entirely.

    After either mode, virtual START/END edges are recomputed against the
    final graph so their counts stay correct.
    """
    nodes_by_cluster, _ = _build_nodes(clustered_steps)
    node_id_for_cluster: Dict[int, str] = {cid: n.node_id for cid, n in nodes_by_cluster.items()}

    paths = _trajectory_paths(clustered_steps, trajectories)
    trajectory_ids = []
    for ts in trajectories:
        ts = list(ts)
        trajectory_ids.append(ts[0].trajectory_id if ts else None)

    re_added_nodes = 0
    re_added_edges = 0
    edge_traj: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    # Seed edge_traj from real edges already in the graph so multiplicities
    # from trajectories already present aren't lost.
    for u, v, data in graph.edges(data=True):
        if data.get("virtual"):
            continue
        for t in data.get("trajectories") or []:
            edge_traj[(u, v)].append(t)

    if re_add_pruned_nodes:
        # Restore mode: walk each consecutive transition from the original
        # trajectory and re-add any missing nodes/edges.
        for path, trajectory_id in zip(paths, trajectory_ids):
            if not path or trajectory_id is None:
                continue
            for c_a, c_b in path:
                u = node_id_for_cluster[c_a]
                v = node_id_for_cluster[c_b]
                for cid, nid in ((c_a, u), (c_b, v)):
                    if nid in graph:
                        continue
                    src_node = nodes_by_cluster[cid]
                    graph.add_node(
                        nid,
                        node_obj=src_node,
                        canonical_text=src_node.canonical_text,
                        support_count=src_node.support_count,
                        trajectory_ids=list(src_node.trajectory_ids),
                        supporting_steps=list(src_node.supporting_steps),
                        representative_raw_text=src_node.representative_raw_text,
                        virtual=False,
                        is_conclusion=src_node.is_conclusion,
                        conclusion_trajectories=list(src_node.conclusion_trajectories),
                    )
                    re_added_nodes += 1
                if not graph.has_edge(u, v):
                    re_added_edges += 1
                edge_traj.setdefault((u, v), [])
                if trajectory_id not in edge_traj[(u, v)]:
                    edge_traj[(u, v)].append(trajectory_id)
    else:
        # Chain-over-pruned mode: each trajectory contributes edges only
        # between its visited clusters that are still present in the graph,
        # with pruned intermediates skipped.
        for path, trajectory_id in zip(paths, trajectory_ids):
            if trajectory_id is None:
                continue
            # Reconstruct the trajectory's cluster sequence from path edges.
            seq: List[int] = []
            for i, (c_a, c_b) in enumerate(path):
                if i == 0:
                    seq.append(c_a)
                seq.append(c_b)
            present_seq = [
                cid for cid in seq
                if node_id_for_cluster.get(cid) in graph
            ]
            for a, b in zip(present_seq, present_seq[1:]):
                u = node_id_for_cluster[a]
                v = node_id_for_cluster[b]
                if u == v:
                    continue  # self-loops collapsed
                if not graph.has_edge(u, v):
                    re_added_edges += 1
                edge_traj.setdefault((u, v), [])
                if trajectory_id not in edge_traj[(u, v)]:
                    edge_traj[(u, v)].append(trajectory_id)

    # Rebuild every real edge in one pass to canonicalise count/probability.
    real_pairs = [(u, v) for u, v in list(graph.edges())
                  if not graph.edges[u, v].get("virtual")]
    for u, v in real_pairs:
        graph.remove_edge(u, v)
    for (u, v), trajs in edge_traj.items():
        if u not in graph or v not in graph:
            continue
        trajs_sorted = sorted(set(trajs))
        graph.add_edge(
            u, v,
            trajectories=trajs_sorted,
            count=len(trajs_sorted),
            virtual=False,
        )

    # Rebuild virtual endpoints if they were present (drop & recreate so
    # counts reflect the restored predecessors / successors).
    had_start = VIRTUAL_START_ID in graph
    had_end = VIRTUAL_END_ID in graph
    if had_start:
        graph.remove_node(VIRTUAL_START_ID)
    if had_end:
        graph.remove_node(VIRTUAL_END_ID)
    if had_start or had_end:
        # Build a step.id -> cluster_id lookup for O(1) trajectory walking.
        step_to_cluster_id: Dict[str, int] = {}
        for cid, ss in clustered_steps.items():
            for s in ss:
                step_to_cluster_id[s.id] = cid

        first_clusters: Dict[str, int] = {}
        last_clusters: Dict[str, int] = {}
        for tsteps in trajectories:
            ts = list(tsteps)
            if not ts:
                continue
            tid = ts[0].trajectory_id
            # First and last cluster *on this trajectory's path* that is still
            # present in the graph after pruning. Pruned intermediates and
            # pruned first/last steps are skipped so that every trajectory is
            # represented in the START / END fan-out even if its literal
            # opening or closing reasoning step was pruned.
            first_cid = next(
                (step_to_cluster_id.get(s.id) for s in ts
                 if step_to_cluster_id.get(s.id) is not None
                 and node_id_for_cluster[step_to_cluster_id[s.id]] in graph),
                None,
            )
            last_cid = next(
                (step_to_cluster_id.get(s.id) for s in reversed(ts)
                 if step_to_cluster_id.get(s.id) is not None
                 and node_id_for_cluster[step_to_cluster_id[s.id]] in graph),
                None,
            )
            if first_cid is not None:
                first_clusters[tid] = first_cid
            if last_cid is not None:
                last_clusters[tid] = last_cid
        if had_start and first_clusters:
            graph.add_node(VIRTUAL_START_ID,
                node_obj=None, canonical_text="START",
                support_count=len(first_clusters),
                trajectory_ids=sorted(first_clusters.keys()),
                supporting_steps=[], representative_raw_text="(virtual entry node)",
                virtual=True, is_conclusion=False, conclusion_trajectories=[])
            buckets: Dict[str, List[str]] = defaultdict(list)
            for tid, cid in first_clusters.items():
                buckets[node_id_for_cluster[cid]].append(tid)
            for tgt, trajs in buckets.items():
                graph.add_edge(VIRTUAL_START_ID, tgt,
                    trajectories=sorted(set(trajs)), count=len(trajs), virtual=True)
        if had_end and last_clusters:
            graph.add_node(VIRTUAL_END_ID,
                node_obj=None, canonical_text="END",
                support_count=len(last_clusters),
                trajectory_ids=sorted(last_clusters.keys()),
                supporting_steps=[], representative_raw_text="(virtual exit node)",
                virtual=True, is_conclusion=False, conclusion_trajectories=[])
            buckets = defaultdict(list)
            for tid, cid in last_clusters.items():
                buckets[node_id_for_cluster[cid]].append(tid)
            for src, trajs in buckets.items():
                graph.add_edge(src, VIRTUAL_END_ID,
                    trajectories=sorted(set(trajs)), count=len(trajs), virtual=True)

    # Recompute transition probabilities per source.
    for u in graph.nodes:
        out_data = [d for _, _, d in graph.out_edges(u, data=True)]
        total = sum(int(d.get("count", 0)) for d in out_data) or 1
        for d in out_data:
            d["probability"] = int(d.get("count", 0)) / total

    graph.graph["trajectory_flow_restored"] = True
    return {
        "restored": True,
        "n_nodes_after": graph.number_of_nodes(),
        "n_edges_after": graph.number_of_edges(),
        "n_nodes_re_added": re_added_nodes,
        "n_edges_re_added": re_added_edges,
    }


def prune_from_source(graph: nx.DiGraph) -> Dict[str, Any]:
    """Remove every node that is NOT reachable from a trajectory entry.

    "Source" is defined as:
      * the virtual START node (if it exists in the graph), or
      * every cluster that is a trajectory's first reasoning step
        (``graph.graph["trajectory_first_clusters"]``).

    A node is *kept* iff it equals a source OR there is a directed path from
    at least one source into it. This removes intermediate clusters whose
    only incoming edges were severed by DAG enforcement — they look like
    "extra source" nodes in the visualization but cannot actually be reached
    from any trajectory's true beginning.
    """
    if VIRTUAL_START_ID in graph:
        sources = {VIRTUAL_START_ID}
    else:
        sources = set((graph.graph.get("trajectory_first_clusters") or {}).values())
    sources = {s for s in sources if s in graph}

    if not sources:
        return {
            "pruned": False,
            "reason": "no source nodes resolvable",
            "n_nodes_before": graph.number_of_nodes(),
            "n_nodes_after": graph.number_of_nodes(),
        }

    keep: set = set(sources)
    for s in sources:
        keep.update(nx.descendants(graph, s))

    # The virtual END node (if present) is kept iff at least one of its
    # surviving predecessors connects to it.
    if VIRTUAL_END_ID in graph and any(
        (v == VIRTUAL_END_ID and u in keep) for u, v in graph.edges()
    ):
        keep.add(VIRTUAL_END_ID)

    n_before, n_after, e_before, e_after = _prune_keeping(graph, keep)
    graph.graph["pruned_from_source"] = True

    return {
        "pruned": True,
        "n_nodes_before": n_before,
        "n_nodes_after": n_after,
        "n_nodes_removed": n_before - n_after,
        "n_edges_before": e_before,
        "n_edges_after": e_after,
        "sources_used": sorted(sources),
    }


def prune_to_conclusion(graph: nx.DiGraph) -> Dict[str, Any]:
    """Remove every node that is not on a path to a conclusion node.

    A node is *kept* iff it is a conclusion node itself, OR it can reach a
    conclusion node by following directed edges. The virtual START node is
    kept iff any of its successors survive (with stale edges from START to
    removed nodes deleted). The virtual END node is kept iff any of its
    predecessors survive.

    Mutates the graph in place and returns a small audit dict.
    """
    conclusion_nodes = {
        n for n, d in graph.nodes(data=True)
        if bool(d.get("is_conclusion", False)) and not d.get("virtual")
    }
    if not conclusion_nodes:
        return {
            "pruned": False,
            "reason": "no conclusion clusters present — nothing to prune",
            "n_nodes_before": graph.number_of_nodes(),
            "n_nodes_after": graph.number_of_nodes(),
        }

    keep: set = set(conclusion_nodes)
    for c in conclusion_nodes:
        keep.update(nx.ancestors(graph, c))

    for vid in (VIRTUAL_START_ID, VIRTUAL_END_ID):
        if vid in graph and any(
            (u == vid and v in keep) or (v == vid and u in keep)
            for u, v in graph.edges()
        ):
            keep.add(vid)

    n_before, n_after, e_before, e_after = _prune_keeping(graph, keep)
    graph.graph["pruned_to_conclusion"] = True

    return {
        "pruned": True,
        "n_nodes_before": n_before,
        "n_nodes_after": n_after,
        "n_nodes_removed": n_before - n_after,
        "n_edges_before": e_before,
        "n_edges_after": e_after,
        "conclusion_nodes": sorted(conclusion_nodes),
    }


def build_reasoning_graph(
    clustered_steps: Dict[int, List[ReasoningStep]],
    trajectories: Iterable[Iterable[ReasoningStep]],
    drop_self_loops: bool = True,
    enforce_dag: bool = True,
    add_virtual_endpoints: bool = False,
) -> nx.DiGraph:
    """Construct the reasoning DAG from clustered steps and ordered trajectories.

    Parameters
    ----------
    clustered_steps : dict[int, list[ReasoningStep]]
        Output of :func:`clustering.cluster_steps`.
    trajectories : iterable of iterable of ReasoningStep
        Each inner iterable is one trajectory's steps in trajectory order.
    drop_self_loops : bool
        If True, transitions that map to the same cluster are skipped.
    enforce_dag : bool
        If True, edges that create cycles are removed (lowest-support first).

    Returns
    -------
    networkx.DiGraph
        Nodes carry the ReasoningNode payload under attr key 'node_obj'.
        Edges carry trajectory ids ('trajectories'), 'count' and 'probability'.
    """
    nodes_by_cluster, step_to_cluster = _build_nodes(clustered_steps)

    graph = nx.DiGraph()
    for cid, node in nodes_by_cluster.items():
        graph.add_node(
            node.node_id,
            node_obj=node,
            canonical_text=node.canonical_text,
            support_count=node.support_count,
            trajectory_ids=list(node.trajectory_ids),
            supporting_steps=list(node.supporting_steps),
            representative_raw_text=node.representative_raw_text,
            virtual=False,
            is_conclusion=node.is_conclusion,
            conclusion_trajectories=list(node.conclusion_trajectories),
        )

    # Track each trajectory's first/last cluster + transition count for accounting.
    first_clusters: Dict[str, int] = {}
    last_clusters: Dict[str, int] = {}
    transitions_per_traj: Dict[str, int] = {}

    edge_trajectories: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for traj_steps in trajectories:
        traj_steps = list(traj_steps)
        if not traj_steps:
            continue
        trajectory_id = traj_steps[0].trajectory_id
        first_cid = step_to_cluster.get(traj_steps[0].id)
        last_cid = step_to_cluster.get(traj_steps[-1].id)
        if first_cid is not None:
            first_clusters[trajectory_id] = first_cid
        if last_cid is not None:
            last_clusters[trajectory_id] = last_cid

        transitions_per_traj[trajectory_id] = max(0, len(traj_steps) - 1)
        for s1, s2 in zip(traj_steps, traj_steps[1:]):
            c1 = step_to_cluster.get(s1.id)
            c2 = step_to_cluster.get(s2.id)
            if c1 is None or c2 is None:
                continue
            if drop_self_loops and c1 == c2:
                continue
            u = nodes_by_cluster[c1].node_id
            v = nodes_by_cluster[c2].node_id
            edge_trajectories[(u, v)].append(trajectory_id)

    for (u, v), trajs in edge_trajectories.items():
        graph.add_edge(
            u,
            v,
            trajectories=sorted(set(trajs)),
            count=len(trajs),
            virtual=False,
        )

    dropped_back_edges: List[Tuple[str, str]] = []
    if enforce_dag:
        dropped_back_edges = _enforce_dag(graph)

    # Optionally add synthetic START and END super-nodes so that:
    #   - exactly 1 source (START) with out-degree = #distinct entry clusters
    #   - exactly 1 leaf  (END)   with in-degree = #distinct exit clusters
    #   - the `count` on edges into/out of START/END sums to the number of trajectories
    virtual_added = False
    if add_virtual_endpoints and (first_clusters or last_clusters):
        start_to_target: Dict[str, List[str]] = defaultdict(list)
        for tid, fcid in first_clusters.items():
            start_to_target[nodes_by_cluster[fcid].node_id].append(tid)
        source_from: Dict[str, List[str]] = defaultdict(list)
        for tid, lcid in last_clusters.items():
            source_from[nodes_by_cluster[lcid].node_id].append(tid)

        if start_to_target:
            graph.add_node(
                VIRTUAL_START_ID,
                node_obj=None,
                canonical_text="START",
                support_count=len(first_clusters),
                trajectory_ids=sorted(first_clusters.keys()),
                supporting_steps=[],
                representative_raw_text="(virtual entry node — connects to each trajectory's first reasoning cluster)",
                virtual=True,
            )
            for tgt, trajs in start_to_target.items():
                graph.add_edge(
                    VIRTUAL_START_ID, tgt,
                    trajectories=sorted(set(trajs)),
                    count=len(trajs),
                    virtual=True,
                )

        if source_from:
            graph.add_node(
                VIRTUAL_END_ID,
                node_obj=None,
                canonical_text="END",
                support_count=len(last_clusters),
                trajectory_ids=sorted(last_clusters.keys()),
                supporting_steps=[],
                representative_raw_text="(virtual exit node — receives from each trajectory's last reasoning cluster)",
                virtual=True,
            )
            for src, trajs in source_from.items():
                graph.add_edge(
                    src, VIRTUAL_END_ID,
                    trajectories=sorted(set(trajs)),
                    count=len(trajs),
                    virtual=True,
                )
        virtual_added = True

    # Compute transition probabilities per source node (after DAG enforcement).
    for u in graph.nodes:
        outgoing = list(graph.out_edges(u, data=True))
        total = sum(d.get("count", 0) for _, _, d in outgoing) or 1
        for _, v, d in outgoing:
            d["probability"] = d.get("count", 0) / total

    graph.graph["dropped_back_edges"] = dropped_back_edges
    graph.graph["virtual_endpoints"] = virtual_added
    graph.graph["trajectory_first_clusters"] = {
        tid: nodes_by_cluster[c].node_id for tid, c in first_clusters.items()
    }
    graph.graph["trajectory_last_clusters"] = {
        tid: nodes_by_cluster[c].node_id for tid, c in last_clusters.items()
    }
    graph.graph["transitions_per_trajectory"] = dict(transitions_per_traj)
    return graph


def graph_to_dump(
    graph: nx.DiGraph,
    trajectories: List[str],
    prompt: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
):
    """Serialize the graph + bookkeeping into a GraphDump."""
    from .models import GraphDump  # local import to avoid cycles

    nodes: List[ReasoningNode] = []
    for n in graph.nodes:
        node_obj = graph.nodes[n].get("node_obj")
        if node_obj is not None:
            nodes.append(node_obj)
            continue
        # Synthetic virtual node (START/END) — build a ReasoningNode shell.
        d = graph.nodes[n]
        nodes.append(
            ReasoningNode(
                node_id=n,
                canonical_text=d.get("canonical_text") or "",
                supporting_steps=list(d.get("supporting_steps") or []),
                trajectory_ids=list(d.get("trajectory_ids") or []),
                support_count=int(d.get("support_count", 0)),
                representative_raw_text=d.get("representative_raw_text"),
                virtual=bool(d.get("virtual", False)),
                is_conclusion=bool(d.get("is_conclusion", False)),
                conclusion_trajectories=list(d.get("conclusion_trajectories") or []),
            )
        )

    edges: List[ReasoningEdge] = []
    for u, v, data in graph.edges(data=True):
        edges.append(
            ReasoningEdge(
                source=u,
                target=v,
                trajectories=data.get("trajectories") or [],
                count=int(data.get("count", 0)),
                probability=data.get("probability"),
                virtual=bool(data.get("virtual", False)),
            )
        )

    return GraphDump(
        prompt=prompt,
        trajectories=sorted(set(trajectories)),
        nodes=nodes,
        edges=edges,
        metrics=metrics or {},
        config=config or {},
    )
