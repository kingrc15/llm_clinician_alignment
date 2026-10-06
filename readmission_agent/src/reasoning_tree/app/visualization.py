"""Phase 6 — Visualization.

Emit an interactive HTML view of the reasoning DAG with pyvis. Nodes are
sized and colored by support count; edges are weighted by their transition
count and labeled with probability. A JSON dump of the graph is also written
for downstream tooling.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import networkx as nx
from pyvis.network import Network

from .utils import ensure_dir


def _color_for_support(support: int, max_support: int) -> str:
    """Map support count to a blue-to-red gradient (more support = warmer)."""
    if max_support <= 0:
        ratio = 0.0
    else:
        ratio = max(0.0, min(1.0, support / max_support))
    r = int(round(63 + ratio * (224 - 63)))
    g = int(round(127 - ratio * (127 - 38)))
    b = int(round(191 - ratio * (191 - 38)))
    return f"#{r:02x}{g:02x}{b:02x}"


def _short(text: str, n: int = 90) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "\u2026"


def graph_to_json_dict(graph: nx.DiGraph) -> Dict[str, Any]:
    """Lightweight JSON dump compatible with d3/cytoscape/etc."""
    # Pre-compute trajectory-weighted degrees (= sum of edge counts).
    weighted_in: Dict[Any, int] = {n: 0 for n in graph.nodes}
    weighted_out: Dict[Any, int] = {n: 0 for n in graph.nodes}
    for u, v, data in graph.edges(data=True):
        c = int(data.get("count", 0))
        weighted_out[u] += c
        weighted_in[v] += c

    nodes: List[Dict[str, Any]] = []
    for n in graph.nodes:
        d = graph.nodes[n]
        nodes.append({
            "id": n,
            "canonical_text": d.get("canonical_text"),
            "representative_raw_text": d.get("representative_raw_text"),
            "support_count": int(d.get("support_count", 0)),
            "trajectory_ids": list(d.get("trajectory_ids", []) or []),
            "supporting_steps": list(d.get("supporting_steps", []) or []),
            "in_degree": graph.in_degree(n),
            "out_degree": graph.out_degree(n),
            "trajectory_in_degree": weighted_in.get(n, 0),
            "trajectory_out_degree": weighted_out.get(n, 0),
            "virtual": bool(d.get("virtual", False)),
            "is_conclusion": bool(d.get("is_conclusion", False)),
            "conclusion_trajectories": list(d.get("conclusion_trajectories") or []),
        })
    edges: List[Dict[str, Any]] = []
    for u, v, data in graph.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "count": int(data.get("count", 0)),
            "probability": data.get("probability"),
            "trajectories": list(data.get("trajectories") or []),
            "virtual": bool(data.get("virtual", False)),
        })
    return {
        "nodes": nodes,
        "edges": edges,
        "dropped_back_edges": [list(e) for e in graph.graph.get("dropped_back_edges", []) or []],
    }


def visualize_graph(
    graph: nx.DiGraph,
    output_path: str,
    title: Optional[str] = None,
    height: str = "1000px",
    width: str = "100%",
    layout: str = "auto",
    node_spacing: int = 360,
    level_separation: int = 260,
    tree_spacing: int = 360,
    label_width: int = 260,
    label_chars: int = 110,
) -> str:
    """Write an interactive HTML visualization. Returns the path written.

    Parameters
    ----------
    layout : str
        "hierarchical" (default) puts the START at the top and ANSWER at the
        bottom, levels arranged by directed sort. "physics" runs a
        force-directed simulation (better for very dense graphs with shared
        nodes).
    node_spacing / level_separation / tree_spacing : int
        Pixel spacing knobs for the hierarchical layout.
    label_width : int
        Max width (px) before node labels wrap onto a new line. Larger =
        wider node boxes, less wrapping.
    label_chars : int
        Maximum characters of canonical text included in each label before
        truncating with an ellipsis. Set high (e.g. 200) for verbose labels.
    """
    ensure_dir(os.path.dirname(os.path.abspath(output_path)))

    net = Network(
        height=height,
        width=width,
        directed=True,
        notebook=False,
        cdn_resources="in_line",
        bgcolor="#ffffff",
        font_color="#222222",
    )

    max_support = max((int(graph.nodes[n].get("support_count", 0)) for n in graph.nodes), default=1)

    # Per-node trajectory-weighted degrees: how many trajectory transits
    # enter/leave each node. Each edge's `count` is the number of trajectories
    # that share that transition; the weighted in-degree is the sum, which is
    # also the number of times trajectories enter the node. For a node visited
    # at most once per trajectory, weighted_in == support_count.
    weighted_in = {n: 0 for n in graph.nodes}
    weighted_out = {n: 0 for n in graph.nodes}
    for u, v, data in graph.edges(data=True):
        c = int(data.get("count", 0))
        weighted_out[u] += c
        weighted_in[v] += c

    for n in graph.nodes:
        d = graph.nodes[n]
        support = int(d.get("support_count", 0))
        canonical = d.get("canonical_text") or ""
        raw = d.get("representative_raw_text") or canonical
        is_virtual = bool(d.get("virtual"))
        is_conclusion = bool(d.get("is_conclusion"))
        conclusion_trajs = d.get("conclusion_trajectories") or []
        if is_virtual:
            label = f"{canonical} (x{support})"
            color = "#222831"
            shape = "diamond"
            size = 34
            font = {"color": "#ffffff", "size": 16, "face": "Arial"}
        elif is_conclusion:
            label = f"★ ANSWER [{len(conclusion_trajs)}/{support}]\n{_short(canonical, label_chars)}"
            color = {"background": "#f5b301", "border": "#7a4f00",
                     "highlight": {"background": "#ffd24d", "border": "#7a4f00"}}
            shape = "box"
            size = 32 + 6 * len(conclusion_trajs)
            font = {"color": "#222222", "size": 15, "face": "Arial", "bold": True}
        else:
            label = f"[{support}] {_short(canonical, label_chars)}"
            color = _color_for_support(support, max_support)
            shape = "box"
            size = 18 + 6 * support
            font = {"color": "#ffffff" if support >= 2 else "#222222",
                    "size": 13, "face": "Arial"}
        # Weighted (trajectory-count) and unweighted (distinct-cluster-count)
        # degrees give complementary information: the weighted form equals the
        # number of trajectory transits at this node; the unweighted form is the
        # number of distinct predecessor/successor clusters.
        w_in, w_out = weighted_in.get(n, 0), weighted_out.get(n, 0)
        u_in, u_out = graph.in_degree(n), graph.out_degree(n)
        tooltip_lines = [
            f"node_id: {n}",
            f"support_count: {support}",
            f"trajectories: {', '.join(d.get('trajectory_ids') or [])}",
            f"in/out degree (trajectories): {w_in} / {w_out}",
            f"in/out degree (distinct neighbors): {u_in} / {u_out}",
        ]
        if is_conclusion:
            tooltip_lines.append(
                f"conclusion_for: {', '.join(conclusion_trajs)} "
                f"({len(conclusion_trajs)}/{support} of supporters)"
            )
        tooltip_lines += [
            "",
            "canonical:",
            canonical,
            "",
            "representative:",
            raw,
        ]
        net.add_node(
            n,
            label=label,
            title="\n".join(tooltip_lines),
            color=color,
            size=size,
            shape=shape,
            font=font,
            # Box-shape labels wrap when widthConstraint is set; "minimum" makes
            # short labels still look like consistent-sized cards.
            widthConstraint={"minimum": max(140, label_width // 2),
                             "maximum": int(label_width)},
            margin=10,
        )

    for u, v, data in graph.edges(data=True):
        count = int(data.get("count", 0))
        prob = data.get("probability")
        is_virtual = bool(data.get("virtual"))
        prob_str = f"{prob:.2f}" if isinstance(prob, (int, float)) else ""
        edge_label = f"x{count}" + (f" ({prob_str})" if prob_str else "")
        tooltip = (
            f"count: {count}\n"
            f"probability: {prob}\n"
            f"trajectories: {', '.join(data.get('trajectories') or [])}"
        )
        net.add_edge(
            u,
            v,
            value=count,
            title=tooltip,
            label=edge_label,
            arrows="to",
            smooth={"type": "cubicBezier", "roundness": 0.3},
            dashes=is_virtual,
            color={"color": "#888888" if not is_virtual else "#3b3b3b"},
        )

    # "auto" picks hierarchical for DAGs (clean top-down chain) and physics
    # for cyclic graphs (vis-network's hierarchical layout can't sensibly
    # layer a graph with back-edges, which produces giant empty rows). This
    # is the case after `restore_trajectory_flow` re-adds DAG-enforcement-
    # dropped transitions.
    effective_layout = layout
    if layout == "auto":
        effective_layout = "hierarchical" if nx.is_directed_acyclic_graph(graph) else "physics"

    if effective_layout == "physics":
        options = {
            "layout": {"hierarchical": {"enabled": False}},
            "physics": {
                "enabled": True,
                "barnesHut": {
                    "gravitationalConstant": -12000,
                    "centralGravity": 0.1,
                    "springLength": 220,
                    "springConstant": 0.04,
                    "damping": 0.6,
                    "avoidOverlap": 1.0,
                },
                "stabilization": {"enabled": True, "iterations": 600},
            },
            "edges": {
                "color": {"color": "#888888", "highlight": "#444444"},
                "font": {"size": 10, "align": "horizontal"},
                "smooth": {"type": "dynamic"},
            },
            "nodes": {"font": {"size": 13, "face": "Arial"}, "borderWidth": 1},
            "interaction": {"hover": True, "tooltipDelay": 100, "navigationButtons": True, "keyboard": True},
        }
    else:
        options = {
            "layout": {
                "hierarchical": {
                    "enabled": True,
                    "direction": "UD",
                    "sortMethod": "directed",
                    "levelSeparation": int(level_separation),
                    "nodeSpacing": int(node_spacing),
                    "treeSpacing": int(tree_spacing),
                    "blockShifting": True,
                    "edgeMinimization": True,
                    "parentCentralization": True,
                }
            },
            "physics": {"enabled": False},
            "edges": {
                "color": {"color": "#888888", "highlight": "#444444"},
                "font": {"size": 10, "align": "horizontal"},
                "smooth": {"type": "cubicBezier", "forceDirection": "vertical", "roundness": 0.4},
            },
            "nodes": {"font": {"size": 13, "face": "Arial"}, "borderWidth": 1, "shapeProperties": {"interpolation": False}},
            "interaction": {"hover": True, "tooltipDelay": 100, "navigationButtons": True, "keyboard": True},
        }
    net.set_options(json.dumps(options))

    title_html = ""
    if title:
        title_html = f"<h2 style='font-family:Arial;margin:8px'>{title}</h2>"

    html = net.generate_html(notebook=False)
    if title_html:
        html = html.replace("<body>", f"<body>{title_html}", 1)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
