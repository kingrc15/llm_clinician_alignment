import unittest

import numpy as np
import networkx as nx

from reasoning_tree.app.clustering import cluster_steps
from reasoning_tree.app.graph_builder import build_reasoning_graph
from reasoning_tree.app.models import ReasoningStep


def _step(idx, traj, text, emb):
    return ReasoningStep(
        id=f"{traj}-{idx}",
        trajectory_id=traj,
        step_index=idx,
        raw_text=text,
        canonical_text=text.lower(),
        embedding=list(map(float, emb)),
    )


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


class GraphBuilderTests(unittest.TestCase):
    def test_shared_prefix_and_reconvergence(self):
        A = _unit([1.0, 0.0, 0.0, 0.0])
        B1 = _unit([0.0, 1.0, 0.0, 0.0])
        B2 = _unit([0.0, 0.0, 1.0, 0.0])
        D = _unit([0.0, 0.0, 0.0, 1.0])

        traj_alpha = [
            _step(0, "alpha", "A", A),
            _step(1, "alpha", "B1", B1),
            _step(2, "alpha", "D", D),
        ]
        traj_beta = [
            _step(0, "beta", "A", A),
            _step(1, "beta", "B2", B2),
            _step(2, "beta", "D", D),
        ]
        all_steps = traj_alpha + traj_beta
        clusters = cluster_steps(all_steps, similarity_threshold=0.95)
        self.assertEqual(len(clusters), 4)

        g = build_reasoning_graph(clusters, [traj_alpha, traj_beta])
        self.assertTrue(nx.is_directed_acyclic_graph(g))
        self.assertEqual(g.number_of_nodes(), 4)
        self.assertEqual(g.number_of_edges(), 4)

        shared_starts = [
            n for n in g.nodes
            if g.in_degree(n) == 0 and int(g.nodes[n]["support_count"]) == 2
        ]
        shared_ends = [
            n for n in g.nodes
            if g.out_degree(n) == 0 and int(g.nodes[n]["support_count"]) == 2
        ]
        self.assertEqual(len(shared_starts), 1)
        self.assertEqual(len(shared_ends), 1)

    def test_self_loops_dropped(self):
        A = _unit([1.0, 0.0, 0.0])
        A_dup = _unit([0.999, 0.04472, 0.0])
        B = _unit([0.0, 1.0, 0.0])
        traj = [
            _step(0, "t", "A", A),
            _step(1, "t", "A again", A_dup),
            _step(2, "t", "B", B),
        ]
        clusters = cluster_steps(traj, similarity_threshold=0.95)
        g = build_reasoning_graph(clusters, [traj])
        self.assertTrue(all(u != v for u, v in g.edges))

    def test_edge_probabilities_sum_to_one_per_source(self):
        A = _unit([1.0, 0.0, 0.0, 0.0])
        B = _unit([0.0, 1.0, 0.0, 0.0])
        C = _unit([0.0, 0.0, 1.0, 0.0])
        D = _unit([0.0, 0.0, 0.0, 1.0])

        t1 = [_step(0, "t1", "A", A), _step(1, "t1", "B", B), _step(2, "t1", "D", D)]
        t2 = [_step(0, "t2", "A", A), _step(1, "t2", "C", C), _step(2, "t2", "D", D)]
        clusters = cluster_steps(t1 + t2, similarity_threshold=0.95)
        g = build_reasoning_graph(clusters, [t1, t2])

        for u in g.nodes:
            out = list(g.out_edges(u, data=True))
            if not out:
                continue
            probs = sum(d.get("probability", 0.0) for _, _, d in out)
            self.assertAlmostEqual(probs, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
