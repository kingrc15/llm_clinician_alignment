import unittest

import numpy as np

from reasoning_tree.app.clustering import cluster_steps
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


class ClusteringTests(unittest.TestCase):
    def test_merges_near_duplicates(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.999, 0.04472])
        b = b / np.linalg.norm(b)
        c = np.array([0.0, 1.0])
        steps = [
            _step(0, "A", "high troponin indicates myocardial injury", a),
            _step(1, "B", "elevated troponin suggests cardiac damage", b),
            _step(2, "A", "patient lives alone", c),
        ]
        clusters = cluster_steps(steps, similarity_threshold=0.95)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(sorted(len(v) for v in clusters.values()), [1, 2])

    def test_separates_dissimilar_steps(self):
        vecs = np.eye(4)
        steps = [_step(i, "X", f"item {i}", vecs[i]) for i in range(4)]
        clusters = cluster_steps(steps, similarity_threshold=0.9)
        self.assertEqual(len(clusters), 4)

    def test_requires_embeddings(self):
        s = ReasoningStep(id="x", trajectory_id="t", step_index=0, raw_text="a", canonical_text="a")
        with self.assertRaises(ValueError):
            cluster_steps([s])


if __name__ == "__main__":
    unittest.main()
