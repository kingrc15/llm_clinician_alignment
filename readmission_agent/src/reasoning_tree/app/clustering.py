"""Phase 4 — Semantic clustering.

Agglomerative clustering on cosine distance (= 1 - cosine similarity) with a
configurable distance threshold. With L2-normalized embeddings, cosine
similarity equals the dot product. Clustering is deterministic for a fixed
input order: sklearn's AgglomerativeClustering with 'average' linkage and a
fixed `distance_threshold` is order-stable.
"""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .models import ReasoningStep


_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokens(text: str) -> set:
    if not text:
        return set()
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def post_merge_clusters_by_jaccard(
    clusters: Dict[int, List[ReasoningStep]],
    jaccard_threshold: float = 0.6,
) -> Dict[int, List[ReasoningStep]]:
    """Second-pass merge: union clusters whose canonical texts share lots of tokens.

    Uses the shortest-canonical text in each cluster as the cluster's "representative"
    string, then merges any two clusters whose representatives have Jaccard token
    overlap >= ``jaccard_threshold``. Union-find over cluster ids; deterministic
    for a fixed input order.
    """
    if not clusters or jaccard_threshold >= 1.0:
        return clusters

    ids = sorted(clusters.keys())
    reps: Dict[int, set] = {}
    for cid in ids:
        steps = clusters[cid]
        rep_step = min(
            steps,
            key=lambda s: (len(s.canonical_text or s.raw_text or ""), s.step_index),
        )
        reps[cid] = _tokens(rep_step.canonical_text or rep_step.raw_text)

    parent = {cid: cid for cid in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Always merge larger id into smaller so the result is deterministic.
        lo, hi = sorted((ra, rb))
        parent[hi] = lo

    for i, ca in enumerate(ids):
        ta = reps[ca]
        if not ta:
            continue
        for cb in ids[i + 1:]:
            tb = reps[cb]
            if not tb:
                continue
            if _jaccard(ta, tb) >= jaccard_threshold:
                union(ca, cb)

    merged: Dict[int, List[ReasoningStep]] = {}
    remap: Dict[int, int] = {}
    next_id = 0
    for cid in ids:
        root = find(cid)
        if root not in remap:
            remap[root] = next_id
            merged[next_id] = []
            next_id += 1
        merged[remap[root]].extend(clusters[cid])
    return merged


def _cluster_pool(
    steps: List[ReasoningStep],
    similarity_threshold: float,
    linkage: str,
) -> Dict[int, List[ReasoningStep]]:
    """Cluster a single pool of steps; cluster ids start at 0 and are dense."""
    if not steps:
        return {}
    X = np.asarray([s.embedding for s in steps], dtype=np.float32)
    if X.shape[0] == 1:
        return {0: list(steps)}

    distance_threshold = max(0.0, 1.0 - float(similarity_threshold))
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage=linkage,
        distance_threshold=distance_threshold,
    )
    labels = model.fit_predict(X)

    remap: Dict[int, int] = {}
    next_id = 0
    buckets: Dict[int, List[ReasoningStep]] = {}
    for step, lbl in zip(steps, labels):
        lbl = int(lbl)
        if lbl not in remap:
            remap[lbl] = next_id
            buckets[next_id] = []
            next_id += 1
        buckets[remap[lbl]].append(step)
    return buckets


def cluster_steps(
    steps: List[ReasoningStep],
    similarity_threshold: float = 0.83,
    linkage: str = "average",
    post_merge_jaccard: float = 0.0,
    partition_conclusions: bool = True,
) -> Dict[int, List[ReasoningStep]]:
    """Cluster steps by semantic similarity.

    Parameters
    ----------
    steps : list[ReasoningStep]
        Each step must already have an embedding written via embed_steps().
    similarity_threshold : float
        Cosine similarity above which two steps are considered the "same"
        reasoning node. Recommended range 0.80–0.88. Internally we convert
        this to a distance threshold of (1 - similarity_threshold).
    linkage : str
        Linkage method for AgglomerativeClustering ('average' is a good default).
    partition_conclusions : bool
        If True (default), conclusion steps are clustered in a separate pool
        from non-conclusion reasoning steps. This guarantees that the cluster
        containing each trajectory's appended ``principal_reason`` step has
        in-degree at most equal to the number of trajectories (because the
        only incoming edges are the transitions from each trajectory's
        immediately preceding reasoning step). Without partitioning, an
        earlier reasoning step that happened to be a paraphrase of the
        principal reason would also incoming-edge the conclusion cluster.

    Returns
    -------
    dict[int, list[ReasoningStep]]
        cluster_id -> list of steps in that cluster (cluster_id is a small
        deterministic integer based on the canonical first step encountered).
    """
    if not steps:
        return {}

    missing = [s.id for s in steps if s.embedding is None]
    if missing:
        raise ValueError(
            f"cluster_steps: {len(missing)} steps are missing embeddings "
            f"(first: {missing[0]!r}). Call embed_steps() first."
        )

    if not (0.0 < similarity_threshold <= 1.0):
        raise ValueError(
            f"similarity_threshold must be in (0, 1], got {similarity_threshold}"
        )

    if partition_conclusions:
        reasoning_pool = [s for s in steps if not getattr(s, "is_conclusion", False)]
        conclusion_pool = [s for s in steps if getattr(s, "is_conclusion", False)]
        reasoning_buckets = _cluster_pool(reasoning_pool, similarity_threshold, linkage)
        conclusion_buckets = _cluster_pool(conclusion_pool, similarity_threshold, linkage)
        buckets: Dict[int, List[ReasoningStep]] = {}
        for cid, ss in reasoning_buckets.items():
            buckets[cid] = ss
        offset = (max(buckets.keys()) + 1) if buckets else 0
        for cid, ss in conclusion_buckets.items():
            buckets[cid + offset] = ss
    else:
        buckets = _cluster_pool(steps, similarity_threshold, linkage)

    if post_merge_jaccard and post_merge_jaccard > 0:
        # Restrict post-merge to never merge across the conclusion boundary.
        if partition_conclusions:
            r_only = {k: v for k, v in buckets.items() if not any(getattr(s, "is_conclusion", False) for s in v)}
            c_only = {k: v for k, v in buckets.items() if any(getattr(s, "is_conclusion", False) for s in v)}
            r_merged = post_merge_clusters_by_jaccard(r_only, jaccard_threshold=float(post_merge_jaccard))
            c_merged = post_merge_clusters_by_jaccard(c_only, jaccard_threshold=float(post_merge_jaccard))
            buckets = {}
            for cid, ss in r_merged.items():
                buckets[cid] = ss
            offset = (max(buckets.keys()) + 1) if buckets else 0
            for cid, ss in c_merged.items():
                buckets[cid + offset] = ss
        else:
            buckets = post_merge_clusters_by_jaccard(buckets, jaccard_threshold=float(post_merge_jaccard))

    return buckets


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity for inspection/debugging."""
    if X.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    # Assumes rows are L2-normalized; otherwise normalize defensively.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    Xn = X / norms
    return (Xn @ Xn.T).astype(np.float32)
