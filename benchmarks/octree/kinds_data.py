"""Synthetic point clouds, meshes and networks for ``bench_kinds.py``.

All three share one uneven density: a few clusters with broad (Student-t)
tails over a sparse uniform background, in a ``BOX``-wide cube.  Each
generator returns about ``n`` vertices and an object id per vertex.

* point cloud: points drawn from the density; each cluster's points (and the
  background's) split at random into objects of ~1,000 points, spread over
  their cluster;
* mesh: closed blobby surfaces (icospheres, 162 to 2,562 vertices, their
  radius bumped by low-frequency noise) centred on points drawn from the
  density; one object per surface;
* network: branching trees (a persistent random walk that forks), 100 to
  10,000 nodes, rooted at points drawn from the density; one object per tree.
"""

from __future__ import annotations

import numpy as np

BOX = 1000.0
KINDS = ("points", "mesh", "network")


def _density(rng: np.random.Generator, n: int, clusters: int = 6) -> np.ndarray:
    centres = rng.uniform(0.2 * BOX, 0.8 * BOX, (clusters, 3))
    scale = rng.uniform(0.02, 0.06, clusters) * BOX
    weight = rng.dirichlet(np.ones(clusters) * 2)
    n_bg = n // 10
    lab = rng.choice(clusters, n - n_bg, p=weight)
    pts = centres[lab] + rng.standard_t(3, (len(lab), 3)) * scale[lab, None]
    pts = np.concatenate([pts, rng.uniform(0, BOX, (n_bg, 3))])
    lab = np.concatenate([lab, np.full(n_bg, clusters)])
    keep = np.all((pts >= 0) & (pts < BOX), axis=1)
    return pts[keep], lab[keep]


def points(n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    pts, lab = _density(rng, n)
    ids = np.empty(len(pts), dtype=np.int64)
    nxt = 0
    for v in np.unique(lab):
        idx = rng.permutation(np.flatnonzero(lab == v))
        k = max(1, round(len(idx) / 1000))
        ids[idx] = nxt + np.arange(len(idx)) % k
        nxt += k
    return {"positions": pts.astype(np.float32), "object_ids": ids}


def _icosphere(level: int) -> tuple[np.ndarray, np.ndarray]:
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                  [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                  [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    for _ in range(level):
        edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        uniq, inv = np.unique(edges, axis=0, return_inverse=True)
        mid = len(v) + inv.reshape(3, -1).T  # the midpoints of each face's edges (01, 12, 20)
        v = np.concatenate([v, (v[uniq[:, 0]] + v[uniq[:, 1]]) / 2])
        a, b, c = f.T
        ab, bc, ca = mid.T
        f = np.concatenate([np.stack(x, 1) for x in ([a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca])])
    return v / np.linalg.norm(v, axis=1, keepdims=True), f


def mesh(n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    spheres = {lv: _icosphere(lv) for lv in (2, 3, 4)}  # 162, 642, 2,562 vertices
    centres, _ = _density(rng, max(64, n // 300))
    verts, faces, ids, total = [], [], [], 0
    for i, c in enumerate(rng.permutation(centres)):
        if total >= n:
            break
        v, f = spheres[int(rng.choice([2, 3, 4], p=[0.5, 0.35, 0.15]))]
        r = 3.0 * np.sqrt(len(v)) ** 0.5 * rng.uniform(0.6, 1.6)  # bigger surfaces, more vertices
        k = rng.normal(size=(4, 3))
        bump = 1 + 0.25 * np.tanh(np.sin(v @ k.T * 2 + rng.uniform(0, 6, 4)).sum(1) / 2)
        p = c + v * (r * bump)[:, None]
        if np.any(p < 0) or np.any(p >= BOX):
            continue
        faces.append(f + total)
        verts.append(p)
        ids.append(np.full(len(v), i))
        total += len(v)
    return {"positions": np.concatenate(verts).astype(np.float32), "faces": np.concatenate(faces).astype(np.int64),
            "object_ids": np.concatenate(ids)}


def network(n: int, seed: int = 0, step: float = 2.0) -> dict:
    rng = np.random.default_rng(seed)
    roots, _ = _density(rng, max(64, n // 500))
    pos, edges, ids, total = [], [], [], 0
    for i, root in enumerate(rng.permutation(roots)):
        if total >= n:
            break
        size = int(np.exp(rng.uniform(np.log(100), np.log(10_000))))
        p = np.empty((size, 3))
        parent = np.full(size, -1)
        p[0] = root
        tips = [(0, rng.normal(size=3))]  # (node, heading)
        m = 1
        while m < size and tips:
            j = int(rng.integers(len(tips)))
            node, d = tips[j]
            d = d / np.linalg.norm(d) + rng.normal(scale=0.3, size=3)
            d /= np.linalg.norm(d)
            q = p[node] + step * d
            if np.any(q < 0) or np.any(q >= BOX):
                tips.pop(j)
                continue
            p[m], parent[m] = q, node
            tips[j] = (m, d)
            if rng.random() < 0.02:  # a fork
                tips.append((m, d + rng.normal(scale=1.0, size=3)))
            m += 1
        p, parent = p[:m], parent[:m]
        child = np.arange(1, m)
        edges.append(np.stack([parent[child], child], 1) + total)
        pos.append(p)
        ids.append(np.full(m, i))
        total += m
    return {"positions": np.concatenate(pos).astype(np.float32), "edges": np.concatenate(edges).astype(np.int64),
            "object_ids": np.concatenate(ids)}


GENERATORS = {"points": points, "mesh": mesh, "network": network}
