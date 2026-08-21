"""Timing, statistics, and synthetic-data helpers for the paper figure.

Kept deliberately small and dependency-light (``numpy`` only) so that
``run_sweep.py`` can be executed on a headless reference machine with
nothing installed beyond the core package and the competitor readers.

Every geometry generator takes a target **vertex count** ``n`` and
returns something the corresponding ``zarr_vectors.types.*`` writer
accepts.  The generators are the single source of truth for what
"a dataset of size N" means -- both the zarr-vectors side and the
competitor side are built from the same arrays, so a size or timing
difference is never an artefact of different input data.

Objects per dataset follow from the vertex target and a fixed
vertices-per-object constant (``VERTS_PER_OBJECT``), so "N vertices" also
pins how many objects there are, and every measured row can report both.
"""

from __future__ import annotations

import gzip
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------
# Domain constants -- shared by every measurement in the sweep.
# --------------------------------------------------------------------

DOMAIN = 1000.0                      # cube side, in store units
BOUNDS = ([0.0, 0.0, 0.0], [DOMAIN] * 3)
CHUNK = (200.0, 200.0, 200.0)        # 5^3 = 125 chunks over the domain
BIN = (50.0, 50.0, 50.0)
SEED = 0

# Vertices per object, per geometry.  Fixing these makes "N vertices"
# mean the same amount of data on both sides of every comparison, and it
# makes the object count a function of the vertex count: an object is the
# same size at every point on the curve, so a curve against vertices is
# also a curve against objects, at a fixed conversion the key names.
VERTS_PER_STREAMLINE = 12
VERTS_PER_SKELETON = 200
MESH_PATCH_SIDE = 12                 # 12x12 grid -> 144 verts, 242 faces

# Vertices per object, keyed the same way as ``GENERATORS``.  A point is
# its own object, so for a point cloud "objects" and "vertices" are the
# same axis.  ``gen_graph`` counts one object per node for the same
# reason -- which is what the CSV records -- even though the store it
# writes holds the nodes as a single graph object; graphs are not in the
# object-swept block, so nothing is measured against that number.
VERTS_PER_OBJECT = {
    "points": 1,
    "streamlines": VERTS_PER_STREAMLINE,
    "skeletons": VERTS_PER_SKELETON,
    "graphs": 1,
    "meshes": MESH_PATCH_SIDE ** 2,
}


# --------------------------------------------------------------------
# Timing + statistics
# --------------------------------------------------------------------

# Student's t, two-sided 95%, by degrees of freedom.  Hard-coded so the
# sweep does not need scipy on the reference machine.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 19: 2.093, 24: 2.064, 29: 2.045,
    39: 2.023, 49: 2.010, 59: 2.001, 99: 1.984, 199: 1.972,
}
_T95_INF = 1.960


def t95(df: int) -> float:
    """Two-sided 95% Student's-t critical value for ``df``.

    Interpolation is downward -- the largest tabulated ``df`` at or below
    the one asked for -- so an untabulated value always yields a slightly
    *wider* interval than the true one rather than a narrower one.
    """
    if df in _T95:
        return _T95[df]
    below = [k for k in _T95 if k < df]
    if not below:
        return _T95[min(_T95)]
    return _T95[max(below)] if df <= max(_T95) else _T95_INF


def mean_ci95(samples) -> tuple[float, float]:
    """``(mean, half-width)`` of a 1-D sample, Student's t at df = n-1."""
    arr = np.asarray(samples, dtype=float)
    if arr.size < 2:
        return (float(arr.mean()) if arr.size else float("nan")), 0.0
    hw = t95(arr.size - 1) * arr.std(ddof=1) / np.sqrt(arr.size)
    return float(arr.mean()), float(hw)


def timed(fn, *args, **kwargs) -> tuple[float, object]:
    """Call ``fn``; return ``(elapsed_seconds, return_value)``."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return time.perf_counter() - t0, out


MIN_TOTAL_S = 0.5      # keep sampling a cheap operation until this much
MAX_RUNS = 200         # ... but never more than this many repeats


def repeat(fn, n_runs: int, *, setup=None, teardown=None,
           min_total: float = MIN_TOTAL_S) -> tuple[float, float, int]:
    """Time ``fn()``; return ``(mean, ci_halfwidth, n)``.

    Runs at least ``n_runs`` times, then keeps going while the samples so
    far total less than ``min_total`` seconds, up to ``MAX_RUNS``.  A
    sub-millisecond operation timed seven times is dominated by clock
    noise -- the confidence interval comes out as wide as the mean, which
    is a statement about the sample size rather than about the operation.
    Cheap calls therefore earn more repeats at negligible cost, while an
    expensive one still stops at ``n_runs``.

    ``setup(run_index)`` runs before each timed call and its return value
    is passed to ``fn``; ``teardown(setup_result)`` runs after.  Neither
    is included in the timing.
    """
    samples: list[float] = []
    i = 0
    while True:
        ctx = setup(i) if setup is not None else None
        try:
            dt, _ = timed(fn, ctx) if setup is not None else timed(fn)
            samples.append(dt)
        finally:
            if teardown is not None:
                teardown(ctx)
        i += 1
        if i >= n_runs and (sum(samples) >= min_total or i >= MAX_RUNS):
            break
    m, hw = mean_ci95(samples)
    return m, hw, len(samples)


# --------------------------------------------------------------------
# Filesystem helpers
# --------------------------------------------------------------------

def fstype(path) -> str | None:
    """Filesystem type under ``path``, or None if it cannot be read.

    Worth recording next to the numbers: the scratch directory decides
    how a chunked store's thousands of small files behave, and a run on
    NTFS-via-ntfs3 is not the same measurement as a run on ext4 -- the
    precomputed competitor, which writes one file per annotation, cannot
    even complete on the former.
    """
    try:
        return subprocess.run(
            ["findmnt", "-no", "FSTYPE", "--target", str(path)],
            capture_output=True, text=True, check=True).stdout.strip() or None
    except Exception:
        return None


def path_bytes(path) -> int:
    """On-disk size of a file or of a directory tree, in bytes."""
    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def gzip_bytes(path, level: int = 6) -> int:
    """Size of ``path`` after gzip, without keeping the compressed copy.

    Used to give every text competitor its best-case storage number, so
    the size comparison is not merely "binary versus ASCII".
    """
    n = 0
    src = Path(path)
    with open(src, "rb") as fh:
        comp = gzip.compress(fh.read(), compresslevel=level)
        n = len(comp)
    return n


class Workspace:
    """Scratch directory that cleans itself up."""

    def __init__(self, root=None):
        self.root = Path(tempfile.mkdtemp(prefix="zv_paper_", dir=root))

    def store(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d / "store.zarrvectors"

    def file(self, name: str, suffix: str = "") -> Path:
        """A path for one competitor artefact, in its own directory.

        The suffix is not cosmetic: ``trx_save`` picks its container from
        the extension and writes a *directory* of loose arrays when it
        does not see ``.trx``, which is the format's content but not the
        file anyone ships.  Everything else is written by an explicit
        writer, but they all get their real extension so the artefact on
        disk is the artefact being claimed.
        """
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d / f"data{suffix}"

    def clear(self, name: str) -> None:
        shutil.rmtree(self.root / name, ignore_errors=True)

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


# --------------------------------------------------------------------
# Synthetic datasets
# --------------------------------------------------------------------

def gen_points(n: int, seed: int = SEED) -> dict:
    """``n`` uniformly random points, one object per point.

    Uniform random is deliberately the *worst* case for both sides: it
    defeats compression (no spatial correlation to exploit) and it
    defeats chunk locality (every chunk is equally populated).  Any
    zarr-vectors advantage measured here is therefore a lower bound.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, DOMAIN, (n, 3)).astype(np.float32)
    return {
        "positions": pos,
        "object_ids": np.arange(n, dtype=np.int64),
        "n_objects": n,
        "n_vertices": n,
    }


def _random_walk(rng, start, n_steps, step_sd):
    walk = rng.normal(0.0, step_sd, (n_steps, 3)).cumsum(axis=0)
    return np.clip(start + walk, 0.0, DOMAIN - 1.0).astype(np.float32)


def gen_streamlines(n: int, seed: int = SEED) -> dict:
    """``n / 12`` smooth random-walk polylines totalling about ``n`` vertices."""
    rng = np.random.default_rng(seed)
    n_obj = max(1, n // VERTS_PER_STREAMLINE)
    counts = rng.integers(VERTS_PER_STREAMLINE - 4,
                          VERTS_PER_STREAMLINE + 4, size=n_obj)
    starts = rng.uniform(0, DOMAIN - 100.0, (n_obj, 3))
    polys = [_random_walk(rng, starts[i], int(counts[i]), 8.0)
             for i in range(n_obj)]
    return {
        "polylines": polys,
        "n_objects": n_obj,
        "n_vertices": int(sum(len(p) for p in polys)),
    }


def gen_skeleton(n: int, seed: int = SEED) -> dict:
    """``n / 200`` long chains with a per-vertex radius -- the SWC shape.

    Each object is a single unbranched path, so the parent of vertex *i*
    is vertex *i - 1*.  That is exactly the implicit sequential link the
    zarr-vectors layout stores for free and that SWC must spell out as an
    explicit parent column on every row.
    """
    rng = np.random.default_rng(seed)
    n_obj = max(1, n // VERTS_PER_SKELETON)
    counts = rng.integers(VERTS_PER_SKELETON - 40,
                          VERTS_PER_SKELETON + 40, size=n_obj)
    starts = rng.uniform(0, DOMAIN - 200.0, (n_obj, 3))
    polys = [_random_walk(rng, starts[i], int(counts[i]), 3.0)
             for i in range(n_obj)]
    radii = [rng.uniform(0.5, 4.0, len(p)).astype(np.float32) for p in polys]
    return {
        "polylines": polys,
        "radii": radii,
        "n_objects": n_obj,
        "n_vertices": int(sum(len(p) for p in polys)),
    }


def gen_graph(n: int, seed: int = SEED) -> dict:
    """``n`` positioned nodes with about ``1.5 n`` undirected edges."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, DOMAIN, (n, 3)).astype(np.float32)
    m = (3 * n) // 2
    src = rng.integers(0, n, size=m)
    dst = rng.integers(0, n, size=m)
    keep = src != dst
    edges = np.stack([src[keep], dst[keep]], axis=1).astype(np.int64)
    return {
        "positions": pos,
        "edges": edges,
        "n_objects": n,
        "n_vertices": n,
    }


def gen_mesh(n: int, seed: int = SEED) -> dict:
    """``n / 144`` disjoint surface patches -- many objects, not one blob.

    Real mesh collections (segmented cells, nuclei, ROIs) are thousands of
    small closed surfaces, and that is what makes "fetch one object"
    meaningful.  Neither STL nor OBJ carries an object index, so both must
    scan the whole file to find one patch.
    """
    rng = np.random.default_rng(seed)
    side = MESH_PATCH_SIDE
    per_patch = side * side
    n_obj = max(1, n // per_patch)

    u, v = np.meshgrid(np.linspace(0.0, 40.0, side),
                       np.linspace(0.0, 40.0, side), indexing="ij")
    i = np.arange(side - 1)
    ii, jj = np.meshgrid(i, i, indexing="ij")
    a = (ii * side + jj).ravel()
    b, c, d = a + 1, a + side, a + side + 1
    base_faces = np.concatenate([np.stack([a, b, c], 1),
                                 np.stack([b, d, c], 1)]).astype(np.int64)

    origins = rng.uniform(0, DOMAIN - 60.0, (n_obj, 3))
    verts = np.empty((n_obj * per_patch, 3), dtype=np.float32)
    faces = np.empty((n_obj * len(base_faces), 3), dtype=np.int64)
    object_ids = np.empty(n_obj * per_patch, dtype=np.int64)
    for k in range(n_obj):
        w = rng.uniform(0.0, 6.0, (side, side))
        patch = np.stack([u, v, w], axis=-1).reshape(-1, 3) + origins[k]
        verts[k * per_patch:(k + 1) * per_patch] = patch.astype(np.float32)
        faces[k * len(base_faces):(k + 1) * len(base_faces)] = (
            base_faces + k * per_patch
        )
        object_ids[k * per_patch:(k + 1) * per_patch] = k
    return {
        "vertices": np.clip(verts, 0.0, DOMAIN - 1.0),
        "faces": faces,
        "object_ids": object_ids,
        "n_objects": n_obj,
        "n_vertices": len(verts),
        "verts_per_object": per_patch,
    }


GENERATORS = {
    "points": gen_points,
    "streamlines": gen_streamlines,
    "skeletons": gen_skeleton,
    "graphs": gen_graph,
    "meshes": gen_mesh,
}


# --------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------

def environment() -> dict:
    """Everything needed to say where a number came from."""
    import numpy

    def _ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    def _tree_sha(pkg_dir):
        """Content digest of every ``.py`` in the installed package.

        An editable install measured from a working tree can change
        between one block of the sweep and the next -- the commit is the
        same, ``-dirty`` is the same, and the numbers are not.  A digest
        over the sources actually imported is the only thing that
        distinguishes those two states, and it is what makes the
        ``previous_run`` check in ``run_sweep.py`` notice.
        """
        import hashlib
        try:
            h = hashlib.sha256()
            for f in sorted(Path(pkg_dir).rglob("*.py")):
                h.update(str(f.relative_to(pkg_dir)).encode())
                h.update(hashlib.sha256(f.read_bytes()).digest())
            return h.hexdigest()[:12]
        except Exception:
            return None

    def _git(repo):
        """Short HEAD, suffixed ``-dirty`` when the tree has local edits.

        A bare hash would claim a measurement came from a commit anyone
        can check out, which is false whenever the package was measured
        with uncommitted work in it -- and that is exactly the state a
        benchmark gets re-run in while something is being optimised.
        """
        try:
            head = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except Exception:
            return None
        try:
            dirty = subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except Exception:
            return head
        return f"{head}-dirty" if dirty else head

    try:
        import zarr_vectors
        zv_pkg = Path(zarr_vectors.__file__).resolve().parent
        zv_path = zv_pkg.parent
        zv_ver, zv_git = zarr_vectors.__version__, _git(zv_path)
        zv_sha = _tree_sha(zv_pkg)
    except Exception:
        zv_ver = zv_git = zv_sha = None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
        "numpy": numpy.__version__,
        "zarr": _ver("zarr"),
        "numcodecs": _ver("numcodecs"),
        "nibabel": _ver("nibabel"),
        "networkx": _ver("networkx"),
        "zarr_vectors": zv_ver,
        "zarr_vectors_git": zv_git,
        "zarr_vectors_tree_sha": zv_sha,
        "zarr_vectors_tools_git": _git(Path(__file__).resolve().parents[2]),
        "domain": DOMAIN,
        "chunk_shape": CHUNK,
        "bin_shape": BIN,
        "seed": SEED,
    }
