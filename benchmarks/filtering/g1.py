"""Gate G1: is numpy fast enough to be the in-browser filter engine?

The build guide's gate G1 asks whether numpy under Pyodide meets the filter
plan's interactive targets (C8) on 10^5 and 10^6 streamline tractograms; if
it does, the Rust kernel is needed only for the spatial work.  This module
is the measurement.  It imports nothing but numpy and the filtering
subpackage, which the caller passes in, so the very same code runs under
CPython and inside Pyodide (see run.py and pyodide_worker.mjs).

Each set is filtered by the indexed engine with level 0 already in memory,
so the times are compute only: no fetching, no decoding.  The brute-force
oracle answers one of the same questions as a cross-check, and as the
baseline the index has to beat.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

#: The filter plan's C8 targets, in the units each measurement reports.
GATES = {
    "p3_ms": 16.0,          # operator flip, every leaf cached
    "p1_ms": 50.0,          # one column comparison
    "sphere_ms": 100.0,     # sphere r = 5 mm, geometry in memory
    "publish_ms": 20.0,     # answers -> one byte per streamline
    "peak_mb": 1536.0,      # peak memory while filtering
    "first_result_s": 3.0,  # index build + first sphere
}

#: Synthetic sets: name -> streamline count.  Whole-brain-like, 1 mm steps.
SYNTHETIC = {"synth-1e5": 100_000, "synth-1e6": 1_000_000}


def prepare_set(filtering, name: str, data_dir: str | Path, store: str | None = None) -> None:
    """Write set ``name`` to ``data_dir/name/`` as .npy files.

    A synthetic set is generated; ``real`` is read from ``store`` (positions
    kept as float32 RAS).  Both runtimes then load the same arrays, and
    neither pays for generating them: a wasm heap never shrinks, so
    generating inside Pyodide would count the generator's temporaries
    against the engine.
    """
    if name in SYNTHETIC:
        tg = filtering.whole_brain_like(SYNTHETIC[name], seed=0)
        columns = {k: tg.columns[k] for k in ("fa_mean", "label")}
    elif name == "real":
        tg = filtering.load_tractogram(store, columns=["label_id"])
        columns = {k: v for k, v in tg.columns.items() if k == "label_id"}
    else:
        raise ValueError(f"unknown set {name!r}")
    out = Path(data_dir) / name
    (out / "columns").mkdir(parents=True, exist_ok=True)
    np.save(out / "vertices.npy", tg.vertices.astype(np.float32))
    np.save(out / "offsets.npy", tg.offsets)
    for key, values in columns.items():
        np.save(out / "columns" / f"{key}.npy", values)


def make_set(filtering, name: str, data_dir: str | Path):
    """Load the set :func:`prepare_set` wrote (``length`` is computed on load)."""
    where = Path(data_dir) / name
    columns = {p.stem: np.load(p) for p in sorted((where / "columns").glob("*.npy"))}
    return filtering.Tractogram(
        np.load(where / "vertices.npy"), np.load(where / "offsets.npy"), columns=columns,
    )


def _timed(fn):
    start = time.perf_counter()
    out = fn()
    return (time.perf_counter() - start) * 1e3, out


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values)
    return {"median": float(np.median(arr)), "p90": float(np.percentile(arr, 90)),
            "max": float(arr.max())}


def measure(filtering, tractogram, *, trials: int = 9, seed: int = 0, memory_mb=None) -> dict:
    """Time every kind of question on one tractogram.

    Each region query uses a fresh region (centred on a random vertex), so
    its leaf is never cached; P3 re-evaluates trees whose leaves all are.
    ``memory_mb``, if given, reports the process's memory now; it is
    sampled once the set is loaded and again once the index is built.
    """
    F = filtering
    tg = tractogram
    engine = None
    out: dict = {
        "n_streamlines": int(tg.n_objects),
        "n_vertices": int(len(tg.vertices)),
        "tractogram_mb": round((tg.vertices.nbytes + tg.offsets.nbytes) / 1e6, 1),
    }
    if memory_mb is not None:
        out["loaded_mb"] = round(memory_mb())
    build_ms, engine = _timed(lambda: F.IndexedEngine(tg))
    out["build_s"] = build_ms / 1e3
    out["index_mb"] = round(engine.index_bytes / 1e6, 1)
    if memory_mb is not None:
        out["built_mb"] = round(memory_mb())
    out["cell_size_mm"] = round(engine.cell_size, 3)

    rng = np.random.default_rng(seed)
    centres = tg.vertices[rng.integers(0, len(tg.vertices), trials)].astype(np.float64)

    def regions(make, label: str) -> None:
        times, counts, work = [], [], []
        for centre in centres:
            ms, result = _timed(lambda c=centre: engine.evaluate(make(c)))
            times.append(ms)
            counts.append(int(result.sum()))
            work.append(dict(engine.last_query))
        out[f"{label}_ms"] = _stats(times)
        out[f"{label}_accepted"] = int(np.median(counts))
        if work[0]:
            # Segments filed near the region, and how many needed the exact test.
            out[f"{label}_candidates"] = int(np.median([w["candidates"] for w in work]))
            out[f"{label}_tested"] = int(np.median([w["tested"] for w in work]))
            # Segments tested per second: the rate a faster kernel would improve.
            tested = np.array([w["tested"] for w in work], dtype=float)
            out[f"{label}_tested_per_s"] = float(np.median(tested / (np.asarray(times) / 1e3)))

    regions(lambda c: F.Passes(F.Sphere(c, 5.0)), "sphere")
    out["first_result_s"] = out["build_s"] + out["sphere_ms"]["max"] / 1e3
    regions(lambda c: F.Passes(F.Box(c - 5.0, c + 5.0)), "box")
    regions(lambda c: F.Passes(F.Slab(1, c[1] - 1.0, c[1] + 1.0)), "slab")
    normals = rng.normal(size=(trials, 3))
    normal_of = {tuple(c): n for c, n in zip(centres, normals)}
    regions(lambda c: F.Passes(F.Disk(c, normal_of[tuple(c)], 5.0)), "disk")
    blob = rng.random((10, 10, 10)) < 0.5

    def mask_at(c):
        affine = np.eye(4)
        affine[:3, 3] = c - 4.5           # 10 x 1 mm voxels centred on c
        return F.Passes(F.Mask(blob, affine))

    regions(mask_at, "mask")
    regions(lambda c: F.Passes(F.Sphere(c + 0.25, 5.0), "endpoints"), "endpoints")

    lengths = tg.columns["length"]
    lo_hi = np.percentile(lengths, [10, 90])
    p1 = []
    for k in range(trials):
        leaf = F.Column("length", "between", (float(lo_hi[0]) + 0.01 * k, float(lo_hi[1])))
        p1.append(_timed(lambda leaf=leaf: engine.evaluate(leaf))[0])
    out["p1_ms"] = _stats(p1)

    # P3: four cached leaves, the operators flipped between evaluations.
    a = F.Passes(F.Sphere(centres[0], 5.0))
    b = F.Passes(F.Box(centres[1] - 5.0, centres[1] + 5.0))
    c = F.Column("length", ">", float(np.median(lengths)))
    d = F.Passes(F.Slab(1, centres[2][1] - 1.0, centres[2][1] + 1.0))
    trees = [F.And(a, b, F.Not(c), d), F.Or(a, b, c, d), F.And(F.Or(a, b), F.Not(d), c),
             F.Or(F.And(a, c), F.Not(b), d)]
    for tree in trees:
        engine.evaluate(tree)  # fill the cache
    p3 = [_timed(lambda t=trees[k % len(trees)]: engine.evaluate(t))[0] for k in range(4 * trials)]
    out["p3_ms"] = _stats(p3)

    groups = [engine.evaluate(t) for t in trees[:3]]
    out["publish_ms"] = _stats([_timed(lambda: engine.publish(groups))[0] for _ in range(trials)])

    # The oracle answers the first sphere again, testing every segment.
    leaf = F.Passes(F.Sphere(centres[0], 5.0))
    oracle_ms, truth = _timed(lambda: F.evaluate(leaf, tg))
    out["oracle_sphere_s"] = oracle_ms / 1e3
    out["agrees_with_oracle"] = bool(np.array_equal(truth, engine.evaluate(leaf)))
    return out


def verdict(result: dict) -> dict[str, bool]:
    """Which C8 gates one set's result meets."""
    seen = {
        "p3_ms": result["p3_ms"]["median"],
        "p1_ms": result["p1_ms"]["median"],
        "sphere_ms": result["sphere_ms"]["median"],
        "publish_ms": result["publish_ms"]["median"],
        "peak_mb": result.get("peak_mb", float("nan")),
        "first_result_s": result["first_result_s"],
    }
    return {gate: bool(seen[gate] <= limit) for gate, limit in GATES.items()}
