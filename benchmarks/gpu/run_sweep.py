#!/usr/bin/env python
"""Measure ``read_cells`` on the host and on a CUDA device, into a tidy CSV.

    python run_sweep.py                 # 10^3..10^7, a few minutes
    python run_sweep.py --quick         # smoke test, seconds
    python run_sweep.py --sizes 1e5,1e6 --workdir /scratch

Measurement is separated from plotting, as in ``../paper/``: this script
writes ``results/measurements.csv`` and ``results/environment.json`` and
draws nothing; ``make_figure.py`` turns the CSV into panels.

What is measured
----------------
One operation: ``read_cells`` over **every** cell of a point-cloud store,
for ``vertices`` and one float32 vertex attribute -- the whole level,
decoded into CSR form.  It is timed three ways on the same store:

``cpu``                  ``read_cells(...)``: read and decode on the host.
``gpu (host decode)``    ``device="cuda", decode="host"``: decode on the
                         host, then one upload per array.
``gpu (device decode)``  ``device="cuda", decode="device"``: the stored
                         bytes go to the device and are decoded there --
                         by the fragment kernels, and through nvCOMP for
                         zstd.

and each on two stores per size, uncompressed and ``compressor='zstd'``,
so the ``codec`` column holds ``none`` or ``zstd``.  A device read is
timed up to a ``cudaDeviceSynchronize``, so an asynchronous kernel still
running when the call returns is counted.

Every GPU result is compared element-wise with the host result before
its row is written; a mismatch stops the sweep rather than plotting a
fast wrong answer.

Statistics
----------
One untimed warm-up call per (store, mode) -- it pays for CUDA context
set-up, kernel compilation and nvCOMP handle creation, none of which a
reading process pays twice -- then exactly ``RUNS`` timed calls.  Rows
hold their mean and a Student's-t 95 % half-width (``_harness.mean_ci95``).
The count is fixed rather than adaptive, so every point on a curve is
the same number of samples.

The store sits in the page cache after the warm-up, so this times
decoding and assembly, not the disk.

The domain, chunk grid and point generator are the paper sweep's
(``../paper/_harness.py``): uniform points in a 1000-unit cube, 200-unit
chunks, 5^3 = 125 cells.  Sizes step by half-decades rather than decades
because the host and device curves separate between 10^6 and 10^7.

Environment
-----------
Needs cupy, and nvCOMP's Python bindings for the zstd device decode;
``zarr_vectors.runtime_capabilities()`` must report ``device_decode`` and
``gpu_codecs``.  ``environment.json`` records the device, the driver and
runtime versions, cupy and nvCOMP, alongside everything the paper
sweep records.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "paper"))
import _harness as H  # noqa: E402

RESULTS = HERE / "results"

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000,
         1_000_000, 3_000_000, 10_000_000]
QUICK_SIZES = [1_000, 10_000]
RUNS = 5
CODECS = ["none", "zstd"]
ARRAYS = ["vertices", "vertex_attributes/intensity"]

# (impl, read_cells kwargs) -- the order is the order rows are written.
MODES = [
    ("cpu", {}),
    ("gpu (host decode)", {"device": "cuda", "decode": "host"}),
    ("gpu (device decode)", {"device": "cuda", "decode": "device"}),
]

FIELDS = ["block", "geometry", "codec", "n_target", "n_vertices", "n_cells",
          "op", "impl", "unit", "value", "ci_hw", "n_runs", "detail"]


def log(msg):
    print(msg, flush=True)


def _size(text: str) -> int:
    return int(float(text.strip().replace("_", "")))


def _sync():
    import cupy
    cupy.cuda.Device().synchronize()


def _time_read(fn) -> tuple[float, float, int, object]:
    """Warm up once, then time ``RUNS`` calls to device-synchronised end."""
    out = fn()
    _sync()
    samples = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        out = fn()
        _sync()
        samples.append(time.perf_counter() - t0)
    m, hw = H.mean_ci95(samples)
    return m, hw, len(samples), out


def _check(impl, got, ref):
    import cupy
    for name in ARRAYS:
        for part in ("data", "offsets"):
            a = cupy.asnumpy(getattr(got[name], part))
            b = getattr(ref[name], part)
            if not np.array_equal(a, b):
                raise AssertionError(
                    f"{impl}: {name}.{part} differs from the host read")


def _write_store(path, data, codec):
    from zarr_vectors.types.points import write_points
    rng = np.random.default_rng(H.SEED + 1)
    write_points(
        str(path), data["positions"],
        chunk_shape=H.CHUNK, bounds=H.BOUNDS,
        vertex_attributes={
            "intensity": rng.uniform(0, 1, data["n_vertices"]).astype("float32"),
        },
        compressor=None if codec == "none" else codec,
    )


def sweep(sizes, ws) -> list[dict]:
    from zarr_vectors.core import arrays
    from zarr_vectors.core.cells import read_cells
    from zarr_vectors.core.store import get_resolution_level, open_store

    rows = []
    for codec in CODECS:
        for n in sizes:
            data = H.gen_points(n)
            path = ws.store(f"{codec}_{n}")
            _write_store(path, data, codec)
            lg = get_resolution_level(open_store(str(path)), 0)
            cells = np.asarray(arrays.list_chunk_keys(lg), dtype=np.int64)
            ref = None
            for impl, kw in MODES:
                m, hw, k, out = _time_read(
                    lambda: read_cells(lg, cells, ARRAYS, **kw))
                if ref is None:
                    ref = out
                else:
                    _check(impl, out, ref)
                log(f"  {codec:4s} N={n:>11,}  {impl:20s} "
                    f"{m * 1e3:9.2f} ms  ± {hw * 1e3:.2f}")
                rows.append({
                    "block": "read_cells", "geometry": "points",
                    "codec": codec, "n_target": n,
                    "n_vertices": data["n_vertices"], "n_cells": len(cells),
                    "op": "read_all_cells", "impl": impl, "unit": "s",
                    "value": m, "ci_hw": hw, "n_runs": k,
                    "detail": ",".join(ARRAYS),
                })
            del ref, out, lg
            ws.clear(f"{codec}_{n}")
    return rows


def environment(sizes) -> dict:
    import cupy
    import zarr_vectors

    env = H.environment()
    rt = cupy.cuda.runtime
    props = rt.getDeviceProperties(0)
    try:
        import nvidia.nvcomp as nvcomp
        nvcomp_ver = getattr(nvcomp, "__version__", "present")
    except ImportError:
        nvcomp_ver = None
    env.update({
        "gpu": props["name"].decode(),
        "gpu_memory_bytes": int(props["totalGlobalMem"]),
        "cuda_driver": rt.driverGetVersion(),
        "cuda_runtime": rt.runtimeGetVersion(),
        "cupy": cupy.__version__,
        "nvcomp": nvcomp_ver,
        "capabilities": zarr_vectors.runtime_capabilities(),
        "sizes": sizes,
        "codecs": CODECS,
        "arrays": ARRAYS,
        "runs": RUNS,
        "warmup_runs": 1,
    })
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="two small sizes only, for a smoke test")
    ap.add_argument("--sizes", default=None,
                    help="comma-separated vertex counts, e.g. '1e5,1e6'")
    ap.add_argument("--workdir", default=None,
                    help="parent directory for scratch stores")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    if args.sizes:
        sizes = sorted({_size(t) for t in args.sizes.split(",") if t.strip()})
    else:
        sizes = QUICK_SIZES if args.quick else SIZES

    import zarr_vectors
    caps = zarr_vectors.runtime_capabilities()
    missing = [c for c in ("device_decode", "gpu_codecs") if not caps.get(c)]
    if missing:
        sys.exit(f"this install cannot run the sweep: {missing} unavailable")

    ws = H.Workspace(args.workdir)
    log(f"scratch: {ws.root}")
    try:
        rows = sweep(sizes, ws)
    finally:
        ws.close()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "measurements.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    env = environment(sizes)
    env["workdir_fstype"] = H.fstype(ws.root.parent)
    (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    log(f"wrote {out / 'measurements.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
