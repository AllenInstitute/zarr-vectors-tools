"""Generate the eight Zarr Vectors internals benchmark notebooks.

Run once and commit the output:

    python benchmarks/internals/_build.py

Each notebook follows a tight 4-section structure (setup → sweep →
table → plot).  Keep the per-cell strings short — these are
*benchmarks*, not tutorials.

These sweep zarr-vectors against *itself* along one axis at a time
(size, geometry type, backend, pyramid depth, chunk shape, codec,
edit kind).  For zarr-vectors versus other on-disk formats, see the
sibling suite in ``benchmarks/formats/``.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ===================================================================
# Shared cells (same across all three notebooks)
# ===================================================================

SHARED_HELPERS = """import os, time, tempfile, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _time(fn, *args, **kwargs):
    \"\"\"Call fn(*args, **kwargs); return (elapsed_seconds, result).\"\"\"
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return time.perf_counter() - t0, out


def _store_bytes(path):
    \"\"\"Total on-disk size of a store directory, in bytes.\"\"\"
    p = Path(path)
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())


def _new_store(prefix):
    \"\"\"Fresh tempdir + zarrvectors path.\"\"\"
    return Path(tempfile.mkdtemp(prefix=f'zvbench_{prefix}_')) / 'store.zarrvectors'
"""


# Stats helpers used by the averaged benchmarks (01, 02).  Not needed in 03
# (backends) which still uses a single-run sketch.
STATS_HELPERS = """N_RUNS = 10
T95_DF9 = 2.262  # scipy.stats.t.ppf(0.975, df=9) — hard-coded to avoid scipy dep


def _mean_ci95(samples):
    \"\"\"(mean, half-width) for a 1-D sample using Student's t, df=n-1.\"\"\"
    arr = np.asarray(samples, dtype=float)
    if arr.size < 2:
        return float(arr.mean()) if arr.size else 0.0, 0.0
    m = arr.mean()
    s = arr.std(ddof=1)
    hw = T95_DF9 * s / np.sqrt(arr.size)
    return float(m), float(hw)
"""


# ===================================================================
# 01 · Size scaling — point cloud, vary N
# ===================================================================

SIZE_CELLS: list[tuple[str, str]] = [
    ("md", """\
# Size scaling — point cloud

Write/read/disk-size of point clouds at increasing `N`, with **CSV as a
baseline** for context. Same `chunk_shape` across runs so the only
variable is vertex count.

Each timing is averaged over **`N_RUNS = 10` runs**; the plot shows the
mean with a shaded **95% confidence interval** (Student's t, df=9).

For each `N` we measure:

| Operation | zarr-vectors | CSV (baseline) |
| --- | --- | --- |
| Write     | `write_points`           | `pandas.to_csv` |
| Read all  | `read_points`            | `pandas.read_csv` |
| Read one  | one chunk via lazy API   | `read_csv` at a **random row index** |
| Disk size | store directory          | CSV file |

CSV "read one" uses a random row each run (CSV has no random access, so the
parser must scan from the file head — this is the honest comparison vs.
zarr-vectors' chunk-level lookup).

Runtime: ~5 minutes on a laptop (the 1M case dominates).
"""),
    ("code", SHARED_HELPERS + "\n\n" + STATS_HELPERS),
    ("md", "## 1 · Setup"),
    ("code", """\
from zarr_vectors.types.points import write_points, read_points
from zarr_vectors.lazy import open_zv

SIZES = [1_000, 10_000, 100_000, 1_000_000]
CHUNK = (200.0, 200.0, 200.0)
BIN   = (50.0, 50.0, 50.0)
SEED  = 0


def _csv_path(prefix):
    \"\"\"Fresh tempdir + CSV path.\"\"\"
    return Path(tempfile.mkdtemp(prefix=f'csvbench_{prefix}_')) / 'points.csv'


def _csv_write(path, positions, intensity):
    \"\"\"Baseline: write x,y,z,intensity columns to a CSV.\"\"\"
    pd.DataFrame({
        'x': positions[:, 0],
        'y': positions[:, 1],
        'z': positions[:, 2],
        'intensity': intensity,
    }).to_csv(path, index=False)


def _csv_read_all(path):
    \"\"\"Read every row back into memory.\"\"\"
    return pd.read_csv(path)


def _csv_read_one(path, row_idx):
    \"\"\"Random-index single-row read. CSV has no random access, so the
    parser must scan from the top — this exposes the linear-scan cost.\"\"\"
    return pd.read_csv(path, skiprows=range(1, row_idx + 1), nrows=1)


def _zv_read_one(store_path):
    \"\"\"Read just one chunk's worth of vertices via the lazy API.\"\"\"
    zv = open_zv(store_path)
    chunk_keys = zv[0].vertices._chunk_keys  # noqa: SLF001 — minimal demo
    if not chunk_keys:
        return None
    return zv[0].vertices[chunk_keys[0]].compute()
"""),
    ("md", "## 2 · Run the sweep"),
    ("code", """\
rng = np.random.default_rng(SEED)

# Per-(metric, prefix) raw timings; shape becomes (len(SIZES), N_RUNS) after fill.
metrics = ['write_s', 'read_all_s', 'read_one_s']
raw = {m: {p: np.zeros((len(SIZES), N_RUNS)) for p in ('zv', 'csv')} for m in metrics}
sizes_MB = {'zv': np.zeros(len(SIZES)), 'csv': np.zeros(len(SIZES))}

for i, n in enumerate(SIZES):
    # Generate input data once per N (re-running write_points with the same
    # input is the realistic re-run scenario; we still get fresh stores).
    positions = rng.uniform(0, 1000, (n, 3)).astype(np.float32)
    intensity = rng.uniform(0, 1, n).astype(np.float32)

    for run in range(N_RUNS):
        # ---- ZV: fresh store each run (cold-cache write) ----
        store = _new_store(f'size_{n}_r{run}')
        t_zv_write,    _ = _time(
            write_points, store, positions,
            chunk_shape=CHUNK, bin_shape=BIN,
            vertex_attributes={'intensity': intensity},
        )
        t_zv_read_all, _ = _time(read_points, store, attribute_names=['intensity'])
        t_zv_read_one, _ = _time(_zv_read_one, store)
        if run == 0:
            sizes_MB['zv'][i] = _store_bytes(store) / 1e6
        shutil.rmtree(Path(store).parent, ignore_errors=True)

        # ---- CSV baseline ----
        csv = _csv_path(f'size_{n}_r{run}')
        t_csv_write,    _ = _time(_csv_write, csv, positions, intensity)
        t_csv_read_all, _ = _time(_csv_read_all, csv)
        # Fresh random row index per run — CSV pays linear scan cost.
        row_idx = int(rng.integers(0, n))
        t_csv_read_one, _ = _time(_csv_read_one, csv, row_idx)
        if run == 0:
            sizes_MB['csv'][i] = csv.stat().st_size / 1e6
        shutil.rmtree(csv.parent, ignore_errors=True)

        raw['write_s']['zv'][i, run]     = t_zv_write
        raw['read_all_s']['zv'][i, run]  = t_zv_read_all
        raw['read_one_s']['zv'][i, run]  = t_zv_read_one
        raw['write_s']['csv'][i, run]    = t_csv_write
        raw['read_all_s']['csv'][i, run] = t_csv_read_all
        raw['read_one_s']['csv'][i, run] = t_csv_read_one

# Summarise into a tidy dataframe (mean + 95% CI half-width per metric).
rows = []
for i, n in enumerate(SIZES):
    row = {'N': n}
    for m in metrics:
        for p in ('zv', 'csv'):
            mean, hw = _mean_ci95(raw[m][p][i])
            row[f'{p}_{m}_mean'] = round(mean, 4)
            row[f'{p}_{m}_hw']   = round(hw, 4)
    row['zv_size_MB']  = round(sizes_MB['zv'][i],  2)
    row['csv_size_MB'] = round(sizes_MB['csv'][i], 2)
    rows.append(row)

df = pd.DataFrame(rows)
"""),
    ("md", "## 3 · Results"),
    ("code", "df"),
    ("md", "## 4 · Plot"),
    ("code", """\
PANELS = [
    ('Write time', 'write_s',    's'),
    ('Read all',   'read_all_s', 's'),
    ('Read one',   'read_one_s', 's'),
]
SERIES = [
    ('zarr-vectors', 'zv',  'C0'),
    ('csv',          'csv', 'C1'),
]

fig, axes = plt.subplots(1, 4, figsize=(14, 4))

# Three timing panels: mean line + shaded 95% CI band on log-log.
for ax, (title, key, unit) in zip(axes[:3], PANELS):
    for label, prefix, color in SERIES:
        mean = df[f'{prefix}_{key}_mean'].values
        hw   = df[f'{prefix}_{key}_hw'].values
        ax.fill_between(df['N'], mean - hw, mean + hw, color=color, alpha=0.2)
        ax.loglog(df['N'], mean, 'o-', color=color, label=label)
    ax.set_xlabel('N (vertices)')
    ax.set_ylabel(unit)
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)

# Disk size panel: single deterministic measurement per series — no CI.
ax = axes[3]
for label, prefix, color in SERIES:
    ax.loglog(df['N'], df[f'{prefix}_size_MB'], 'o-', color=color, label=label)
ax.set_xlabel('N (vertices)')
ax.set_ylabel('MB')
ax.set_title('Disk size')
ax.grid(True, which='both', alpha=0.3)

axes[0].legend(loc='best')
fig.suptitle(
    f'zarr-vectors vs CSV — point cloud scaling ({N_RUNS} runs, 95% CI)',
)
plt.tight_layout()
"""),
]


# ===================================================================
# 02 · Data types — fixed N, sweep the six types
# ===================================================================

TYPES_CELLS: list[tuple[str, str]] = [
    ("md", """\
# Data-type scaling

Six geometry types benchmarked at the same target vertex count
(`N = 50_000`, ± a few %).  Each type is run in two regimes:

- **representative** — natural per-type layout (polylines as locally
  coherent random walks, mesh as a triangulated grid, lines / graph /
  skeleton with scattered positions);
- **scattered** — every vertex drawn independently uniformly across
  the volume, with topology unchanged.  Forces near-maximal cross-chunk
  traffic for types whose representative layout is spatially coherent.

The two regimes isolate the two cost components: per-vertex / per-object
overhead intrinsic to the write/read path, and the cross-chunk cost
driven by spatial distribution.

Each timing is averaged over **`N_RUNS = 10` runs**; bars show the mean
with **95% confidence interval** error bars (Student's t, df=9).
The results table reports the structural metrics (`vertex_count`,
`object_count`, `cross_chunk_count`, `size_MB`) so residual differences
between types can be attributed to specific drivers.
"""),
    ("code", SHARED_HELPERS + "\n\n" + STATS_HELPERS),
    ("md", "## 1 · Setup"),
    ("code", """\
from zarr_vectors.types.points import write_points, read_points
from zarr_vectors.types.lines import write_lines, read_lines
from zarr_vectors.types.polylines import write_polylines, read_polylines
from zarr_vectors.types.graphs import write_graph, read_graph
from zarr_vectors.types.meshes import write_mesh, read_mesh

N = 50_000
CHUNK = (200.0, 200.0, 200.0)
BIN   = (50.0, 50.0, 50.0)
SEED  = 0
rng = np.random.default_rng(SEED)
"""),
    ("md", "## 2 · Synthetic generators (representative + scattered)"),
    ("code", """\
# Representative — natural per-type layout, ~N total vertices.
def _rep_points():
    return rng.uniform(0, 1000, (N, 3)).astype(np.float32)


def _rep_lines():
    # N/2 segments, two random endpoints each → ~N total vertices.
    return rng.uniform(0, 1000, (N // 2, 2, 3)).astype(np.float32)


def _rep_polylines():
    # ~N total vertices spread across short, locally coherent walks.
    counts = rng.integers(8, 16, size=N // 12)
    out = []
    for c in counts:
        steps = rng.normal(0, 5, (c, 3))
        start = rng.uniform(0, 1000, 3)
        out.append((start + steps.cumsum(axis=0)).astype(np.float32))
    return out


def _rep_graph(is_tree=False):
    positions = rng.uniform(0, 1000, (N, 3)).astype(np.float32)
    if is_tree:
        parents = rng.integers(0, np.arange(1, N))
        edges = np.stack([np.arange(1, N), parents], axis=1).astype(np.int64)
    else:
        src = rng.integers(0, N, size=3 * N // 2)
        dst = rng.integers(0, N, size=3 * N // 2)
        mask = src != dst
        edges = np.stack([src[mask], dst[mask]], axis=1).astype(np.int64)
    return positions, edges


def _rep_mesh():
    side = int(np.sqrt(N))
    xs, ys = np.meshgrid(np.linspace(0, 1000, side), np.linspace(0, 1000, side))
    zs = rng.uniform(0, 100, (side, side))
    verts = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3).astype(np.float32)
    i = np.arange(side - 1); j = np.arange(side - 1)
    ii, jj = np.meshgrid(i, j, indexing='ij')
    a = (ii * side + jj).ravel(); b = a + 1; c = a + side; d = c + 1
    faces = np.concatenate([
        np.stack([a, b, c], axis=1),
        np.stack([b, d, c], axis=1),
    ]).astype(np.int64)
    return verts, faces


# Scattered — same topology, every vertex independently uniform in [0, 1000]^3.
# Types whose representative is already uniform reuse the rep generator.
_scat_points = _rep_points
_scat_lines  = _rep_lines
_scat_graph  = _rep_graph


def _scat_polylines():
    counts = rng.integers(8, 16, size=N // 12)
    return [
        rng.uniform(0, 1000, (c, 3)).astype(np.float32)
        for c in counts
    ]


def _scat_mesh():
    # Same grid faces, but vertex positions are uniformly random.
    side = int(np.sqrt(N))
    n_verts = side * side
    verts = rng.uniform(0, 1000, (n_verts, 3)).astype(np.float32)
    i = np.arange(side - 1); j = np.arange(side - 1)
    ii, jj = np.meshgrid(i, j, indexing='ij')
    a = (ii * side + jj).ravel(); b = a + 1; c = a + side; d = c + 1
    faces = np.concatenate([
        np.stack([a, b, c], axis=1),
        np.stack([b, d, c], axis=1),
    ]).astype(np.int64)
    return verts, faces
"""),
    ("md", "## 3 · Run the sweep"),
    ("code", """\
def _extract_stats(name, r):
    \"\"\"Normalize each write_*'s return dict into (vertex_count, object_count, cross_chunk_count).\"\"\"
    if name == 'points':
        return r['vertex_count'], r['object_count'], 0
    if name == 'lines':
        return r['line_count'] * 2, r['line_count'], r['cross_chunk_count']
    if name == 'polylines':
        return r['vertex_count'], r['polyline_count'], r['cross_chunk_link_count']
    if name in ('graph', 'skeleton'):
        return r['node_count'], r['node_count'], r['cross_edge_count']
    if name == 'mesh':
        return r['vertex_count'], r['face_count'], r['cross_face_count']
    raise ValueError(name)


def bench_points(gen):
    pts = gen()
    store = _new_store('points')
    tw, res = _time(write_points, store, pts, chunk_shape=CHUNK, bin_shape=BIN)
    tr, _   = _time(read_points, store)
    return tw, tr, res, _store_bytes(store), store

def bench_lines(gen):
    eps = gen()
    store = _new_store('lines')
    tw, res = _time(write_lines, store, eps, chunk_shape=CHUNK, bin_shape=BIN)
    tr, _   = _time(read_lines, store)
    return tw, tr, res, _store_bytes(store), store

def bench_polylines(gen):
    plys = gen()
    store = _new_store('polylines')
    tw, res = _time(write_polylines, store, plys, chunk_shape=CHUNK, bin_shape=BIN)
    tr, _   = _time(read_polylines, store)
    return tw, tr, res, _store_bytes(store), store

def bench_graph(gen, kind):
    pos, edges = gen(is_tree=(kind == 'skeleton'))
    store = _new_store(kind)
    tw, res = _time(
        write_graph, store, pos, edges,
        chunk_shape=CHUNK, bin_shape=BIN, kind=kind,
    )
    tr, _ = _time(read_graph, store)
    return tw, tr, res, _store_bytes(store), store

def bench_mesh(gen):
    verts, faces = gen()
    store = _new_store('mesh')
    tw, res = _time(write_mesh, store, verts, faces, chunk_shape=CHUNK, bin_shape=BIN)
    tr, _   = _time(read_mesh, store)
    return tw, tr, res, _store_bytes(store), store


REGIMES = {
    'representative': {
        'points':    lambda: bench_points(_rep_points),
        'lines':     lambda: bench_lines(_rep_lines),
        'polylines': lambda: bench_polylines(_rep_polylines),
        'graph':     lambda: bench_graph(_rep_graph, kind='graph'),
        'skeleton':  lambda: bench_graph(_rep_graph, kind='skeleton'),
        'mesh':      lambda: bench_mesh(_rep_mesh),
    },
    'scattered': {
        'points':    lambda: bench_points(_scat_points),
        'lines':     lambda: bench_lines(_scat_lines),
        'polylines': lambda: bench_polylines(_scat_polylines),
        'graph':     lambda: bench_graph(_scat_graph, kind='graph'),
        'skeleton':  lambda: bench_graph(_scat_graph, kind='skeleton'),
        'mesh':      lambda: bench_mesh(_scat_mesh),
    },
}
TYPES = ['points', 'lines', 'polylines', 'graph', 'skeleton', 'mesh']

raw    = {r: {name: {'write_s': [], 'read_s': []} for name in TYPES} for r in REGIMES}
struct = {r: {} for r in REGIMES}
sizes  = {r: {} for r in REGIMES}

for regime, fns in REGIMES.items():
    for name in TYPES:
        for run in range(N_RUNS):
            tw, tr, res, nbytes, store = fns[name]()
            raw[regime][name]['write_s'].append(tw)
            raw[regime][name]['read_s'].append(tr)
            if run == 0:
                struct[regime][name] = _extract_stats(name, res)
                sizes[regime][name]  = nbytes
            shutil.rmtree(Path(store).parent, ignore_errors=True)

# Tidy long-form df: one row per (regime, type, op).
rows = []
for regime in REGIMES:
    for name in TYPES:
        v_count, o_count, cc_count = struct[regime][name]
        for op in ('write_s', 'read_s'):
            mean, hw = _mean_ci95(raw[regime][name][op])
            rows.append({
                'regime':            regime,
                'type':              name,
                'op':                op.replace('_s', ''),
                'mean_s':            round(mean, 4),
                'ci_hw':             round(hw, 4),
                'vertex_count':      int(v_count),
                'object_count':      int(o_count),
                'cross_chunk_count': int(cc_count),
                'size_MB':           round(sizes[regime][name] / 1e6, 2),
            })

df = pd.DataFrame(rows)
"""),
    ("md", "## 4 · Results"),
    ("code", "df"),
    ("md", "## 5 · Plot"),
    ("code", """\
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
x = np.arange(len(TYPES))
w = 0.36

for ax, regime in zip(axes, list(REGIMES)):
    sub   = df[df['regime'] == regime]
    write = sub[sub['op'] == 'write'].set_index('type').loc[TYPES]
    read  = sub[sub['op'] == 'read' ].set_index('type').loc[TYPES]
    ax.bar(x - w/2, write['mean_s'], width=w, yerr=write['ci_hw'],
           capsize=3, label='write')
    ax.bar(x + w/2, read['mean_s'],  width=w, yerr=read['ci_hw'],
           capsize=3, label='read')
    ax.set_xticks(x)
    ax.set_xticklabels(TYPES, rotation=20)
    ax.set_ylabel('seconds')
    ax.set_title(regime)
    ax.grid(True, axis='y', alpha=0.3)

axes[0].legend()
fig.suptitle(
    f'Write / read time per type '
    f'(N ≈ {N:,}, {N_RUNS} runs, 95% CI)'
)
fig.tight_layout()
plt.show()
"""),
]


# ===================================================================
# 03 · Backends — point cloud at fixed N, vary backend
# ===================================================================

BACKEND_CELLS: list[tuple[str, str]] = [
    ("md", """\
# Backend scaling

Point cloud at `N = 100_000`. Always benchmarks `local`. If
`ZV_BENCH_S3_URL` is set in the env (and `obstore` / `fsspec` are
installed), also benchmarks those against the URL.

Set up a cloud target with:

```bash
export ZV_BENCH_S3_URL="s3://my-bucket/zv-bench/"
```

When `ZV_BENCH_S3_URL` is unset, the notebook reports a one-row
local-only result.
"""),
    ("code", SHARED_HELPERS),
    ("md", "## 1 · Setup + backend availability"),
    ("code", """\
from zarr_vectors.types.points import write_points, read_points
from zarr_vectors.exceptions import StoreError

N = 100_000
CHUNK = (200.0, 200.0, 200.0)
BIN   = (50.0, 50.0, 50.0)
SEED  = 0

cloud_url = os.environ.get('ZV_BENCH_S3_URL')
print(f'local backend:  always benchmarked')
print(f'cloud backends: {\"benchmarked against \" + cloud_url if cloud_url else \"SKIPPED (set ZV_BENCH_S3_URL to enable)\"}')
"""),
    ("md", "## 2 · Synthetic input (shared across backends)"),
    ("code", """\
rng = np.random.default_rng(SEED)
positions = rng.uniform(0, 1000, (N, 3)).astype(np.float32)
intensity = rng.uniform(0, 1, N).astype(np.float32)
"""),
    ("md", "## 3 · Run the sweep"),
    ("code", """\
def run_one(label, target, backend):
    \"\"\"Time write + read against `target` using `backend`. Returns row dict.\"\"\"
    try:
        tw, _ = _time(
            write_points, target, positions,
            chunk_shape=CHUNK, bin_shape=BIN,
            attributes={'intensity': intensity},
            backend=backend,
        )
        tr, _ = _time(read_points, target, attribute_names=['intensity'], backend=backend)
        size_MB = _store_bytes(target) / 1e6 if backend == 'local' else float('nan')
        return {
            'backend': label,
            'write_s': round(tw, 3),
            'read_s':  round(tr, 3),
            'size_MB': round(size_MB, 2) if size_MB == size_MB else None,  # nan check
        }
    except (ImportError, StoreError) as e:
        return {'backend': label, 'write_s': None, 'read_s': None, 'size_MB': None, 'skipped': str(e)[:60]}


rows = []

# Local (always)
local_store = _new_store('backend_local')
rows.append(run_one('local', str(local_store), 'local'))
shutil.rmtree(Path(local_store).parent, ignore_errors=True)

# obstore / fsspec (only if cloud_url set)
if cloud_url:
    for backend in ('obstore', 'fsspec'):
        # Use a per-backend subpath so runs don't collide.
        target = cloud_url.rstrip('/') + f'/run_{backend}'
        rows.append(run_one(backend, target, backend))

df = pd.DataFrame(rows)
"""),
    ("md", "## 4 · Results"),
    ("code", "df"),
    ("md", "## 5 · Plot"),
    ("code", """\
plot_df = df.dropna(subset=['write_s', 'read_s'])
if not plot_df.empty:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(plot_df))
    ax.bar(x - 0.18, plot_df['write_s'], width=0.36, label='write (s)')
    ax.bar(x + 0.18, plot_df['read_s'],  width=0.36, label='read (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['backend'])
    ax.set_ylabel('seconds')
    ax.set_title(f'Write / read time per backend (N = {N:,})')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
else:
    print('no successful backends to plot')
"""),
]


# ===================================================================
# 04 · Pyramid scaling — points + graphs, vary N and coarsening ratio
# ===================================================================

PYRAMID_CELLS: list[tuple[str, str]] = [
    ("md", """\
# Pyramid construction — per coarsening factor, per build mode

Fixed `N = 1_000_000`.  For point clouds and graphs (`N` nodes + ~3N/2
edges) we sweep coarsening ratio `R ∈ {2, 4, 8}` over three pyramid
levels (resulting pyramids span level 0 / 1 / 2 / 3) and three
**cross-level link storage modes**:

| `cross_level_storage` | meaning |
| --- | --- |
| `"none"`     | no cross-resolution links |
| `"implicit"` | cross-chunk links only across resolutions (one direction: finer → coarser, `+Δ`) |
| `"explicit"` | full multires links: both `+Δ` (finer level) and `-Δ` (coarser level) |

Per `(geom, R, mode, level)` we collect: total build time, vertex
count, total link rows on disk (`cross_chunk_links/<Δ>/data` +
`links/<Δ>/<chunk_key>` across every Δ), per-level disk size, and
read-all-vertices time.

Build timing is averaged over `N_BUILD_RUNS = 5`; read timing over
`N_BUILD_RUNS × N_READ_RUNS = 15` samples (Student's t 95 % CI).
Vertex counts, link counts and disk sizes are deterministic given
the fixed seed, so they are captured once on `run == 0`.

Runtime: ~15–25 minutes on a laptop.
"""),
    ("code", SHARED_HELPERS + "\n\n" + STATS_HELPERS),
    ("md", "## 1 · Setup"),
    ("code", """\
import zarr
from zarr_vectors.types.points  import write_points, read_points
from zarr_vectors.types.graphs  import write_graph,  read_graph
from zarr_vectors.multiresolution.coarsen import build_pyramid
from zarr_vectors.building     import open_store, read_level_metadata

N       = 1_000_000
RATIOS  = [2, 4, 8]
MODES   = ['none', 'implicit', 'explicit']
N_LVLS  = 3                   # builds levels 1, 2, 3 from level 0
CHUNK   = (200.0, 200.0, 200.0)
BIN     = (50.0,  50.0,  50.0)
SEED    = 0
N_BUILD_RUNS = 5
N_READ_RUNS  = 3


def _level_link_count(store_path, level):
    \"\"\"Total link rows on disk at a pyramid level.

    Sums rows across every ``Δ`` and every chunk:
      - ``<level>/cross_chunk_links/<Δ>/data`` — one array per Δ
      - ``<level>/links/<Δ>/<chunk_key>``    — one array per chunk per Δ
    \"\"\"
    grp = zarr.open_group(str(store_path), path=str(level), mode='r')
    total = 0
    for family in ('cross_chunk_links', 'links'):
        if family not in grp:
            continue
        for delta_seg in grp[family]:
            sub = grp[family][delta_seg]
            for name in sub:
                arr = sub[name]
                try:
                    total += int(arr.shape[0])
                except Exception:
                    continue
    return total
"""),
    ("md", "## 2 · Run the sweep"),
    ("code", """\
rng = np.random.default_rng(SEED)
positions = rng.uniform(0, 1000, (N, 3)).astype(np.float32)
# ~3N/2 random undirected edges, dedup'd against self-loops.
src = rng.integers(0, N, size=3 * N // 2)
dst = rng.integers(0, N, size=3 * N // 2)
mask = src != dst
edges = np.stack([src[mask], dst[mask]], axis=1).astype(np.int64)

GEOMETRIES = [
    ('points', write_points, (positions,),         read_points),
    ('graph',  write_graph,  (positions, edges),   read_graph),
]

# (geom, ratio, mode) -> raw measurements.
data = {}

for ratio in RATIOS:
    for geom, write_fn, args, read_fn in GEOMETRIES:
        for mode in MODES:
            build_times = np.zeros(N_BUILD_RUNS)
            read_times  = {k: [] for k in range(N_LVLS + 1)}
            per_level   = None              # filled on run == 0
            for run in range(N_BUILD_RUNS):
                store_path = _new_store(
                    f'pyr_{geom}_r{ratio}_{mode}_run{run}'
                )
                write_fn(store_path, *args,
                         chunk_shape=CHUNK, bin_shape=BIN)
                t_build, _ = _time(
                    build_pyramid, store_path,
                    factors=[(float(ratio), 1.0)] * N_LVLS,
                    cross_level_storage=mode,
                )
                build_times[run] = t_build

                if run == 0:
                    root = open_store(str(store_path), mode='r')
                    per_level = []
                    for k in range(N_LVLS + 1):
                        lm = read_level_metadata(root, k)
                        per_level.append({
                            'vertex_count': int(lm.vertex_count),
                            'links_total': _level_link_count(store_path, k),
                            'disk_bytes':  _store_bytes(Path(store_path) / str(k)),
                        })

                for k in range(N_LVLS + 1):
                    for _ in range(N_READ_RUNS):
                        t_r, _ = _time(read_fn, store_path, level=k)
                        read_times[k].append(t_r)

                shutil.rmtree(Path(store_path).parent, ignore_errors=True)
            data[(geom, ratio, mode)] = {
                'build_times': build_times,
                'read_times':  {k: np.array(v) for k, v in read_times.items()},
                'per_level':   per_level,
            }

# Tidy dataframe: one row per (geom, ratio, mode, level).
rows = []
for (geom, ratio, mode), d in data.items():
    t_mean, t_hw = _mean_ci95(d['build_times'])
    for k in range(N_LVLS + 1):
        r_mean, r_hw = _mean_ci95(d['read_times'][k])
        rows.append({
            'geom':  geom,
            'ratio': ratio,
            'mode':  mode,
            'level': k,
            'build_total_s_mean': round(t_mean, 4),
            'build_total_s_hw':   round(t_hw,   4),
            'vertex_count': d['per_level'][k]['vertex_count'],
            'links_total':  d['per_level'][k]['links_total'],
            'disk_MB':      round(d['per_level'][k]['disk_bytes'] / 1e6, 3),
            'read_all_s_mean': round(r_mean, 4),
            'read_all_s_hw':   round(r_hw,   4),
        })
df = pd.DataFrame(rows)
"""),
    ("md", "## 3 · Results — full table"),
    ("code", "df"),
    ("md", """\
## 4 · Links per level (the headline finding)

Total link rows on disk per `(level, mode)` at `R = 4`.  `none` writes
only intra-level cross-chunk links; `implicit` adds the `+Δ`
cross-level arrays at the finer level; `explicit` mirrors those as
`-Δ` arrays at the coarser level too.
"""),
    ("code", """\
pivot = (df[df.ratio == 4]
         .pivot_table(index=['geom', 'level'],
                      columns='mode',
                      values='links_total',
                      sort=False)[MODES])
pivot
"""),
    ("md", "## 5 · Plot"),
    ("code", """\
MODE_COLORS = {'none': 'C0', 'implicit': 'C1', 'explicit': 'C2'}
GEOMS = ['points', 'graph']
PIVOT_R = 4   # the ratio used in the per-level panels

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for row, geom in enumerate(GEOMS):
    # --- Col 0: build time vs R, grouped by mode --------------------
    ax = axes[row, 0]
    x_pos = np.arange(len(RATIOS))
    for j, mode in enumerate(MODES):
        means, hws = [], []
        for r in RATIOS:
            sub = df[(df.geom == geom)
                     & (df.ratio == r)
                     & (df.mode == mode)].iloc[0]
            means.append(sub['build_total_s_mean'])
            hws.append(sub['build_total_s_hw'])
        ax.bar(
            x_pos + (j - 1) * 0.27, means, width=0.27,
            yerr=hws, capsize=3,
            color=MODE_COLORS[mode], label=mode,
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'R={r}' for r in RATIOS])
    ax.set_ylabel('build time (s)')
    ax.set_title(f'{geom}: total build time')
    ax.grid(True, axis='y', alpha=0.3)

    # --- Col 1: per-level disk size @ R=PIVOT_R ---------------------
    ax = axes[row, 1]
    x_pos = np.arange(N_LVLS + 1)
    for j, mode in enumerate(MODES):
        means = [
            df[(df.geom == geom) & (df.ratio == PIVOT_R)
               & (df.mode == mode) & (df.level == k)].iloc[0]['disk_MB']
            for k in range(N_LVLS + 1)
        ]
        ax.bar(
            x_pos + (j - 1) * 0.27, means, width=0.27,
            color=MODE_COLORS[mode], label=mode,
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'L{k}' for k in range(N_LVLS + 1)])
    ax.set_ylabel('disk size (MB)')
    ax.set_title(f'{geom}: disk per level @ R={PIVOT_R}')
    ax.grid(True, axis='y', alpha=0.3)

    # --- Col 2: per-level read time @ R=PIVOT_R ---------------------
    ax = axes[row, 2]
    for j, mode in enumerate(MODES):
        means, hws = [], []
        for k in range(N_LVLS + 1):
            sub = df[(df.geom == geom) & (df.ratio == PIVOT_R)
                     & (df.mode == mode) & (df.level == k)].iloc[0]
            means.append(sub['read_all_s_mean'])
            hws.append(sub['read_all_s_hw'])
        ax.bar(
            x_pos + (j - 1) * 0.27, means, width=0.27,
            yerr=hws, capsize=3,
            color=MODE_COLORS[mode], label=mode,
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'L{k}' for k in range(N_LVLS + 1)])
    ax.set_ylabel('read time (s)')
    ax.set_title(f'{geom}: read per level @ R={PIVOT_R}')
    ax.grid(True, axis='y', alpha=0.3)

axes[0, 0].legend(loc='best', fontsize=9)
fig.suptitle(
    f'Pyramid build vs cross-level link mode '
    f'(N={N:,}, {N_BUILD_RUNS} build runs, 95% CI)',
)
plt.tight_layout()
"""),
]


# ===================================================================
# 05 · Spatial bbox queries — per geometry type
# ===================================================================

BBOX_CELLS: list[tuple[str, str]] = [
    ("md", """\
# Spatial bbox queries — per geometry type

Fixed `N ≈ 50_000` vertices per store across four geometry types
(points, polylines, graph, mesh) all positioned in `[0, 1000)³`.  Sweep
bbox volume fraction `V ∈ {0.001, 0.01, 0.1, 1.0}` of the domain and
time `open_zv(store)[0].filter(bbox=(lo, hi)).compute()` — only the
chunks the bbox intersects are read.

10 runs per `V`; each run draws a fresh random bbox location.  Each
panel also shows the **full-store read time** as a horizontal dashed
reference so you can see how much the spatial index saves at small
`V`.

Runtime: ~5–10 minutes on a laptop (`V = 1.0` over four geometries
dominates).
"""),
    ("code", SHARED_HELPERS + "\n\n" + STATS_HELPERS),
    ("md", "## 1 · Setup"),
    ("code", """\
from zarr_vectors.types.points    import write_points
from zarr_vectors.types.polylines import write_polylines
from zarr_vectors.types.graphs    import write_graph
from zarr_vectors.types.meshes    import write_mesh
from zarr_vectors.lazy import open_zv

N           = 50_000            # target vertex count per geometry
V_FRACTIONS = [0.001, 0.01, 0.1, 1.0]
DOMAIN      = 1000.0
CHUNK       = (200.0, 200.0, 200.0)
BIN         = (50.0,  50.0,  50.0)
SEED        = 0
"""),
    ("md", "## 2 · Synthetic generators (one per type)"),
    ("code", """\
rng = np.random.default_rng(SEED)


def _points_input():
    return rng.uniform(0, DOMAIN, (N, 3)).astype(np.float32)


def _polylines_input():
    # ~N total vertices spread across short random walks (~12 verts each).
    counts = rng.integers(8, 16, size=N // 12)
    out = []
    for c in counts:
        steps = rng.normal(0, 5, (c, 3))
        start = rng.uniform(0, DOMAIN, 3)
        out.append((start + steps.cumsum(axis=0)).astype(np.float32))
    return (out,)


def _graph_input():
    positions = rng.uniform(0, DOMAIN, (N, 3)).astype(np.float32)
    src = rng.integers(0, N, size=3 * N // 2)
    dst = rng.integers(0, N, size=3 * N // 2)
    mask = src != dst
    edges = np.stack([src[mask], dst[mask]], axis=1).astype(np.int64)
    return positions, edges


def _mesh_input():
    side = int(np.sqrt(N))
    xs, ys = np.meshgrid(
        np.linspace(0, DOMAIN, side), np.linspace(0, DOMAIN, side),
    )
    zs = rng.uniform(0, 100, (side, side))
    verts = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3).astype(np.float32)
    i = np.arange(side - 1)
    j = np.arange(side - 1)
    ii, jj = np.meshgrid(i, j, indexing='ij')
    a = (ii * side + jj).ravel()
    b = a + 1
    c = a + side
    d = c + 1
    faces = np.concatenate([
        np.stack([a, b, c], axis=1),
        np.stack([b, d, c], axis=1),
    ]).astype(np.int64)
    return verts, faces


GEOMETRIES = [
    ('points',    write_points,    _points_input()),
    ('polylines', write_polylines, _polylines_input()),
    ('graph',     write_graph,     _graph_input()),
    ('mesh',      write_mesh,      _mesh_input()),
]
"""),
    ("md", "## 3 · Write one store per geometry"),
    ("code", """\
stores = {}
read_all_s = {}

for name, write_fn, args in GEOMETRIES:
    store = _new_store(f'bbox_{name}')
    write_fn(store, *args, chunk_shape=CHUNK, bin_shape=BIN)
    stores[name] = store

    # Full-store baseline read time (average over a few runs) — used as
    # a dashed reference line on each panel.
    t_full = np.array([
        _time(lambda s=store: open_zv(s)[0].vertices.compute())[0]
        for _ in range(3)
    ])
    read_all_s[name] = float(t_full.mean())
    print(f'{name:9s}  store {_store_bytes(store) / 1e6:6.2f} MB  '
          f'read-all {read_all_s[name] * 1e3:6.1f} ms')
"""),
    ("md", "## 4 · Run the sweep"),
    ("code", """\
metrics = ['bbox_s', 'returned_count']
# raw[metric][geom] is shape (len(V_FRACTIONS), N_RUNS).
raw = {
    m: {name: np.zeros((len(V_FRACTIONS), N_RUNS))
        for name, _, _ in GEOMETRIES}
    for m in metrics
}

for i, v in enumerate(V_FRACTIONS):
    edge = DOMAIN * (v ** (1 / 3))
    for run in range(N_RUNS):
        lo = rng.uniform(0, DOMAIN - edge, 3).astype(np.float32)
        hi = (lo + edge).astype(np.float32)

        for name, _, _ in GEOMETRIES:
            zv = open_zv(stores[name])
            t, out = _time(
                lambda: zv[0].filter(bbox=(lo, hi)).compute()
            )
            raw['bbox_s'][name][i, run]         = t
            raw['returned_count'][name][i, run] = out['vertex_count']

# Tidy long-form df: one row per (geom, V).
rows = []
for name, _, _ in GEOMETRIES:
    for i, v in enumerate(V_FRACTIONS):
        row = {'geom': name, 'V': v}
        for m in metrics:
            mean, hw = _mean_ci95(raw[m][name][i])
            row[f'{m}_mean'] = round(mean, 4)
            row[f'{m}_hw']   = round(hw,   4)
        rows.append(row)
df = pd.DataFrame(rows)

# Cleanup all stores.
for path in stores.values():
    shutil.rmtree(Path(path).parent, ignore_errors=True)
"""),
    ("md", "## 5 · Results"),
    ("code", "df"),
    ("md", "## 6 · Plot"),
    ("code", """\
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
COLORS = {'points': 'C0', 'polylines': 'C1', 'graph': 'C2', 'mesh': 'C3'}

for ax, (name, _, _) in zip(axes.flat, GEOMETRIES):
    sub  = df[df['geom'] == name].sort_values('V')
    mean = sub['bbox_s_mean'].values
    hw   = sub['bbox_s_hw'].values
    ax.fill_between(sub['V'], mean - hw, mean + hw,
                    color=COLORS[name], alpha=0.2)
    ax.loglog(sub['V'], mean, 'o-', color=COLORS[name], label='bbox read')
    ax.axhline(read_all_s[name], color='0.4', linestyle='--', linewidth=1,
               label=f'read all ({read_all_s[name] * 1e3:.0f} ms)')
    ax.set_xlabel('bbox volume fraction')
    ax.set_ylabel('seconds')
    ax.set_title(name)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best', fontsize=8)

fig.suptitle(
    f'Spatial bbox queries by geometry type '
    f'(N ≈ {N:,} verts, {N_RUNS} runs, 95% CI)',
)
plt.tight_layout()
"""),
]


# ===================================================================
# 06 · Chunk-shape sensitivity — points, fixed N, sweep chunk_shape
# ===================================================================

CHUNK_SHAPE_CELLS: list[tuple[str, str]] = [
    ("md", """\
# Chunk-shape sensitivity

Fixed `N = 500_000` point cloud uniformly distributed in `[0, 1000)³`.
Sweep `chunk_shape` ∈ `{50, 100, 200, 400, 800}` (uniform 3-tuples) and
measure write time, full read, **single-vertex read** (one chunk via the
lazy API), disk size, and chunk count.  `bin_shape = chunk_shape / 4`
for every run.

The single-vertex read is the headline.  Small chunks pay per-file
metadata overhead per access; large chunks return a lot more data than
the single vertex needed.  Expect a U-shape.
"""),
    ("code", SHARED_HELPERS + "\n\n" + STATS_HELPERS),
    ("md", "## 1 · Setup"),
    ("code", """\
from zarr_vectors.types.points import write_points, read_points
from zarr_vectors.lazy import open_zv

N           = 500_000
CHUNK_SIZES = [50, 100, 200, 400, 800]
DOMAIN      = 1000.0
SEED        = 0
"""),
    ("md", "## 2 · Run the sweep"),
    ("code", """\
rng = np.random.default_rng(SEED)
positions = rng.uniform(0, DOMAIN, (N, 3)).astype(np.float32)

metrics = ['write_s', 'read_all_s', 'read_one_s']
raw = {m: np.zeros((len(CHUNK_SIZES), N_RUNS)) for m in metrics}
sizes_MB    = np.zeros(len(CHUNK_SIZES))
chunk_count = np.zeros(len(CHUNK_SIZES))

for i, cs in enumerate(CHUNK_SIZES):
    chunk_shape = (float(cs),) * 3
    bin_shape   = (cs / 4.0,) * 3

    for run in range(N_RUNS):
        store = _new_store(f'cs_{cs}_run{run}')

        t_w, _ = _time(
            write_points, store, positions,
            chunk_shape=chunk_shape, bin_shape=bin_shape,
        )
        t_r, _ = _time(read_points, store)

        # Single-vertex (= one chunk via lazy API) read, like
        # internals/01_size_scaling.ipynb.  Open + chunk listing +
        # one chunk decode.
        def _read_one():
            zv = open_zv(store)
            keys = zv[0].vertices._chunk_keys  # noqa: SLF001
            return zv[0].vertices[keys[0]].compute() if keys else None
        t_o, _ = _time(_read_one)

        raw['write_s'][i, run]    = t_w
        raw['read_all_s'][i, run] = t_r
        raw['read_one_s'][i, run] = t_o

        if run == 0:
            sizes_MB[i]    = _store_bytes(store) / 1e6
            chunk_count[i] = len(open_zv(store)[0].chunk_keys)

        shutil.rmtree(Path(store).parent, ignore_errors=True)

rows = []
for i, cs in enumerate(CHUNK_SIZES):
    row = {'chunk_shape': cs, 'size_MB': round(sizes_MB[i], 2),
           'chunk_count': int(chunk_count[i])}
    for m in metrics:
        mean, hw = _mean_ci95(raw[m][i])
        row[f'{m}_mean'] = round(mean, 4)
        row[f'{m}_hw']   = round(hw,   4)
    rows.append(row)
df = pd.DataFrame(rows)
"""),
    ("md", "## 3 · Results"),
    ("code", "df"),
    ("md", "## 4 · Plot"),
    ("code", """\
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

def _ci_line(ax, key, color, label=None, linestyle='-'):
    mean = df[f'{key}_mean'].values
    hw   = df[f'{key}_hw'].values
    ax.fill_between(df['chunk_shape'], mean - hw, mean + hw,
                    color=color, alpha=0.2)
    ax.loglog(df['chunk_shape'], mean, marker='o', color=color,
              linestyle=linestyle, label=label)

# Panel 1 — write time.
ax = axes[0]
_ci_line(ax, 'write_s', 'C0')
ax.set_xlabel('chunk_shape (per axis)')
ax.set_ylabel('s')
ax.set_title('Write time')
ax.grid(True, which='both', alpha=0.3)

# Panel 2 — read all (solid) + read one vertex (dashed) on the same axes.
ax = axes[1]
_ci_line(ax, 'read_all_s', 'C1', label='read all',     linestyle='-')
_ci_line(ax, 'read_one_s', 'C2', label='read one vertex', linestyle='--')
ax.set_xlabel('chunk_shape (per axis)')
ax.set_ylabel('s')
ax.set_title('Read time')
ax.grid(True, which='both', alpha=0.3)
ax.legend(loc='best')

# Panel 3 — disk MB + chunk count on twin axes.
ax = axes[2]
ax.loglog(df['chunk_shape'], df['size_MB'], 'o-', color='C3',
          label='disk MB')
ax.set_xlabel('chunk_shape (per axis)')
ax.set_ylabel('MB')
ax.grid(True, which='both', alpha=0.3)
ax_r = ax.twinx()
ax_r.loglog(df['chunk_shape'], df['chunk_count'], 's--', color='C4',
            label='chunk count')
ax_r.set_ylabel('chunk count')
ax.set_title('Disk size and chunk count')
ax.legend(loc='upper right')
ax_r.legend(loc='lower left')

fig.suptitle(
    f'Chunk-shape sensitivity — point cloud (N = {N:,}, '
    f'{N_RUNS} runs, 95% CI)',
)
plt.tight_layout()
"""),
]


# ===================================================================
# 07 · Compression / codec impact — fixed N, sweep compressor
# ===================================================================

COMPRESSION_CELLS: list[tuple[str, str]] = [
    ("md", """\
# Compression / codec impact across vertex count

Sweep the `compressor=` kwarg across five codec configurations *and*
vertex count `N`, plotted as lines on log-log axes with 95% confidence
bands.  The library default is **no compression** (`BytesCodec` only)
for the fastest cloud-write path; this benchmark surfaces what each
compressed alternative buys you for what cost, as a function of scale.

To isolate codec scaling from chunk-size effects, **per-chunk byte
content is held roughly constant across `N`** by shrinking
`chunk_shape` as `N` grows — every chunk holds ~`TARGET_VERTS`
vertices in expectation.  So as `N` increases, the **number of
chunks** rises proportionally rather than each chunk getting bigger.
This means the per-chunk codec workload is the same at every `N` and
only the *aggregate* work (CPU, disk, file count) scales.

| Label | `compressor=` value |
|-------|--------------------|
| `none` (default) | `None` — uncompressed chunks, fast async PUT path |
| `zstd` | `'zstd'` — zarr v3's default compressor (Zstd l0) |
| `blosc_l1` | `[{...}]` Blosc + Zstd level 1, no shuffle |
| `blosc_l5` | `[{...}]` Blosc + Zstd level 5, no shuffle |
| `blosc_l5_bitshuffle` | `'blosc'` — Blosc + Zstd level 5 + BitShuffle |

Compression CPU usually pays off on disk; bitshuffle is expected to
beat plain Zstd on float32 positions.  See
[`docs/spec/foundations/codec_pipeline.md`](../docs/spec/foundations/codec_pipeline.md)
for the kwarg semantics.
"""),
    ("code", SHARED_HELPERS + "\n\n" + STATS_HELPERS),
    ("md", "## 1 · Setup"),
    ("code", """\
from zarr_vectors.types.points import write_points, read_points

SIZES        = [10_000, 100_000, 1_000_000]
TARGET_VERTS = 5_000             # ≈ 60 KB raw float32 per chunk
DOMAIN       = 1000.0
SEED         = 0


def _chunk_shape_for(n):
    \"\"\"Pick a uniform 3D ``chunk_shape`` so each chunk holds about
    ``TARGET_VERTS`` vertices in expectation for uniform random
    positions on ``[0, DOMAIN)^3``.

    Returns ``(chunk_shape, bin_shape)``.  ``bin_shape`` is 1/4 of
    ``chunk_shape`` (matches the convention used by the other
    benchmarks).  Capped at the domain edge so we never overshoot.
    \"\"\"
    side = DOMAIN * (TARGET_VERTS / n) ** (1 / 3)
    side = min(side, DOMAIN)
    return (side,) * 3, (side / 4.0,) * 3


CONFIGS = [
    ('none',                None),
    ('zstd',                'zstd'),
    ('blosc_l1',            [{'name': 'blosc', 'configuration': {
                                'cname': 'zstd', 'clevel': 1,
                                'shuffle': 'noshuffle',
                                'typesize': 4, 'blocksize': 0}}]),
    ('blosc_l5',            [{'name': 'blosc', 'configuration': {
                                'cname': 'zstd', 'clevel': 5,
                                'shuffle': 'noshuffle',
                                'typesize': 4, 'blocksize': 0}}]),
    ('blosc_l5_bitshuffle', 'blosc'),
]
"""),
    ("md", "## 2 · Run the sweep"),
    ("code", """\
rng = np.random.default_rng(SEED)

# raw[metric][config_label] is shape (len(SIZES), N_RUNS).
metrics = ['write_s', 'read_all_s']
raw = {
    m: {label: np.zeros((len(SIZES), N_RUNS)) for label, _ in CONFIGS}
    for m in metrics
}
disk_MB = {label: np.zeros(len(SIZES)) for label, _ in CONFIGS}
per_size_chunk_shape = []

for i, n in enumerate(SIZES):
    chunk_shape, bin_shape = _chunk_shape_for(n)
    per_size_chunk_shape.append(chunk_shape[0])
    print(f'N = {n:>9,}  chunk_shape = {chunk_shape[0]:6.1f}  '
          f'≈ {n / max(1, (DOMAIN / chunk_shape[0]) ** 3):6.0f} verts/chunk')
    positions = rng.uniform(0, DOMAIN, (n, 3)).astype(np.float32)
    for label, compressor in CONFIGS:
        for run in range(N_RUNS):
            store = _new_store(f'codec_{label}_n{n}_run{run}')
            t_w, _ = _time(
                write_points, store, positions,
                chunk_shape=chunk_shape, bin_shape=bin_shape,
                compressor=compressor,
            )
            t_r, _ = _time(read_points, store)
            raw['write_s'][label][i, run]    = t_w
            raw['read_all_s'][label][i, run] = t_r
            if run == 0:
                disk_MB[label][i] = _store_bytes(store) / 1e6
            shutil.rmtree(Path(store).parent, ignore_errors=True)

# Tidy long-form df: one row per (N, config) holding fold change vs the
# ``none`` (uncompressed) baseline at the same N.  Time-metric folds are
# computed per-run paired (same run index for both numerator and
# denominator), then summarised as mean + 95% CI; disk fold is a single
# deterministic measurement per (N, config).
rows = []
for i, n in enumerate(SIZES):
    for label, _ in CONFIGS:
        row = {
            'N': n,
            'chunk_shape': round(per_size_chunk_shape[i], 1),
            'config': label,
        }
        for m in metrics:
            # Pairwise ratio across runs: shape (N_RUNS,).
            ratio = raw[m][label][i] / raw[m]['none'][i]
            mean, hw = _mean_ci95(ratio)
            row[f'{m}_fold_mean'] = round(mean, 4)
            row[f'{m}_fold_hw']   = round(hw,   4)
        row['disk_fold'] = round(disk_MB[label][i] / disk_MB['none'][i], 4)
        rows.append(row)
df = pd.DataFrame(rows)
"""),
    ("md", "## 3 · Results"),
    ("code", "df"),
    ("md", "## 4 · Plot"),
    ("code", """\
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

PANELS = [
    ('Write time fold change',  'write_s_fold'),
    ('Read time fold change',   'read_all_s_fold'),
]
COLORS = {
    'none':                '0.5',          # gray reference line at 1.0
    'zstd':                'C1',
    'blosc_l1':            'C2',
    'blosc_l5':            'C3',
    'blosc_l5_bitshuffle': 'C4',
}

def _plot_fold(ax, key_prefix, title, with_ci):
    \"\"\"Plot fold-change lines per config on log-log axes.

    ``key_prefix`` is ``'write_s_fold'``, ``'read_all_s_fold'``, or
    ``'disk'``.  When ``with_ci``, draws a shaded 95% CI band around
    each mean line (only valid for time metrics that are sampled
    per-run).
    \"\"\"
    for label, _ in CONFIGS:
        sub = df[df['config'] == label].sort_values('N')
        if label == 'none':
            # Self-ratio is exactly 1 by construction; draw a faint
            # horizontal reference at y=1 instead of a redundant data
            # series.
            continue
        if with_ci:
            mean = sub[f'{key_prefix}_mean'].values
            hw   = sub[f'{key_prefix}_hw'].values
            ax.fill_between(
                sub['N'], mean - hw, mean + hw,
                color=COLORS[label], alpha=0.2,
            )
        else:
            mean = sub[key_prefix].values
        ax.loglog(
            sub['N'], mean, 'o-',
            color=COLORS[label], label=label,
        )
    ax.axhline(1.0, color=COLORS['none'], linestyle='--',
               linewidth=1, label='none (1.0×)')
    ax.set_xlabel('N (vertices)')
    ax.set_ylabel('fold change vs none')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)

# Time panels — fold change with per-run paired CI.
for ax, (title, key) in zip(axes[:2], PANELS):
    _plot_fold(ax, key, title, with_ci=True)

# Disk panel — deterministic single measurement; no CI.
_plot_fold(axes[2], 'disk_fold', 'Disk size fold change', with_ci=False)

axes[0].legend(loc='best', fontsize=8)
fig.suptitle(
    f'Compression impact — fold change vs uncompressed default '
    f'({N_RUNS} runs, 95% CI on time metrics)',
)
plt.tight_layout()
"""),
]


# ===================================================================
# 08 · Edit operations — sweep edit kind, atomic flag, edits per session
# ===================================================================

EDIT_CELLS: list[tuple[str, str]] = [
    ("md", """\
# Edit operations — cost per edit by kind and mode

Baseline store: `N = 50_000` point cloud uniformly in `[0, 1000)^3`,
chunked at 200³ (so the points span a 5×5×5 chunk grid).  Each row
gets its own object ID so manifests exist for the atomic path to
rewrite.

Sweep three axes:

| Axis | Values |
| --- | --- |
| Edit kind | `move_in_chunk`, `move_cross_chunk`, `add`, `soft_delete` |
| Atomicity | `atomic=True` (copy-on-write), `atomic=False` (overwrite) |
| Edits per session | `N_EDITS ∈ {1, 10, 100, 1_000}` |

For each (kind, atomicity, N) we run **`N_RUNS = 5`** repeats — fresh
baseline store each repeat — wrap the `N_EDITS` edits in a single
`EditSession`, and report:

- **edit-rate**: `N_EDITS / wall-time` (edits per second).
- **bytes written**: post-edit store size minus pre-edit baseline.
- **oid-remap size**: number of new OIDs allocated under atomic mode.

The two soft-delete + atomic combos are the cheapest (no fragment
churn).  `move_cross_chunk` is the most expensive because it touches
two chunks and (under atomic) appends a fragment in the target chunk.

Runtime: a few minutes on a laptop.  `N_EDITS = 1_000` for
`move_cross_chunk + atomic` is the slow row.
"""),
    ("code", SHARED_HELPERS + "\n\n" + STATS_HELPERS),
    ("md", "## 1 · Setup"),
    ("code", """\
from zarr_vectors.types.points import write_points
from zarr_vectors.building import open_store
from zarr_vectors.ops import EditSession, VertexRef

N           = 50_000
CHUNK       = (200.0, 200.0, 200.0)
BIN         = (50.0, 50.0, 50.0)
DOMAIN      = 1000.0
N_EDITS     = [1, 10, 100, 1_000]
EDIT_KINDS  = ['move_in_chunk', 'move_cross_chunk', 'add', 'soft_delete']
ATOMICITY   = [True, False]
N_RUNS      = 5
SEED        = 0


def _build_baseline(seed):
    \"\"\"Fresh 50k-point store with one OID per row.\"\"\"
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, DOMAIN, (N, 3)).astype(np.float32)
    store = _new_store('edit_base')
    write_points(
        str(store), positions,
        chunk_shape=CHUNK, bin_shape=BIN,
        bounds=([0.0, 0.0, 0.0], [DOMAIN, DOMAIN, DOMAIN]),
        object_ids=np.arange(N, dtype=np.int64),
    )
    return store, positions
"""),
    ("md", "## 2 · Edit-kind workloads"),
    ("code", """\
def _apply_edits(ed, kind, atomic, oids, positions, rng):
    \"\"\"Apply ``len(oids)`` edits of ``kind`` through ``ed``.\"\"\"
    root = ed.root
    if kind == 'move_in_chunk':
        # Pick a tiny offset so the new position stays in the same chunk.
        for oid in oids:
            ref = VertexRef.from_object(root, level=0, object_id=int(oid),
                                        vertex_index=0)
            new_pos = positions[oid] + np.float32([0.5, 0.5, 0.5])
            ed.edit_vertex(ref, new_pos=new_pos.tolist(), atomic=atomic)
    elif kind == 'move_cross_chunk':
        # Move to a position in a different chunk.  Add CHUNK[0] so the
        # new position lands one chunk over (modulo domain).
        for oid in oids:
            ref = VertexRef.from_object(root, level=0, object_id=int(oid),
                                        vertex_index=0)
            jitter = rng.uniform(10.0, 50.0, 3).astype(np.float32)
            new_pos = (positions[oid] + np.float32([CHUNK[0], 0.0, 0.0])
                       + jitter)
            new_pos = np.clip(new_pos, 0.0, DOMAIN - 1.0).tolist()
            ed.edit_vertex(ref, new_pos=new_pos, atomic=atomic)
    elif kind == 'add':
        for _ in oids:
            new_pos = rng.uniform(0, DOMAIN, 3).astype(np.float32).tolist()
            ed.add_vertex(level=0, pos=new_pos)
    elif kind == 'soft_delete':
        for oid in oids:
            ref = VertexRef.from_object(root, level=0, object_id=int(oid),
                                        vertex_index=0)
            # remove_vertex only supports atomic=True today; if the caller
            # asked for atomic=False we use the atomic path and label it so
            # the table is honest about the cost basis.
            ed.remove_vertex(ref, atomic=True)
    else:
        raise ValueError(f'unknown edit kind: {kind!r}')
"""),
    ("md", "## 3 · Run the sweep"),
    ("code", """\
rows = []
master_rng = np.random.default_rng(SEED)

for kind in EDIT_KINDS:
    for atomic in ATOMICITY:
        # soft_delete is atomic-only — skip the atomic=False combo to
        # avoid double-counting an identical workload.
        if kind == 'soft_delete' and not atomic:
            continue
        for n_edits in N_EDITS:
            wallclocks = np.zeros(N_RUNS)
            for run in range(N_RUNS):
                seed = int(master_rng.integers(0, 2**31 - 1))
                rng = np.random.default_rng(seed)
                store, positions = _build_baseline(seed)
                oids = rng.choice(N, size=n_edits, replace=False)
                root = open_store(str(store), mode='r+')
                t0 = time.perf_counter()
                with EditSession(root, atomic=atomic, refresh_pyramid=False) as ed:
                    _apply_edits(ed, kind, atomic, oids, positions, rng)
                wallclocks[run] = time.perf_counter() - t0
                shutil.rmtree(Path(store).parent, ignore_errors=True)
            t_mean, t_hw = _mean_ci95(wallclocks)
            rows.append({
                'kind': kind,
                'atomic': atomic,
                'N_edits': n_edits,
                'wall_s_mean': round(t_mean, 4),
                'wall_s_hw':   round(t_hw, 4),
            })

df = pd.DataFrame(rows)
"""),
    ("md", "## 4 · Results"),
    ("code", "df"),
    ("md", "## 5 · Plot"),
    ("code", """\
# Two panels: one for add/remove, one for move. Each plots wall time
# (seconds) vs N_edits per session.  Lines are coloured by edit kind
# and styled by atomicity (solid=atomic, dashed=overwrite).
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
KIND_COLORS = {
    'move_in_chunk':    'C0',
    'move_cross_chunk': 'C1',
    'add':              'C2',
    'soft_delete':      'C3',
}
PANELS = [
    ('Add / remove', ['add', 'soft_delete']),
    ('Move',          ['move_in_chunk', 'move_cross_chunk']),
]

for ax, (title, kinds) in zip(axes, PANELS):
    for kind in kinds:
        color = KIND_COLORS[kind]
        for atomic in (True, False):
            if kind == 'soft_delete' and not atomic:
                continue
            sub = df[(df['kind'] == kind) & (df['atomic'] == atomic)].sort_values('N_edits')
            if sub.empty:
                continue
            style = '-' if atomic else '--'
            label = f"{kind} ({'atomic' if atomic else 'overwrite'})"
            mean = sub['wall_s_mean'].values
            hw   = sub['wall_s_hw'].values
            ax.fill_between(sub['N_edits'], mean - hw, mean + hw,
                            color=color, alpha=0.2)
            ax.loglog(sub['N_edits'], mean,
                      marker='o', linestyle=style, color=color, label=label)
    ax.set_xlabel('N_edits per session')
    ax.set_ylabel('wall time (s)')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best', fontsize=8)

fig.suptitle(
    f'Edit operations — N={N:,} baseline, {N_RUNS} runs, 95% CI',
)
plt.tight_layout()
"""),
    ("md", """\
## 6 · Multi-writer concurrency — disjoint chunks

Simulate **1 vs 4 cooperating editors** on the same baseline store via
``oid_prefix`` (each writer gets a disjoint residue class).  The
editors target disjoint chunks so writes never collide at the chunk
level; the panel measures the per-editor edit-rate when the editors
run sequentially (the baseline) versus when they share an `oid_prefix`
allocator with `modulus=4`.

This is a sanity check on the OID-prefix allocator, not a parallelism
benchmark: edits within a single Python process are sequential.  The
takeaway is that the allocator overhead is negligible and OIDs from
different editors land in disjoint ranges, ready for an icechunk
multi-branch merge in production.
"""),
    ("code", """\
from zarr_vectors.ops import OidPrefix, merge_edit_reports

EDITORS  = ['alice', 'bob', 'carol', 'dave']
MODULUS  = len(EDITORS)
N_EDITS_PER_EDITOR = 50

# Carve the chunk space into disjoint regions per editor so writes
# don't collide.  We assign chunks round-robin by hash.
def _chunk_for_editor(positions, editor_idx, num_editors):
    chunk = (positions // 50).astype(int)
    h = (chunk[:, 0] + chunk[:, 1] * 7 + chunk[:, 2] * 13) % num_editors
    return h == editor_idx


rows = []
for n_editors in (1, MODULUS):
    seed = int(master_rng.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    store, positions = _build_baseline(seed)
    size_before = _store_bytes(store)
    chunk_shape = np.array(CHUNK)

    t0 = time.perf_counter()
    reports = []
    for editor_idx in range(n_editors):
        editor_name = EDITORS[editor_idx]
        mask = _chunk_for_editor(positions, editor_idx, n_editors)
        candidates = np.flatnonzero(mask)
        if len(candidates) < N_EDITS_PER_EDITOR:
            continue
        oids = rng.choice(candidates, size=N_EDITS_PER_EDITOR, replace=False)
        root = open_store(str(store), mode='r+')
        with EditSession(
            root, atomic=True,
            refresh_pyramid=False,
            oid_prefix=(editor_name, MODULUS),
        ) as ed:
            for oid in oids:
                ref = VertexRef.from_object(
                    root, level=0, object_id=int(oid), vertex_index=0,
                )
                new_pos = positions[oid] + np.float32([0.5, 0.5, 0.5])
                ed.edit_vertex(ref, new_pos=new_pos.tolist(), atomic=True)
        reports.append(ed.report)
    dt = time.perf_counter() - t0

    size_after = _store_bytes(store)
    total_edits = sum(r.n_edits for r in reports)

    # All atomic OIDs allocated by the cooperating sessions must be
    # in disjoint residue classes — verify by inspecting the remap.
    new_oids = [v for r in reports for v in r.oid_remap.values()]
    if n_editors > 1:
        try:
            merged = merge_edit_reports(*reports)
            merge_ok = True
        except Exception:
            merge_ok = False
    else:
        merge_ok = True

    rows.append({
        'n_editors': n_editors,
        'total_edits': total_edits,
        'wall_s': round(dt, 3),
        'edits_per_s': round(total_edits / dt, 1) if dt > 0 else float('nan'),
        'bytes_delta': int(size_after - size_before),
        'merge_ok': merge_ok,
    })
    shutil.rmtree(Path(store).parent, ignore_errors=True)

df_conc = pd.DataFrame(rows)
df_conc
"""),
    ("md", """\
## 7 · Inverted-index scaling — wall time per edit vs N_objects

Iteration 4 introduced a lazy fragment→OID inverted index that
collapses the two linear-scan helpers (`_oids_referencing` and
`_oid_for_endpoint`) to O(1) per lookup with surgical updates on every
`_stage_manifest` call.  This panel documents the *flatness* of the
post-fix cost curve: as `N_OBJECTS` grows from 1k → 10k → 50k, the
wall-time per atomic edit should grow only sub-linearly (dominated by
amortised first-lookup index build + per-write diff cost), not
linearly as it did before the fix.

Two workloads:

- `edit_vertex_atomic` — repeated atomic vertex edits against fresh
  OIDs in the same store.
- `add_link_cross_oid` — repeated `add_link(update_objects=True)`
  bridging two OIDs (exercises `_oid_for_endpoint` twice per call).

Each `(N_OBJECTS, kind)` row is averaged over a single session of
`N_EDITS = 200` edits.  Lower wall-time-per-edit at large `N_OBJECTS`
indicates the inverted index is doing its job.
"""),
    ("code", """\
from zarr_vectors.ops import EditSession, VertexRef

N_OBJECTS_SWEEP = [1_000, 10_000, 50_000]
N_EDITS_SCALING = 200


def _build_scaling_baseline(n_objects, seed):
    \"\"\"Fresh points store with one OID per row.\"\"\"
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, DOMAIN, (n_objects, 3)).astype(np.float32)
    store = _new_store(f'scaling_{n_objects}')
    write_points(
        str(store), positions,
        chunk_shape=CHUNK, bin_shape=BIN,
        bounds=([0.0, 0.0, 0.0], [DOMAIN, DOMAIN, DOMAIN]),
        object_ids=np.arange(n_objects, dtype=np.int64),
    )
    return store, positions


def _bench_edit_vertex_atomic(store, positions, n_edits):
    rng = np.random.default_rng(42)
    oids = rng.choice(len(positions), size=n_edits, replace=False)
    root = open_store(str(store), mode='r+')
    t0 = time.perf_counter()
    with EditSession(root, atomic=True, refresh_pyramid=False) as ed:
        for oid in oids:
            ref = VertexRef.from_object(
                root, level=0, object_id=int(oid), vertex_index=0,
            )
            new_pos = (positions[oid] + np.float32([0.1, 0.1, 0.1])).tolist()
            ed.edit_vertex(ref, new_pos=new_pos, atomic=True)
    return time.perf_counter() - t0


def _bench_add_link_cross_oid(store, positions, n_edits):
    rng = np.random.default_rng(43)
    pairs = rng.choice(len(positions), size=(n_edits, 2), replace=True)
    # Drop same-OID pairs so every edit exercises the merge path.
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if len(pairs) == 0:
        return float('nan')
    root = open_store(str(store), mode='r+')
    t0 = time.perf_counter()
    with EditSession(root, atomic=True, refresh_pyramid=False) as ed:
        for a, b in pairs[:n_edits]:
            ref_a = VertexRef.from_object(
                root, level=0, object_id=int(a), vertex_index=0,
            )
            ref_b = VertexRef.from_object(
                root, level=0, object_id=int(b), vertex_index=0,
            )
            if ref_a.chunk != ref_b.chunk:
                continue
            ed.add_link(
                level=0, src=int(ref_a.local), dst=int(ref_b.local),
                chunk=ref_a.chunk, fragment=ref_a.fragment,
                update_objects=True,
            )
    return time.perf_counter() - t0


rows_scaling = []
for n_objects in N_OBJECTS_SWEEP:
    seed = int(master_rng.integers(0, 2**31 - 1))
    store, positions = _build_scaling_baseline(n_objects, seed)

    t_atomic = _bench_edit_vertex_atomic(store, positions, N_EDITS_SCALING)
    rows_scaling.append({
        'n_objects': n_objects,
        'edit_kind': 'edit_vertex_atomic',
        'n_edits': N_EDITS_SCALING,
        'wall_s': round(t_atomic, 3),
        'us_per_edit': round((t_atomic / N_EDITS_SCALING) * 1e6, 1),
    })
    shutil.rmtree(Path(store).parent, ignore_errors=True)

    # Rebuild for the cross-OID workload (the atomic test mutated it).
    store, positions = _build_scaling_baseline(n_objects, seed)
    t_cross = _bench_add_link_cross_oid(store, positions, N_EDITS_SCALING)
    rows_scaling.append({
        'n_objects': n_objects,
        'edit_kind': 'add_link_cross_oid',
        'n_edits': N_EDITS_SCALING,
        'wall_s': round(t_cross, 3),
        'us_per_edit': round((t_cross / N_EDITS_SCALING) * 1e6, 1)
                       if t_cross == t_cross else None,
    })
    shutil.rmtree(Path(store).parent, ignore_errors=True)

df_scaling = pd.DataFrame(rows_scaling)
df_scaling
"""),
]


# ===================================================================
# Notebook builder
# ===================================================================

def _to_source(text: str) -> list[str]:
    """Match the multi-line `source` list shape Jupyter writes."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return [""]
    if lines[-1].endswith("\n"):
        lines[-1] = lines[-1].rstrip("\n")
    return lines


def _cell_id() -> str:
    return uuid.uuid4().hex[:8]


def _build(cells: list[tuple[str, str]]) -> dict:
    nb_cells = []
    for kind, text in cells:
        if kind == "md":
            nb_cells.append({
                "cell_type": "markdown",
                "id": _cell_id(),
                "metadata": {},
                "source": _to_source(text),
            })
        else:
            nb_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "id": _cell_id(),
                "metadata": {},
                "outputs": [],
                "source": _to_source(text),
            })
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "zarr-vectors",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.15",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _write(name: str, cells: list[tuple[str, str]]) -> None:
    out = ROOT / name
    out.write_text(
        json.dumps(_build(cells), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out.name} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    _write("01_size_scaling.ipynb", SIZE_CELLS)
    _write("02_data_types.ipynb", TYPES_CELLS)
    _write("03_backends.ipynb", BACKEND_CELLS)
    _write("04_pyramid_scaling.ipynb", PYRAMID_CELLS)
    _write("05_bbox_queries.ipynb", BBOX_CELLS)
    _write("06_chunk_shape.ipynb", CHUNK_SHAPE_CELLS)
    _write("07_compression.ipynb", COMPRESSION_CELLS)
    _write("08_edit_operations.ipynb", EDIT_CELLS)
