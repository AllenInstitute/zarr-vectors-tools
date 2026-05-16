"""Generate the three benchmark notebooks.

Run once and commit the output:

    python benchmarks/_build.py

Mirrors the layout of zarr-vectors-py's benchmark suite (4-section per
notebook: setup -> build inputs -> sweep -> plot) but focuses on
*loading* and *filtering* zarr-vectors stores versus the canonical
file format for each geometry type.  All write paths happen outside
the timing loops; we benchmark reads only.
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
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())


def _new_store(prefix):
    \"\"\"Fresh tempdir + zarrvectors path.\"\"\"
    return Path(tempfile.mkdtemp(prefix=f'zvbench_{prefix}_')) / 'store.zarrvectors'


def _new_file(prefix, suffix):
    \"\"\"Fresh tempdir + competitor-format file path.\"\"\"
    return Path(tempfile.mkdtemp(prefix=f'compbench_{prefix}_')) / f'data.{suffix}'
"""


STATS_HELPERS = """N_RUNS = 10
T95_DF9 = 2.262  # scipy.stats.t.ppf(0.975, df=9) -- hard-coded to avoid scipy dep


def _mean_ci95(samples):
    \"\"\"(mean, half-width) for a 1-D sample using Student's t, df=n-1.\"\"\"
    arr = np.asarray(samples, dtype=float)
    if arr.size < 2:
        return float(arr.mean()) if arr.size else 0.0, 0.0
    m = arr.mean()
    s = arr.std(ddof=1)
    hw = T95_DF9 * s / np.sqrt(arr.size)
    return float(m), float(hw)


def _repeat(fn, *args, n_runs=None, **kwargs):
    \"\"\"Run fn n_runs times; return (mean_s, ci_hw_s).\"\"\"
    n = n_runs if n_runs is not None else N_RUNS
    samples = []
    for _ in range(n):
        t, _ = _time(fn, *args, **kwargs)
        samples.append(t)
    return _mean_ci95(samples)
"""


# Competitor-format helpers used by notebooks 01 and 03 (point cloud,
# graph, mesh sides).  Kept in one block so all three notebooks expose
# the same write/read API for each format.
COMPETITOR_HELPERS = """# ----- PLY (point cloud) -----------------------------------------------------
def ply_write_points(path, positions):
    from plyfile import PlyData, PlyElement
    n = len(positions)
    arr = np.zeros(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    arr['x'] = positions[:, 0]
    arr['y'] = positions[:, 1]
    arr['z'] = positions[:, 2]
    el = PlyElement.describe(arr, 'vertex')
    PlyData([el], text=False).write(str(path))


def ply_load_full(path):
    from plyfile import PlyData
    ply = PlyData.read(str(path))
    v = ply['vertex']
    return np.stack([v['x'], v['y'], v['z']], axis=1)


def ply_load_bbox(path, low, high):
    pts = ply_load_full(path)
    mask = np.all((pts >= np.asarray(low)) & (pts <= np.asarray(high)), axis=1)
    return pts[mask]


# ----- CSV edge-list (graph) -------------------------------------------------
def csv_write_graph(edges_path, nodes_path, positions, edges):
    pd.DataFrame({
        'node_id': np.arange(len(positions)),
        'x': positions[:, 0],
        'y': positions[:, 1],
        'z': positions[:, 2],
    }).to_csv(nodes_path, index=False)
    pd.DataFrame({
        'source': edges[:, 0],
        'target': edges[:, 1],
    }).to_csv(edges_path, index=False)


def csv_load_graph(edges_path, nodes_path):
    nodes = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    pos = nodes[['x', 'y', 'z']].to_numpy()
    e = edges_df[['source', 'target']].to_numpy()
    return pos, e


def csv_load_graph_bbox(edges_path, nodes_path, low, high):
    pos, e = csv_load_graph(edges_path, nodes_path)
    mask = np.all((pos >= np.asarray(low)) & (pos <= np.asarray(high)), axis=1)
    keep = np.where(mask)[0]
    keep_set = set(int(k) for k in keep)
    edge_mask = np.array([int(s) in keep_set and int(t) in keep_set for s, t in e])
    return pos[mask], e[edge_mask]


# ----- OBJ (mesh) -- pure-Python parser for fair comparison ------------------
def obj_write_mesh(path, vertices, faces):
    with open(path, 'w') as f:
        for v in vertices:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\\n')
        for tri in faces:
            f.write(f'f {tri[0]+1} {tri[1]+1} {tri[2]+1}\\n')


def obj_load_full(path):
    verts = []
    faces = []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith('f '):
                parts = line.split()
                tri = tuple(int(p.split('/')[0]) - 1 for p in parts[1:4])
                faces.append(tri)
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def obj_load_bbox(path, low, high):
    v, f = obj_load_full(path)
    mask = np.all((v >= np.asarray(low)) & (v <= np.asarray(high)), axis=1)
    keep = np.where(mask)[0]
    keep_set = set(int(k) for k in keep)
    face_mask = np.array([
        int(a) in keep_set and int(b) in keep_set and int(c) in keep_set
        for a, b, c in f
    ])
    return v[mask], f[face_mask]
"""


# ===================================================================
# 01 - Size scaling -- log-log over N for three geometry types
# ===================================================================

SIZE_CELLS: list[tuple[str, str]] = [
    ("md", """\
# Size scaling -- loading and filtering

How does the time to **load** and **filter** a zarr-vectors store
compare against the canonical file format for each geometry type as
the dataset grows?  Three panels, one per geometry, log-log axes.

| Panel | Geometry | zarr-vectors | Competitor |
| --- | --- | --- | --- |
| A | point cloud | `read_points` | PLY (`plyfile`) |
| B | graph | `read_graph` | edge-list + node CSV (`pandas`) |
| C | mesh | `read_mesh` | OBJ (pure-Python parser) |

For each `N` we measure two operations:

- **load_full** -- read every vertex / edge / face into memory.
- **load_bbox** -- read only the 1 % volume bounding box at the
  centre of the cube.  zarr-vectors uses the public `bbox=` argument
  to the read function; competitors read the whole file and then
  filter in numpy (no competitor supports a native spatial sub-read).

Each timing is averaged over **`N_RUNS = 10` runs**; plots show the
mean with shaded **95 % CI** band (Student's t, df=9).  Writes happen
once per `N` outside the timing loop -- only reads are benchmarked.

Requires `matplotlib`, `plyfile`, and `pandas` in addition to the
usual deps.  Runtime: ~5 minutes on a laptop (the 100 K mesh case
dominates because the pure-Python OBJ parser is the long pole).
"""),
    ("code", SHARED_HELPERS + "\n\n" + STATS_HELPERS),
    ("md", "## 1 · Setup"),
    ("code", """\
from zarr_vectors.types.points import write_points, read_points
from zarr_vectors.types.graphs import write_graph, read_graph
from zarr_vectors.types.meshes import write_mesh, read_mesh

SIZES = [1_000, 10_000, 100_000]   # vertex count per panel
CHUNK = (200.0, 200.0, 200.0)
BIN   = (50.0, 50.0, 50.0)
SEED  = 0
EXTENT = 1000.0                     # cube side; bbox covers ~1% of volume.
BBOX_HW = EXTENT * (0.01 ** (1/3)) / 2  # half-width of the 1% bbox
BBOX_CENTRE = np.array([EXTENT / 2] * 3)
BBOX_LOW    = BBOX_CENTRE - BBOX_HW
BBOX_HIGH   = BBOX_CENTRE + BBOX_HW
print(f'bbox half-width: {BBOX_HW:.2f}  (~1% volume of [0,{EXTENT}]^3)')
"""),
    ("code", COMPETITOR_HELPERS),
    ("md", "## 2 · Build inputs (one zarr-vectors + one competitor per N, per geometry)"),
    ("code", """\
rng = np.random.default_rng(SEED)


def _gen_points(n):
    return rng.uniform(0, EXTENT, (n, 3)).astype(np.float32)


def _gen_graph(n):
    pos = rng.uniform(0, EXTENT, (n, 3)).astype(np.float32)
    src = rng.integers(0, n, size=3 * n // 2)
    dst = rng.integers(0, n, size=3 * n // 2)
    edges = np.stack([src[src != dst], dst[src != dst]], axis=1).astype(np.int64)
    return pos, edges


def _gen_mesh(n):
    side = max(int(np.sqrt(n)), 2)
    xs, ys = np.meshgrid(np.linspace(0, EXTENT, side), np.linspace(0, EXTENT, side))
    zs = rng.uniform(0, EXTENT * 0.1, (side, side))
    v = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3).astype(np.float32)
    i = np.arange(side - 1); j = np.arange(side - 1)
    ii, jj = np.meshgrid(i, j, indexing='ij')
    a = (ii * side + jj).ravel(); b = a + 1; c = a + side; d = c + 1
    f = np.concatenate([
        np.stack([a, b, c], axis=1),
        np.stack([b, d, c], axis=1),
    ]).astype(np.int64)
    return v, f


# inputs[panel][n] = dict with 'zv' (store path) and 'comp' (file path[s])
inputs = {'points': {}, 'graph': {}, 'mesh': {}}

for n in SIZES:
    # Points
    pts = _gen_points(n)
    zv = _new_store(f'points_{n}')
    write_points(zv, pts, chunk_shape=CHUNK, bin_shape=BIN)
    comp = _new_file(f'points_{n}', 'ply')
    ply_write_points(comp, pts)
    inputs['points'][n] = {'zv': zv, 'comp': comp}

    # Graph
    pos, edges = _gen_graph(n)
    zv = _new_store(f'graph_{n}')
    write_graph(zv, pos, edges, chunk_shape=CHUNK, bin_shape=BIN)
    edges_csv = _new_file(f'graph_edges_{n}', 'csv')
    nodes_csv = edges_csv.parent / 'nodes.csv'
    csv_write_graph(edges_csv, nodes_csv, pos, edges)
    inputs['graph'][n] = {'zv': zv, 'comp': (edges_csv, nodes_csv)}

    # Mesh
    v, f = _gen_mesh(n)
    zv = _new_store(f'mesh_{n}')
    write_mesh(zv, v, f, chunk_shape=CHUNK, bin_shape=BIN)
    comp = _new_file(f'mesh_{n}', 'obj')
    obj_write_mesh(comp, v, f)
    inputs['mesh'][n] = {'zv': zv, 'comp': comp}

print('built', sum(len(v) for v in inputs.values()), 'input stores')
"""),
    ("md", "## 3 · Run the sweep (reads only)"),
    ("code", """\
PANELS = ['points', 'graph', 'mesh']
OPS    = ['load_full', 'load_bbox']

# raw[panel][op][prefix] = array shape (len(SIZES), N_RUNS)
raw = {
    p: {op: {prefix: np.zeros((len(SIZES), N_RUNS)) for prefix in ('zv', 'comp')}
        for op in OPS}
    for p in PANELS
}


def _bench(panel, op, prefix, n):
    inp = inputs[panel][n]
    if panel == 'points':
        if prefix == 'zv':
            if op == 'load_full':
                return _time(read_points, inp['zv'])
            return _time(read_points, inp['zv'],
                         bbox=(BBOX_LOW.tolist(), BBOX_HIGH.tolist()))
        # competitor: PLY
        if op == 'load_full':
            return _time(ply_load_full, inp['comp'])
        return _time(ply_load_bbox, inp['comp'], BBOX_LOW, BBOX_HIGH)

    if panel == 'graph':
        if prefix == 'zv':
            if op == 'load_full':
                return _time(read_graph, inp['zv'])
            return _time(read_graph, inp['zv'],
                         bbox=(BBOX_LOW.tolist(), BBOX_HIGH.tolist()))
        edges_csv, nodes_csv = inp['comp']
        if op == 'load_full':
            return _time(csv_load_graph, edges_csv, nodes_csv)
        return _time(csv_load_graph_bbox, edges_csv, nodes_csv,
                     BBOX_LOW, BBOX_HIGH)

    # mesh
    if prefix == 'zv':
        if op == 'load_full':
            return _time(read_mesh, inp['zv'])
        return _time(read_mesh, inp['zv'],
                     bbox=(BBOX_LOW.tolist(), BBOX_HIGH.tolist()))
    if op == 'load_full':
        return _time(obj_load_full, inp['comp'])
    return _time(obj_load_bbox, inp['comp'], BBOX_LOW, BBOX_HIGH)


for panel in PANELS:
    for i, n in enumerate(SIZES):
        for op in OPS:
            for prefix in ('zv', 'comp'):
                for run in range(N_RUNS):
                    t, _ = _bench(panel, op, prefix, n)
                    raw[panel][op][prefix][i, run] = t

# Tidy long-form df.
rows = []
for panel in PANELS:
    for i, n in enumerate(SIZES):
        for op in OPS:
            for prefix in ('zv', 'comp'):
                mean, hw = _mean_ci95(raw[panel][op][prefix][i])
                rows.append({
                    'panel':  panel,
                    'N':      n,
                    'op':     op,
                    'format': prefix,
                    'mean_s': round(mean, 4),
                    'ci_hw':  round(hw, 4),
                })
df = pd.DataFrame(rows)
"""),
    ("md", "## 4 · Results"),
    ("code", "df"),
    ("md", "## 5 · Plot"),
    ("code", """\
PANEL_TITLES = {
    'points': 'point cloud (zv vs PLY)',
    'graph':  'graph (zv vs CSV edge-list)',
    'mesh':   'mesh (zv vs OBJ)',
}
SERIES = [
    ('zarr-vectors load_full', 'zv',   'load_full', '#1f77b4', '-'),
    ('zarr-vectors load_bbox', 'zv',   'load_bbox', '#1f77b4', '--'),
    ('competitor load_full',   'comp', 'load_full', '#ff7f0e', '-'),
    ('competitor load_bbox',   'comp', 'load_bbox', '#ff7f0e', '--'),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
handles, labels = [], []

for ax, panel in zip(axes, PANELS):
    sub = df[df['panel'] == panel]
    for label, prefix, op, color, ls in SERIES:
        s = sub[(sub['format'] == prefix) & (sub['op'] == op)].sort_values('N')
        x    = s['N'].to_numpy()
        mean = s['mean_s'].to_numpy()
        hw   = s['ci_hw'].to_numpy()
        line, = ax.plot(x, mean, marker='o', color=color, linestyle=ls)
        ax.fill_between(x, mean - hw, mean + hw, color=color, alpha=0.20)
        if ax is axes[0]:
            handles.append(line)
            labels.append(label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('N')
    ax.set_ylabel('s')
    ax.set_title(PANEL_TITLES[panel])
    ax.grid(True, which='both', alpha=0.3)

fig.suptitle(f'Load + filter time vs N -- {N_RUNS} runs, 95% CI')
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.06, 1, 1])
plt.show()
"""),
]


# ===================================================================
# 02 - Data-type comparison -- fixed N, six geometries side by side
# ===================================================================

TYPES_CELLS: list[tuple[str, str]] = [
    ("md", """\
# Data-type comparison -- loading and filtering

At a fixed dataset size (`N = 50_000` vertices / elements), how does
loading and filtering compare across the six ZVF geometry types, each
against its canonical competitor format?

| Geometry | Competitor | Reader | Subset op |
| --- | --- | --- | --- |
| point cloud | PLY | `plyfile.PlyData.read` | bbox |
| line | CSV | `pandas.read_csv` | bbox |
| polyline / streamline | TRX | `trx.trx_file_memmap.load` | object_ids |
| graph | GraphML | `networkx.read_graphml` | bbox |
| skeleton | SWC | text parse | bbox |
| mesh | OBJ | pure-Python parser | bbox |

**TRX is the only competitor with a native partial read** (it is
memory-mapped and supports per-streamline slicing); the `supports_partial`
column in the results table flags this asymmetry.  Every other
competitor materialises the entire file and applies the filter in
numpy, which is the honest comparison -- those formats simply have
no other option.

Each timing is averaged over **`N_RUNS = 10` runs**; bars show the
mean with **95 % CI** error bars (Student's t, df=9).  Sections gated
on optional deps (`networkx`, `trx-python`, `plyfile`) skip
gracefully if the package isn't installed.
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
EXTENT = 1000.0
BBOX_HW = EXTENT * (0.01 ** (1/3)) / 2
BBOX_CENTRE = np.array([EXTENT / 2] * 3)
BBOX_LOW    = (BBOX_CENTRE - BBOX_HW).tolist()
BBOX_HIGH   = (BBOX_CENTRE + BBOX_HW).tolist()
rng = np.random.default_rng(SEED)
"""),
    ("code", COMPETITOR_HELPERS),
    ("md", "## 2 · Per-geometry competitor helpers"),
    ("code", """\
# ----- CSV lines -------------------------------------------------------------
def csv_write_lines(path, endpoints):
    # endpoints: (M, 2, 3)
    flat = endpoints.reshape(len(endpoints), 6)
    cols = ['x0', 'y0', 'z0', 'x1', 'y1', 'z1']
    pd.DataFrame(flat, columns=cols).to_csv(path, index=False)


def csv_load_lines(path):
    df_ = pd.read_csv(path)
    return df_.to_numpy().reshape(-1, 2, 3)


def csv_load_lines_bbox(path, low, high):
    arr = csv_load_lines(path)
    midpoints = arr.mean(axis=1)
    mask = np.all((midpoints >= np.asarray(low)) & (midpoints <= np.asarray(high)), axis=1)
    return arr[mask]


# ----- GraphML ---------------------------------------------------------------
def graphml_write(path, positions, edges):
    import networkx as nx
    g = nx.Graph()
    for i, (x, y, z) in enumerate(positions):
        g.add_node(i, x=float(x), y=float(y), z=float(z))
    for s, t in edges:
        g.add_edge(int(s), int(t))
    nx.write_graphml(g, str(path))


def graphml_load_full(path):
    import networkx as nx
    return nx.read_graphml(str(path))


def graphml_load_bbox(path, low, high):
    import networkx as nx
    g = nx.read_graphml(str(path))
    low_a  = np.asarray(low)
    high_a = np.asarray(high)
    keep = []
    for n, data in g.nodes(data=True):
        p = np.array([float(data.get('x', 0)),
                      float(data.get('y', 0)),
                      float(data.get('z', 0))])
        if np.all((p >= low_a) & (p <= high_a)):
            keep.append(n)
    return g.subgraph(keep)


# ----- SWC -- pure text parse ------------------------------------------------
def swc_write(path, positions, parents):
    \"\"\"Write a SWC tree.  parents[i] = parent index (-1 for root, 0-based).\"\"\"
    with open(path, 'w') as f:
        for i, (p, parent) in enumerate(zip(positions, parents)):
            parent_swc = int(parent) + 1 if parent >= 0 else -1
            f.write(f'{i+1} 0 {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} 1.0 {parent_swc}\\n')


def swc_load_full(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            rows.append([float(p) for p in parts[:7]])
    return np.asarray(rows, dtype=np.float64)


def swc_load_bbox(path, low, high):
    rows = swc_load_full(path)
    pos = rows[:, 2:5]
    mask = np.all((pos >= np.asarray(low)) & (pos <= np.asarray(high)), axis=1)
    return rows[mask]


# ----- TRX (only competitor with native partial read) ------------------------
def trx_write_polylines(path, polylines):
    \"\"\"Build a TRX via nibabel.Tractogram then save with trx-python.\"\"\"
    import nibabel as nib
    from nibabel.streamlines import Tractogram
    from trx.trx_file_memmap import TrxFile, save as trx_save
    ref_img = nib.Nifti1Image(np.zeros((1, 1, 1), dtype=np.uint8), affine=np.eye(4))
    tractogram = Tractogram(streamlines=polylines, affine_to_rasmm=np.eye(4))
    trx_obj = TrxFile.from_tractogram(tractogram, reference=ref_img)
    trx_save(trx_obj, str(path))


def trx_load_full(path):
    from trx.trx_file_memmap import load as trx_load
    obj = trx_load(str(path))
    return [np.asarray(s) for s in obj.streamlines]


def trx_load_objects(path, object_ids):
    \"\"\"Native partial read via memory-mapped slicing.\"\"\"
    from trx.trx_file_memmap import load as trx_load
    obj = trx_load(str(path))
    subset = obj.select(list(object_ids))
    return [np.asarray(s) for s in subset.streamlines]
"""),
    ("md", "## 3 · Build inputs"),
    ("code", """\
# Generate synthetic inputs for each geometry, then write the zarr-vectors
# store and the competitor file.  All writes are outside the timing loop.

# Points
pts = rng.uniform(0, EXTENT, (N, 3)).astype(np.float32)
zv_points  = _new_store('points')
ply_points = _new_file('points', 'ply')
write_points(zv_points, pts, chunk_shape=CHUNK, bin_shape=BIN)
ply_write_points(ply_points, pts)

# Lines
endpoints = rng.uniform(0, EXTENT, (N // 2, 2, 3)).astype(np.float32)
zv_lines  = _new_store('lines')
csv_lines = _new_file('lines', 'csv')
write_lines(zv_lines, endpoints, chunk_shape=CHUNK, bin_shape=BIN)
csv_write_lines(csv_lines, endpoints)

# Polylines
counts = rng.integers(8, 16, size=N // 12)
polylines = []
for c in counts:
    start = rng.uniform(0, EXTENT, 3)
    steps = rng.normal(0, 5, (c, 3))
    polylines.append((start + steps.cumsum(axis=0)).astype(np.float32))
zv_polylines = _new_store('polylines')
write_polylines(zv_polylines, polylines, chunk_shape=CHUNK, bin_shape=BIN)
try:
    trx_polylines = _new_file('polylines', 'trx')
    trx_write_polylines(trx_polylines, polylines)
    trx_available = True
except Exception as e:
    print('TRX unavailable, skipping that competitor:', e)
    trx_polylines = None
    trx_available = False

# Graph
graph_pos = rng.uniform(0, EXTENT, (N, 3)).astype(np.float32)
src = rng.integers(0, N, size=3 * N // 2)
dst = rng.integers(0, N, size=3 * N // 2)
graph_edges = np.stack([src[src != dst], dst[src != dst]], axis=1).astype(np.int64)
zv_graph  = _new_store('graph')
write_graph(zv_graph, graph_pos, graph_edges, chunk_shape=CHUNK, bin_shape=BIN)
try:
    gml_graph = _new_file('graph', 'graphml')
    graphml_write(gml_graph, graph_pos, graph_edges)
    networkx_available = True
except Exception as e:
    print('networkx unavailable, skipping GraphML:', e)
    gml_graph = None
    networkx_available = False

# Skeleton (tree)
skel_pos = rng.uniform(0, EXTENT, (N, 3)).astype(np.float32)
skel_parents = np.concatenate([[-1], rng.integers(0, np.arange(1, N))])
skel_edges = np.stack([
    np.arange(1, N),
    skel_parents[1:],
], axis=1).astype(np.int64)
zv_skel = _new_store('skeleton')
write_graph(zv_skel, skel_pos, skel_edges,
            chunk_shape=CHUNK, bin_shape=BIN, kind='skeleton')
swc_skel = _new_file('skeleton', 'swc')
swc_write(swc_skel, skel_pos, skel_parents)

# Mesh
side = int(np.sqrt(N))
xs, ys = np.meshgrid(np.linspace(0, EXTENT, side), np.linspace(0, EXTENT, side))
zs = rng.uniform(0, EXTENT * 0.1, (side, side))
mesh_v = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3).astype(np.float32)
ii, jj = np.meshgrid(np.arange(side - 1), np.arange(side - 1), indexing='ij')
a = (ii * side + jj).ravel(); b = a + 1; c = a + side; d = c + 1
mesh_f = np.concatenate([np.stack([a, b, c], axis=1),
                         np.stack([b, d, c], axis=1)]).astype(np.int64)
zv_mesh = _new_store('mesh')
obj_mesh = _new_file('mesh', 'obj')
write_mesh(zv_mesh, mesh_v, mesh_f, chunk_shape=CHUNK, bin_shape=BIN)
obj_write_mesh(obj_mesh, mesh_v, mesh_f)

print(f'built inputs for N = {N:,}')
"""),
    ("md", "## 4 · Run the sweep (reads only)"),
    ("code", """\
SUBSET_OBJECT_IDS = list(rng.integers(0, len(polylines), size=max(1, len(polylines) // 100)))


def _run_pair(name, supports_partial, zv_full, zv_subset, comp_full, comp_subset):
    \"\"\"Run a (geometry, competitor) pair through N_RUNS for each op.\"\"\"
    out = []
    for op, zv_fn, comp_fn in [('load_full', zv_full, comp_full),
                                ('load_subset', zv_subset, comp_subset)]:
        for fmt, fn in [('zv', zv_fn), ('comp', comp_fn)]:
            if fn is None:
                out.append({'type': name, 'op': op, 'format': fmt,
                            'mean_s': np.nan, 'ci_hw': np.nan,
                            'supports_partial': supports_partial})
                continue
            mean, hw = _repeat(fn)
            out.append({'type': name, 'op': op, 'format': fmt,
                        'mean_s': round(mean, 4), 'ci_hw': round(hw, 4),
                        'supports_partial': supports_partial})
    return out


rows = []

# Points (PLY)
rows += _run_pair('points', False,
    zv_full=lambda: read_points(zv_points),
    zv_subset=lambda: read_points(zv_points, bbox=(BBOX_LOW, BBOX_HIGH)),
    comp_full=lambda: ply_load_full(ply_points),
    comp_subset=lambda: ply_load_bbox(ply_points, BBOX_LOW, BBOX_HIGH))

# Lines (CSV)
rows += _run_pair('lines', False,
    zv_full=lambda: read_lines(zv_lines),
    zv_subset=lambda: read_lines(zv_lines, bbox=(BBOX_LOW, BBOX_HIGH)),
    comp_full=lambda: csv_load_lines(csv_lines),
    comp_subset=lambda: csv_load_lines_bbox(csv_lines, BBOX_LOW, BBOX_HIGH))

# Polylines (TRX) -- partial=True for the competitor
rows += _run_pair('polylines', trx_available,
    zv_full=lambda: read_polylines(zv_polylines),
    zv_subset=lambda: read_polylines(zv_polylines, object_ids=SUBSET_OBJECT_IDS),
    comp_full=(lambda: trx_load_full(trx_polylines)) if trx_available else None,
    comp_subset=(lambda: trx_load_objects(trx_polylines, SUBSET_OBJECT_IDS)) if trx_available else None)

# Graph (GraphML)
rows += _run_pair('graph', False,
    zv_full=lambda: read_graph(zv_graph),
    zv_subset=lambda: read_graph(zv_graph, bbox=(BBOX_LOW, BBOX_HIGH)),
    comp_full=(lambda: graphml_load_full(gml_graph)) if networkx_available else None,
    comp_subset=(lambda: graphml_load_bbox(gml_graph, BBOX_LOW, BBOX_HIGH)) if networkx_available else None)

# Skeleton (SWC)
rows += _run_pair('skeleton', False,
    zv_full=lambda: read_graph(zv_skel),
    zv_subset=lambda: read_graph(zv_skel, bbox=(BBOX_LOW, BBOX_HIGH)),
    comp_full=lambda: swc_load_full(swc_skel),
    comp_subset=lambda: swc_load_bbox(swc_skel, BBOX_LOW, BBOX_HIGH))

# Mesh (OBJ)
rows += _run_pair('mesh', False,
    zv_full=lambda: read_mesh(zv_mesh),
    zv_subset=lambda: read_mesh(zv_mesh, bbox=(BBOX_LOW, BBOX_HIGH)),
    comp_full=lambda: obj_load_full(obj_mesh),
    comp_subset=lambda: obj_load_bbox(obj_mesh, BBOX_LOW, BBOX_HIGH))

df = pd.DataFrame(rows)
"""),
    ("md", "## 5 · Results"),
    ("code", "df"),
    ("md", "## 6 · Plot"),
    ("code", """\
TYPES = ['points', 'lines', 'polylines', 'graph', 'skeleton', 'mesh']

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=False)
x = np.arange(len(TYPES))
w = 0.36

for ax, op in zip(axes, ('load_full', 'load_subset')):
    sub = df[df['op'] == op]
    zv   = sub[sub['format'] == 'zv'  ].set_index('type').reindex(TYPES)
    comp = sub[sub['format'] == 'comp'].set_index('type').reindex(TYPES)
    ax.bar(x - w/2, zv['mean_s'].fillna(0).to_numpy(),
           width=w, yerr=zv['ci_hw'].fillna(0).to_numpy(),
           capsize=3, color='#1f77b4', label='zarr-vectors')
    ax.bar(x + w/2, comp['mean_s'].fillna(0).to_numpy(),
           width=w, yerr=comp['ci_hw'].fillna(0).to_numpy(),
           capsize=3, color='#ff7f0e', label='competitor')
    ax.set_xticks(x)
    ax.set_xticklabels(TYPES, rotation=20)
    ax.set_ylabel('seconds')
    ax.set_title(op)
    ax.grid(True, axis='y', alpha=0.3)
    # Mark TRX (the only partial-read competitor) with an asterisk.
    for i, t in enumerate(TYPES):
        row = sub[(sub['type'] == t) & (sub['format'] == 'comp')]
        if not row.empty and bool(row['supports_partial'].iloc[0]):
            ax.annotate('*', (i + w/2, comp['mean_s'].fillna(0).iloc[i]),
                        ha='center', va='bottom', fontsize=14, color='#ff7f0e')

axes[0].legend()
fig.suptitle(f'Load + filter per type  --  N = {N:,}, {N_RUNS} runs, 95% CI '
             f'(* = native partial read)')
fig.tight_layout()
plt.show()
"""),
]


# ===================================================================
# 03 - Filtering depth -- log-log over subset fraction
# ===================================================================

FILTERING_CELLS: list[tuple[str, str]] = [
    ("md", """\
# Filtering depth -- log-log over subset fraction

Holds `N = 100_000` fixed and sweeps the **subset fraction** so the
chunked-read benefit is visible directly.  Two panels:

| Panel | Geometry | Filter axis | Competitor |
| --- | --- | --- | --- |
| A | point cloud | bbox covering `f` of the cube volume | PLY (`plyfile`) |
| B | polyline / streamline | `f * N_polylines` random object_ids | TRX (`trx-python`, native partial read) |

`FRACTIONS = [0.001, 0.01, 0.1, 0.5, 1.0]` -- spans three decades on
the x-axis.  Log-log axes; shaded **95 % CI** band from `N_RUNS = 10`
repeats.

For the point-cloud panel zarr-vectors should slope down with `f`
(reading fewer chunks); PLY stays flat (it always parses the whole
file).  For the polyline panel both formats can do native partial
reads, so both should slope down -- the comparison there is "by how
much".
"""),
    ("code", SHARED_HELPERS + "\n\n" + STATS_HELPERS),
    ("md", "## 1 · Setup"),
    ("code", """\
from zarr_vectors.types.points import write_points, read_points
from zarr_vectors.types.polylines import write_polylines, read_polylines

N = 100_000
CHUNK = (200.0, 200.0, 200.0)
BIN   = (50.0, 50.0, 50.0)
SEED  = 0
EXTENT = 1000.0
FRACTIONS = [0.001, 0.01, 0.1, 0.5, 1.0]
rng = np.random.default_rng(SEED)


def _bbox_for_fraction(f):
    \"\"\"Centred cube bbox covering fraction f of the EXTENT^3 volume.\"\"\"
    hw = EXTENT * (f ** (1 / 3)) / 2
    centre = np.array([EXTENT / 2] * 3)
    return (centre - hw).tolist(), (centre + hw).tolist()
"""),
    ("code", COMPETITOR_HELPERS),
    ("code", """\
# TRX helpers (subset by object_ids).  Same shape as notebook 02.
def trx_write_polylines(path, polylines):
    import nibabel as nib
    from nibabel.streamlines import Tractogram
    from trx.trx_file_memmap import TrxFile, save as trx_save
    ref_img = nib.Nifti1Image(np.zeros((1, 1, 1), dtype=np.uint8), affine=np.eye(4))
    tractogram = Tractogram(streamlines=polylines, affine_to_rasmm=np.eye(4))
    trx_obj = TrxFile.from_tractogram(tractogram, reference=ref_img)
    trx_save(trx_obj, str(path))


def trx_load_objects(path, object_ids):
    from trx.trx_file_memmap import load as trx_load
    obj = trx_load(str(path))
    subset = obj.select(list(object_ids))
    return [np.asarray(s) for s in subset.streamlines]
"""),
    ("md", "## 2 · Build inputs"),
    ("code", """\
# Points
pts = rng.uniform(0, EXTENT, (N, 3)).astype(np.float32)
zv_points  = _new_store('points')
ply_points = _new_file('points', 'ply')
write_points(zv_points, pts, chunk_shape=CHUNK, bin_shape=BIN)
ply_write_points(ply_points, pts)

# Polylines -- ~N total vertices spread across short walks.
counts = rng.integers(8, 16, size=N // 12)
polylines = []
for c in counts:
    start = rng.uniform(0, EXTENT, 3)
    steps = rng.normal(0, 5, (c, 3))
    polylines.append((start + steps.cumsum(axis=0)).astype(np.float32))
zv_polylines = _new_store('polylines')
write_polylines(zv_polylines, polylines, chunk_shape=CHUNK, bin_shape=BIN)
try:
    trx_polylines = _new_file('polylines', 'trx')
    trx_write_polylines(trx_polylines, polylines)
    trx_available = True
except Exception as e:
    print('TRX unavailable, polyline panel will be ZV-only:', e)
    trx_polylines = None
    trx_available = False

print(f'built: {len(pts):,} points, {len(polylines):,} polylines')
"""),
    ("md", "## 3 · Run the sweep"),
    ("code", """\
n_poly = len(polylines)

# Pre-pick object id subsets per fraction so the same selection is
# used across all N_RUNS repeats.
poly_subsets = {
    f: list(rng.integers(0, n_poly, size=max(1, int(round(n_poly * f)))))
    for f in FRACTIONS
}

raw = {
    'points': {'zv': np.zeros((len(FRACTIONS), N_RUNS)),
               'comp': np.zeros((len(FRACTIONS), N_RUNS))},
    'polylines': {'zv': np.zeros((len(FRACTIONS), N_RUNS)),
                  'comp': np.zeros((len(FRACTIONS), N_RUNS))},
}

for i, f in enumerate(FRACTIONS):
    low, high = _bbox_for_fraction(f)
    ids = poly_subsets[f]

    # Points
    for run in range(N_RUNS):
        t_zv,   _ = _time(read_points, zv_points,
                          bbox=(low, high) if f < 1.0 else None)
        t_comp, _ = _time(
            ply_load_bbox if f < 1.0 else ply_load_full,
            ply_points,
            *((low, high) if f < 1.0 else ()),
        )
        raw['points']['zv'][i, run]   = t_zv
        raw['points']['comp'][i, run] = t_comp

    # Polylines
    for run in range(N_RUNS):
        t_zv, _ = _time(read_polylines, zv_polylines,
                        object_ids=None if f >= 1.0 else ids)
        if trx_available:
            from trx.trx_file_memmap import load as trx_load
            if f >= 1.0:
                t_comp, _ = _time(lambda: [np.asarray(s) for s in trx_load(str(trx_polylines)).streamlines])
            else:
                t_comp, _ = _time(trx_load_objects, trx_polylines, ids)
        else:
            t_comp = np.nan
        raw['polylines']['zv'][i, run]   = t_zv
        raw['polylines']['comp'][i, run] = t_comp

# Tidy long-form df.
rows = []
for panel in raw:
    for fmt in ('zv', 'comp'):
        for i, f in enumerate(FRACTIONS):
            samples = raw[panel][fmt][i]
            if np.any(np.isnan(samples)):
                mean, hw = np.nan, np.nan
            else:
                mean, hw = _mean_ci95(samples)
            rows.append({
                'panel':  panel,
                'fraction': f,
                'format': fmt,
                'mean_s': round(mean, 5) if mean == mean else None,
                'ci_hw':  round(hw, 5) if hw == hw else None,
            })

df = pd.DataFrame(rows)
"""),
    ("md", "## 4 · Results"),
    ("code", "df"),
    ("md", "## 5 · Plot"),
    ("code", """\
PANELS = [('points', 'point cloud (zv vs PLY)'),
          ('polylines', 'polylines (zv vs TRX)')]
SERIES = [('zarr-vectors', 'zv',   '#1f77b4'),
          ('competitor',   'comp', '#ff7f0e')]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
handles, labels = [], []

for ax, (panel, title) in zip(axes, PANELS):
    sub = df[df['panel'] == panel]
    for label, fmt, color in SERIES:
        s = sub[sub['format'] == fmt].sort_values('fraction').dropna(subset=['mean_s'])
        if s.empty:
            continue
        x    = s['fraction'].to_numpy()
        mean = s['mean_s'].to_numpy()
        hw   = s['ci_hw'].to_numpy()
        line, = ax.plot(x, mean, marker='o', color=color)
        ax.fill_between(x, mean - hw, mean + hw, color=color, alpha=0.20)
        if ax is axes[0]:
            handles.append(line)
            labels.append(label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('subset fraction')
    ax.set_ylabel('s')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)

fig.suptitle(f'Filter time vs subset fraction  --  N = {N:,}, {N_RUNS} runs, 95% CI')
fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.06, 1, 1])
plt.show()
"""),
]


# ===================================================================
# Notebook builder (identical shape to the parent)
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
                "display_name": "zarr-vectors-tools",
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
    _write("03_filtering.ipynb", FILTERING_CELLS)
