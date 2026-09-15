"""Pin the batched fast paths of the chunk-local coarseners to the code they replaced.

- ``_decimate_components`` must reproduce ``decimate_skeleton`` per component,
  exactly (positions, edge rows and order, attributes for every aggregation,
  ``kept_source_indices``).
- ``_anchor_join_table`` + ``_cross_edge_shard``'s NumPy join must produce the
  records the old per-chunk dict keymaps did, in the same order.
- The Phase A link-cell reader must decode the same records
  ``read_links_for_tuple`` returns, without per-cell metadata reads.
"""
from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors_tools.multiresolution.skeleton_graph import split_components
from zarr_vectors_tools.multiresolution.strategies import skeletons as sk


def _random_components(rng, n_objects, *, attr_dtype=np.float32, two_col=False):
    comps = []
    forced_pairs = []
    for _ in range(n_objects):
        n = int(rng.integers(1, 40))
        pos = rng.normal(0, 50, size=(n, 3)).astype(np.float32)
        # A random graph over the vertices: chain + a few extra edges, so
        # split_components yields several BFS trees with branches.
        edges = [(i, i - 1) for i in range(1, n) if rng.random() < 0.9]
        edges += [
            (int(rng.integers(0, n)), int(rng.integers(0, n)))
            for _ in range(int(rng.integers(0, 4)))
        ]
        e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        shape = (n, 2) if two_col else (n,)
        if np.issubdtype(attr_dtype, np.integer):
            radius = rng.integers(-5, 50, size=shape).astype(attr_dtype)
        else:
            radius = rng.normal(0, 10, size=shape).astype(attr_dtype)
            radius[rng.random(shape) < 0.05] = -0.0
        pieces = split_components(
            pos, e, {"radius": radius}, vertex_ids=np.arange(n),
        )
        forced = set(rng.choice(n, size=int(rng.integers(0, 4)), replace=True).tolist())
        for comp in pieces:
            comps.append(comp)
            if not forced:
                forced_pairs.append(None)
                continue
            local = {int(v): i for i, v in enumerate(comp["vertex_ids"].tolist())}
            forced_pairs.append([(mv, local[mv]) for mv in forced if mv in local])
    return comps, forced_pairs


def _assert_same(a, b):
    assert a["positions"].dtype == b["positions"].dtype
    assert a["positions"].tobytes() == b["positions"].tobytes()
    assert a["edges"].dtype == b["edges"].dtype
    assert a["edges"].shape == b["edges"].shape
    assert a["edges"].tobytes() == b["edges"].tobytes()
    assert a["kept_source_indices"].dtype == b["kept_source_indices"].dtype
    assert np.array_equal(a["kept_source_indices"], b["kept_source_indices"])
    assert list(a["attributes"]) == list(b["attributes"])
    for name in a["attributes"]:
        x, y = a["attributes"][name], b["attributes"][name]
        assert x.dtype == y.dtype and x.shape == y.shape
        assert x.tobytes() == y.tobytes()


@pytest.mark.parametrize("attr_agg", ["max", "min", "first", "mean"])
@pytest.mark.parametrize("stride", [1, 2, 3, 8])
@pytest.mark.parametrize(
    ("attr_dtype", "two_col"),
    [(np.float32, False), (np.float64, True), (np.int32, False), (np.uint8, True)],
)
def test_decimate_components_matches_decimate_skeleton(attr_agg, stride, attr_dtype, two_col):
    rng = np.random.default_rng(stride * 31 + len(attr_agg))
    comps, forced_pairs = _random_components(
        rng, 25, attr_dtype=attr_dtype, two_col=two_col,
    )
    got = sk._decimate_components(comps, forced_pairs, stride=stride, attr_agg=attr_agg)
    assert len(got) == len(comps)
    for comp, pairs, g in zip(comps, forced_pairs, got):
        ref = sk.decimate_skeleton(
            comp["positions"], comp["edges"], stride=stride,
            forced_keep=None if pairs is None else [ci for _mv, ci in pairs],
            attributes=comp["attributes"], attr_agg=attr_agg,
        )
        _assert_same(g, ref)


def test_decimate_components_mixed_signatures_and_no_attributes():
    rng = np.random.default_rng(7)
    a, fa = _random_components(rng, 6, attr_dtype=np.float32)
    b, fb = _random_components(rng, 6, attr_dtype=np.float64)
    c, fc = _random_components(rng, 4, attr_dtype=np.float32)
    for comp in c:
        comp["attributes"] = {}
    comps = [x for trio in zip(a, b) for x in trio] + c
    forced = [x for trio in zip(fa, fb) for x in trio] + fc
    got = sk._decimate_components(comps, forced, stride=3, attr_agg="mean")
    for comp, pairs, g in zip(comps, forced, got):
        ref = sk.decimate_skeleton(
            comp["positions"], comp["edges"], stride=3,
            forced_keep=None if pairs is None else [ci for _mv, ci in pairs],
            attributes=comp["attributes"], attr_agg="mean",
        )
        _assert_same(g, ref)


def _dict_join_reference(payload, ndim):
    """The per-chunk dict keymap join ``_cross_edge_shard`` used before."""
    arrays = {tuple(e["tcc"]): np.asarray(e["arr"], np.int64) for e in payload["anchors"]}

    def keymap(cc):
        km = {}
        arr = arrays.get(cc)
        if arr is not None and len(arr):
            for row in arr:
                key = (int(row[0]), tuple(int(x) for x in row[1:1 + ndim]))
                if key not in km:
                    km[key] = int(row[1 + ndim])
        return km

    links = []
    for A_, B_ in payload["pairs"]:
        A, B = tuple(A_), tuple(B_)
        amap, bmap = keymap(A), keymap(B)
        if not amap or not bmap:
            continue
        if len(amap) <= len(bmap):
            for k, viA in amap.items():
                if k in bmap:
                    links.append([(A, viA), (B, bmap[k])])
        else:
            for k, viB in bmap.items():
                if k in amap:
                    links.append([(A, amap[k]), (B, viB)])
    return links


def test_cross_edge_join_matches_dict_keymaps(monkeypatch):
    import zarr_vectors.building as building

    written = []
    monkeypatch.setattr(
        building, "write_link_cells", lambda lg, links, *a, **k: written.append(links),
    )
    monkeypatch.setattr(building, "open_store", lambda *a, **k: None)
    monkeypatch.setattr(building, "get_resolution_level", lambda *a, **k: None)
    rng = np.random.default_rng(3)
    chunks = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
    for _ in range(200):
        anchors = []
        for c in chunks:
            n = int(rng.integers(0, 15))
            if n == 0 and rng.random() < 0.5:
                continue
            rows = np.column_stack([
                rng.integers(0, 3, n), rng.integers(-2, 2, (n, 3)), rng.integers(0, 60, n),
            ]).astype(np.int64)
            anchors.append({"tcc": list(c), "arr": rows})
        pairs = [
            (list(chunks[i]), list(chunks[j]))
            for i in range(4) for j in range(4) if i != j and rng.random() < 0.5
        ]
        payload = {"pairs": pairs, "anchors": anchors}
        written.clear()
        res = sk._cross_edge_shard(payload, shared={"ndim": 3, "store_path": "", "target_level": 1})
        ref = _dict_join_reference(payload, 3)
        got = written[0] if written else []
        assert res["n_links"] == len(ref)
        assert got == ref


@pytest.mark.parametrize("directed", [False, True])
def test_link_cell_records_match_read_links_for_tuple(tmp_path, directed):
    from zarr_vectors.building import (
        create_store,
        get_resolution_level,
        read_links_for_tuple,
        write_links,
    )

    from zarr_vectors_tools.algorithms._links import (
        link_cell_prefetch_plan,
        link_cell_reads,
        list_link_cells,
        read_link_cell_records,
    )

    root = create_store(
        str(tmp_path / "store.zv"),
        bounds=([0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]),
        chunk_shape=(100.0, 100.0, 100.0),
        geometry_types=["graph"],
        ndim=3,
    )
    lg = get_resolution_level(root, 0)
    rng = np.random.default_rng(11)
    chunks = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1), (2, 0, 0)]
    links = []
    for _ in range(120):
        a, b = rng.choice(len(chunks), size=2, replace=True)
        links.append(((chunks[a], int(rng.integers(0, 50))), (chunks[b], int(rng.integers(0, 50)))))
    write_links(lg, links, sid_ndim=3, directed=directed)

    cells, segments, spec = link_cell_reads(lg, delta=0)
    assert cells == list_link_cells(lg, delta=0)
    assert spec["intra_path"] is not None
    assert any(a["has_perm"] for a in spec["arrays"]) == (not directed)
    with lg.cached_nodes(), lg.batched_reads(link_cell_prefetch_plan(cells, segments, spec)):
        for cell, seg in zip(cells, segments):
            assert read_link_cell_records(lg, cell, seg, spec) == read_links_for_tuple(
                lg, cell, delta=0,
            )
