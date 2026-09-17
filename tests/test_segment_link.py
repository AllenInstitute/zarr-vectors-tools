"""One segment, found in a skeleton store, a mesh store and a synapse store.

Three stores share segment ids and nothing else.  The skeleton store holds
neurons A and B with a coarser level that sparsity thinned to one of them;
the mesh store holds A, B and C, in its own numbering; the synapse store is
joined to the skeletons.  What has to hold: a segment resolves to its own
object in each store that holds it and to nothing in one that does not; an
object picked in one store names the matching object in the others; a
segment sparsity dropped from a level is absent there; and each store's
attributes for the segment come back together.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from zarr_vectors.building import get_resolution_level, open_store, read_object_attributes
from zarr_vectors.types.meshes import write_mesh

from zarr_vectors_tools.algorithms.segment_link import SegmentLink, store_segment_ids

A, B, C = 720575940000000011, 720575940000000029, 720575940000000031
STRANGER = 720575940000009999


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    from zarr_vectors_tools.convert.ingest.precomputed_skeletons import (
        InMemoryFragsReader,
        SkeletonInfo,
        run_ingest,
    )
    from zarr_vectors_tools.convert.ingest.synapses import ingest_synapses

    root = tmp_path_factory.mktemp("link")
    info = SkeletonInfo(base_url="mem://x", resolution_nm=(32.0, 32.0, 40.0),
                        chunk_size_nm=(16384.0, 16384.0, 20480.0))
    key = "17910-18422_10448-10960_3088-3600.frags"
    origin = np.array([17910 * 32.0, 10448 * 32.0, 3088 * 40.0])
    segments = {}
    for i, (segment, n) in enumerate(((A, 12), (B, 4))):
        vertices = (origin + 500 + np.column_stack([
            np.arange(n) * 90.0, np.full(n, 300.0 * (i + 1)), np.full(n, 200.0),
        ])).astype(np.float32)
        segments[segment] = {"vertices": vertices,
                             "edges": np.array([[j, j - 1] for j in range(1, n)])}
    skeletons = root / "skeletons.zv"
    run_ingest(
        InMemoryFragsReader(info, {key: segments}), str(skeletons), [key],
        bounds_nm=(origin.tolist(), (origin + np.array(info.chunk_size_nm)).tolist()),
        strides=[2], chunk_scale_factors=[2], sparsity_factors=[2.0], progress=False,
    )

    # The mesh store numbers C first, so its object ids differ from the skeletons'.
    meshes = root / "meshes.zv"
    cube = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], np.float32)
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    write_mesh(
        str(meshes), np.concatenate([cube * 10 + 20 * i for i in range(3)]),
        np.concatenate([faces + 8 * i for i in range(3)]), chunk_shape=(64.0, 64.0, 64.0),
        object_ids=np.repeat(np.arange(3), 8),
        object_attributes={
            "segment_id": np.array([C, B, A], dtype=np.uint64),
            "volume_nm3": np.array([3.0, 2.0, 1.0]),
        },
    )

    table = root / "synapses.csv"
    pd.DataFrame({
        "id": [1, 2, 3],
        "pre_pt_root_id": [B, A, B],
        "post_pt_root_id": [A, A, STRANGER],
        "ctr_pt_position": ["[18000 10500 3100]"] * 3,
    }).to_csv(table, index=False)
    synapses = root / "synapses.zv"
    ingest_synapses(table, skeletons, synapses, resolution=(32, 32, 40))
    return {"skeletons": skeletons, "meshes": meshes, "synapses": synapses}


def _level0_oid(store: Path, segment: int) -> int:
    ids = store_segment_ids(store)
    return int(np.flatnonzero(ids == segment)[0])


def test_a_segment_resolves_to_its_object_in_each_store(stores) -> None:
    link = SegmentLink(stores)
    found = link.resolve(A)
    assert found == {
        "skeletons": _level0_oid(stores["skeletons"], A),
        "meshes": 2,
        "synapses": _level0_oid(stores["skeletons"], A),
    }
    # B owns no post-synaptic site, so the synapse store holds no geometry for it.
    assert link.resolve(B) == {"skeletons": _level0_oid(stores["skeletons"], B),
                               "meshes": 1, "synapses": None}
    assert link.resolve(C) == {"skeletons": None, "meshes": 0, "synapses": None}
    assert link.resolve(STRANGER) == {"skeletons": None, "meshes": None, "synapses": None}


def test_many_segments_make_a_table(stores) -> None:
    table = SegmentLink([stores["skeletons"], stores["meshes"]]).resolve_many([C, A, STRANGER])
    assert list(table.columns) == ["skeletons", "meshes"]
    assert table.index.tolist() == [C, A, STRANGER]
    assert table.loc[C, "meshes"] == 0 and pd.isna(table.loc[C, "skeletons"])
    assert table.loc[A, "meshes"] == 2
    assert table.loc[STRANGER].isna().all()


def test_a_picked_object_names_its_match_elsewhere(stores) -> None:
    link = SegmentLink(stores)
    assert link.segment_of("meshes", 1) == B
    assert link.link("meshes", 1)["skeletons"] == _level0_oid(stores["skeletons"], B)
    assert link.link("meshes", 99) == {"skeletons": None, "meshes": None, "synapses": None}
    # The synapse store's unassigned object has segment 0, which is nobody.
    unassigned = len(store_segment_ids(stores["synapses"])) - 1
    assert link.segment_of("synapses", unassigned) is None
    with pytest.raises(KeyError, match="no store named"):
        link.segment_of("volumes", 0)


def test_sparsity_removes_a_segment_from_the_coarse_level(stores) -> None:
    link = SegmentLink({"skeletons": stores["skeletons"]}, level=1)
    kept = [s for s in (A, B) if link.resolve(s)["skeletons"] is not None]
    assert len(kept) == 1
    # Object ids do not move between levels; only presence does.
    assert link.resolve(kept[0])["skeletons"] == _level0_oid(stores["skeletons"], kept[0])
    assert SegmentLink({"skeletons": stores["skeletons"]}).resolve(
        ({A, B} - set(kept)).pop())["skeletons"] is not None


def test_attributes_come_back_together(stores) -> None:
    got = SegmentLink(stores).attributes(A)
    assert got["meshes"] == {"volume_nm3": 1.0}
    assert got["skeletons"]["synapse_post_count"] == 2
    assert got["skeletons"]["synapse_pre_count"] == 1
    assert "segment_id" not in got["skeletons"]

    chosen = SegmentLink(stores).attributes(C, {"meshes": ["volume_nm3"]})
    assert chosen == {"meshes": {"volume_nm3": 3.0}}
    with pytest.raises(KeyError, match="no object attribute 'nope'"):
        SegmentLink(stores).attributes(A, {"meshes": ["nope"]})


def test_the_synapse_store_carries_the_skeleton_segment_ids(stores) -> None:
    ids = store_segment_ids(stores["synapses"])
    skeleton_ids = store_segment_ids(stores["skeletons"])
    assert ids[:-1].tolist() == skeleton_ids[: len(ids) - 1].tolist()
    assert ids[-1] == 0


def test_a_store_without_segment_ids_is_refused(tmp_path: Path) -> None:
    from zarr_vectors.types.points import write_points

    store = tmp_path / "points.zv"
    write_points(str(store), np.ones((3, 3), np.float32), chunk_shape=(10.0, 10.0, 10.0))
    with pytest.raises(ValueError, match="no object_attributes/segment_id"):
        SegmentLink([store]).resolve(A)
    with pytest.raises(ValueError, match="both named"):
        SegmentLink([tmp_path / "a" / "x.zv", tmp_path / "b" / "x.zv"])


def test_level_zero_ids_answer_for_a_level_without_its_own(stores, tmp_path: Path) -> None:
    from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

    level0 = get_resolution_level(open_store(str(stores["meshes"])), 0)
    assert read_object_attributes(level0, "segment_id").tolist() == [C, B, A]
    copy = tmp_path / "meshes.zv"
    shutil.copytree(stores["meshes"], copy)
    build_pyramid(str(copy), factors=[(2.0, 1.0)])
    assert SegmentLink([copy], level=1).resolve(A) == {"meshes": 2}
