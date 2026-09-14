"""``coarsen_factor`` is a ratio against the level below, and it compounds.

Core changed this: the factor used to multiply the ROOT's bin, so
``[(2, 1), (2, 1), (2, 1)]`` produced three levels all binned at 2x. It now
multiplies the SOURCE LEVEL's bin, so the same factors give 2x, 4x, 8x.
Two fields record the result, in two different reference frames, and mixing
them up is silent:

* ``bin_shape`` — absolute, so it must compound down the pyramid.
* ``bin_ratio`` — relative to LEVEL 0, because it is what becomes the NGFF
  ``scale`` transform. It is *not* the per-level factor once factors vary.

Nothing pinned either, which is how the polyline coarsener kept the old
root-relative arithmetic while the per-object and per-fragment ones moved,
and how ``rebuild_pyramid_from_level`` came to feed a level-0-relative
``bin_ratio`` back in as a per-level ``coarsen_factor`` — squaring it.
"""

from __future__ import annotations

import numpy as np
import pytest
from zarr_vectors.building import (
    open_store,
    read_level_metadata,
    read_root_metadata,
)

from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

#: Two levels of 2x on top of level 0.
FACTORS = [(2.0, 1.0), (2.0, 1.0)]


def _polyline_store(tmp_path, name):
    """A streamline store: routes to the ``polyline`` coarsener.

    Via the segment_id-stamping helper — that coarsener refuses a source
    level without ``fragment_attributes/segment_id``.
    """
    from tests._source_helpers import write_polylines_with_segment_id

    store = tmp_path / name
    rng = np.random.default_rng(7)
    polys = []
    for _ in range(12):
        start = rng.uniform(20, 180, size=(1, 3)).astype(np.float32)
        steps = rng.normal(0, 3.0, size=(29, 3)).astype(np.float32)
        polys.append(
            np.clip(np.concatenate([start, start + np.cumsum(steps, axis=0)]),
                    0, 199).astype(np.float32)
        )
    write_polylines_with_segment_id(
        store, polys, chunk_shape=(200.0, 200.0, 200.0),
    )
    return store


def _point_store(tmp_path, name):
    """A plain point store: routes to the ``per_object`` coarsener."""
    from zarr_vectors.types.points import write_points

    store = tmp_path / name
    rng = np.random.default_rng(11)
    write_points(
        str(store),
        rng.uniform(0, 190, size=(600, 3)).astype(np.float32),
        chunk_shape=(200.0, 200.0, 200.0),
    )
    return store


def _bins(store):
    """``{level: (bin_shape, bin_ratio)}`` plus the root's effective bin."""
    root = open_store(str(store), mode="r")
    root_bin = tuple(float(b) for b in read_root_metadata(root).effective_bin_shape)
    out = {}
    for lv in (1, 2):
        lm = read_level_metadata(root, lv)
        out[lv] = (
            tuple(float(b) for b in lm.bin_shape) if lm.bin_shape else None,
            tuple(int(r) for r in lm.bin_ratio) if lm.bin_ratio else None,
        )
    return root_bin, out


@pytest.mark.parametrize("make_store", [_polyline_store, _point_store],
                         ids=["polyline", "per_object"])
def test_bin_shape_compounds_and_bin_ratio_stays_level_0_relative(
    tmp_path, make_store,
):
    store = make_store(tmp_path, "compound.zarrvectors")
    build_pyramid(str(store), factors=FACTORS, cross_level_storage="none")

    root_bin, levels = _bins(store)

    # bin_shape is absolute, so 2x then 2x again is 4x the root.
    assert levels[1][0] == pytest.approx(tuple(b * 2 for b in root_bin))
    assert levels[2][0] == pytest.approx(tuple(b * 4 for b in root_bin))

    # bin_ratio is the level-0-relative fold change -- the NGFF scale.
    # If it echoed coarsen_factor it would read (2, 2, 2) at BOTH levels.
    assert levels[1][1] == (2, 2, 2)
    assert levels[2][1] == (4, 4, 4)


def test_refresh_reproduces_the_pyramid_it_rebuilt(tmp_path):
    """``rebuild_pyramid_from_level`` must round-trip, not re-square.

    It recovers each level's ``coarsen_factor`` from stored metadata. Those
    fields are level-0-relative; the factor is parent-relative. Reading one
    as the other gives 2x, 8x, 64x instead of 2x, 4x, 8x -- and it compounds
    again on every subsequent refresh.
    """
    from zarr_vectors_tools.multiresolution.refresh import rebuild_pyramid_from_level

    store = _point_store(tmp_path, "refresh.zarrvectors")
    build_pyramid(str(store), factors=FACTORS, cross_level_storage="none")
    before = _bins(store)

    rebuild_pyramid_from_level(open_store(str(store), mode="r+"), 0)

    assert _bins(store) == before
