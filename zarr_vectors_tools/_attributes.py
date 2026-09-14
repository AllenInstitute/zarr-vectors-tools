"""Persist a level-wide per-vertex attribute computed by an algorithm.

The algorithms in :mod:`zarr_vectors_tools.algorithms` return one value
per vertex in *level order* — the concatenation of every chunk's
fragments, in the order :func:`chunk_local_to_global_offsets` defines.
Writing that back means cutting the flat array along chunk *and* fragment
boundaries, because the on-disk attribute is per-chunk and ragged in the
same way ``vertices`` is.

This used to go through ``open_zv(...)[level].writer()``, which core has
now deprecated in favour of ``zarr_vectors.open()``.  The data-oriented
``Dataset`` has no "add a per-vertex attribute to an existing level"
operation, though — ``EditPlan`` covers per-vertex *edits*, not a bulk
column append — so the honest replacement for a package that builds
stores is the builder surface, which is what this does.  Retire it if
``Dataset`` ever grows the operation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from zarr_vectors.building import (
    VERTEX_ATTRIBUTES,
    chunk_local_to_global_offsets,
    create_attribute_array,
    read_chunk_vertices,
    rebuild_presence,
    update_level_metadata,
    write_chunk_attributes,
)
from zarr_vectors.exceptions import ArrayError

if TYPE_CHECKING:
    from zarr_vectors.building import Group

__all__ = ["write_vertex_attribute"]


def write_vertex_attribute(
    level_group: Group,
    name: str,
    values: npt.NDArray,
    *,
    dtype: str | np.dtype | None = None,
    channel_names: list[str] | None = None,
    ndim: int = 3,
) -> None:
    """Write ``values`` as ``vertex_attributes/<name>`` across the level.

    Args:
        level_group: An open handle for one resolution level.
        name: Attribute name.
        values: ``(V,)`` or ``(V, C)``, where ``V`` is the level's total
            vertex count, in the same order the algorithms produce.
        dtype: Storage dtype; defaults to ``values``' own.
        channel_names: Optional names for a multi-column attribute.  No
            longer load-bearing: the width is stamped as ``row_shape``, so
            an unnamed multi-column attribute records its own shape.  This
            used to synthesise ``<name>_0..N`` purely so the count could be
            recovered by counting them.
        ndim: Spatial rank used to decode fragment sizes from the
            ``vertices`` blobs.  It must match what
            ``chunk_local_to_global_offsets`` assumed when it built the
            offsets table above — core hardcodes 3 there ("ndim is not
            stored"), so anything else here misaligns the two.

    Raises:
        ArrayError: If ``values`` is scalar, or its length does not match
            the level's vertex count — a silent partial write here would
            misalign every attribute after the first short chunk.
    """
    arr = np.asarray(values)
    if dtype is not None:
        arr = arr.astype(np.dtype(dtype), copy=False)
    if arr.ndim < 1:
        raise ArrayError(
            f"attribute values must be at least 1D; got shape {arr.shape}"
        )

    offsets, chunk_keys, total = chunk_local_to_global_offsets(level_group)
    if arr.shape[0] != total:
        raise ArrayError(
            f"write_vertex_attribute({name!r}): values length "
            f"{arr.shape[0]} != level vertex count {total}"
        )

    ncols = int(arr.shape[1]) if arr.ndim > 1 else 1

    # Allocate once, before any cell write: the write path does not create
    # the container implicitly.  ``ncols`` declares the width directly now;
    # channel names are for humans.
    create_attribute_array(
        level_group, name, dtype=str(arr.dtype), ncols=ncols,
        channel_names=channel_names,
    )

    wrote_any = False
    for cc in chunk_keys:
        # dtype=None reads the dtype the store declares, which is also what
        # chunk_local_to_global_offsets used for its byte arithmetic.  The
        # lazy writer this replaces passed float32 unconditionally, so on a
        # float64 store the two disagreed on the row count and every
        # fragment after the first landed at the wrong offset -- silently,
        # since nothing in the blob records its element type.
        groups = read_chunk_vertices(level_group, cc, None, ndim)
        sizes = [len(g) for g in groups]
        chunk_total = sum(sizes)
        if chunk_total == 0:
            continue
        start = offsets[cc]
        chunk_values = arr[start:start + chunk_total]
        attr_groups: list[npt.NDArray] = []
        cursor = 0
        for s in sizes:
            attr_groups.append(chunk_values[cursor:cursor + s])
            cursor += s
        # record_presence=False + one rebuild afterwards: nonempty_chunks
        # is one array-wide attribute, so stamping it per cell is a
        # read-modify-write of shared state and rewrites the same
        # zarr.json once per chunk even single-threaded.
        write_chunk_attributes(
            level_group, name, cc, attr_groups, arr.dtype,
            record_presence=False,
        )
        wrote_any = True

    if wrote_any:
        rebuild_presence(level_group, f"{VERTEX_ATTRIBUTES}/{name}")
        # The level must advertise the family or readers, which gate on
        # arrays_present, do not see the array at all.
        update_level_metadata(level_group, add_arrays_present=VERTEX_ATTRIBUTES)
