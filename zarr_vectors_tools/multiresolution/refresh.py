"""Pyramid refresh after edits.

:func:`rebuild_pyramid_from_level` re-runs the existing
:func:`zarr_vectors_tools.multiresolution.coarsen.coarsen_level` engine for
every level *above* ``source_level``, reusing each level's existing
``bin_ratio`` / ``object_sparsity`` / ``chunk_shape`` settings so the
post-refresh pyramid is byte-for-byte equivalent to a from-scratch
``build_pyramid`` call.

Note: ``coarsen_level`` re-opens the store from a path/URL internally
and spawns its own backend session.  When called inside an
:class:`~zarr_vectors.ops.edit.EditSession` the caller must have
already committed pending edits (otherwise the refresh will coarsen
the *pre-edit* state).  :meth:`EditSession.flush` handles this by
issuing a pre-refresh commit on icechunk-backed stores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zarr_vectors.exceptions import EditError

if TYPE_CHECKING:
    from zarr_vectors.building import Group


def rebuild_pyramid_from_level(
    root: Group,
    source_level: int,
) -> list[dict[str, Any]]:
    """Re-coarsen every level above ``source_level`` from scratch.

    Reads each existing target level's metadata (``bin_ratio``,
    ``object_sparsity``, ``chunk_shape``) and re-runs
    :func:`zarr_vectors_tools.multiresolution.coarsen.coarsen_level` with the
    same parameters, replacing the old level data in place.

    Returns the list of per-level coarsening summaries.
    """
    from zarr_vectors.building import (
        commit,
        list_resolution_levels,
        read_level_metadata,
        read_root_metadata,
        remove_resolution_level,
        session_for,
    )
    from zarr_vectors_tools.multiresolution.coarsen import coarsen_level

    levels = list_resolution_levels(root)
    above = [lv for lv in levels if lv > source_level]
    if not above:
        return []

    if source_level not in levels:
        raise EditError(
            f"source_level={source_level} not present in store "
            f"(have {levels})"
        )

    root_meta = read_root_metadata(root)
    base_bin = root_meta.effective_bin_shape
    base_chunk = tuple(root_meta.chunk_shape)
    if not base_bin:
        raise EditError(
            "Cannot refresh pyramid: root metadata is missing "
            "base_bin_shape (no level-0 bins to coarsen against)."
        )

    # Snapshot per-target settings *before* deleting; deleting wipes the
    # group's attrs along with everything else.
    #
    # Both factors handed to coarsen_level are ratios against the SOURCE
    # LEVEL, but the two fields they are recovered from are not:
    # ``bin_ratio`` is relative to level 0 (it is the NGFF scale) and
    # ``chunk_shape`` is absolute.  Reading either as a per-level factor
    # compounds it a second time — a pyramid built with [2, 2, 2] would
    # come back as bins 2x, 8x, 64x the root instead of 2x, 4x, 8x.  So
    # snapshot every level's absolute bin/chunk shape first and divide by
    # the parent's.
    shapes: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = {}
    metas: dict[int, Any] = {}
    for lv in sorted(levels):
        if lv == 0:
            shapes[0] = (tuple(float(b) for b in base_bin), base_chunk)
            continue
        try:
            lm = read_level_metadata(root, lv)
        except Exception as e:
            if lv in above:
                raise EditError(
                    f"Cannot read level metadata for level {lv}: {e}"
                ) from None
            continue
        metas[lv] = lm
        if lm.bin_shape:
            bin_shape = tuple(float(b) for b in lm.bin_shape)
        else:
            # bin_ratio is level-0-relative, so it reconstructs the absolute
            # bin from the root's — which is exactly what is wanted here.
            ratio = lm.bin_ratio or (1,) * len(base_bin)
            bin_shape = tuple(float(b) * float(r) for b, r in zip(base_bin, ratio))
        chunk_shape = (
            tuple(float(c) for c in lm.chunk_shape)
            if lm.chunk_shape is not None else base_chunk
        )
        shapes[lv] = (bin_shape, chunk_shape)

    plan: list[dict[str, Any]] = []
    for lv in sorted(above):
        lm = metas[lv]
        parent = lm.parent_level if lm.parent_level is not None else lv - 1
        parent_bin, parent_chunk = shapes.get(
            parent, (tuple(float(b) for b in base_bin), base_chunk),
        )
        target_bin, target_chunk = shapes[lv]
        coarsen_factor = (
            float(target_bin[0]) / float(parent_bin[0]) if parent_bin[0] else 1.0
        )
        try:
            scale = tuple(
                max(1, int(round(t / p)))
                for t, p in zip(target_chunk, parent_chunk)
            )
        except Exception:
            scale = (1,) * len(base_chunk)
        plan.append({
            "level": lv,
            "coarsen_factor": coarsen_factor,
            "sparsity_factor": (
                1.0 / lm.object_sparsity if lm.object_sparsity else 1.0
            ),
            "chunk_scale_factor": scale,
            "parent_level": parent,
        })

    # Commit pending writes so coarsen_level (which re-opens the store)
    # can see them.  No-op on non-transactional backends.
    if session_for(root) is not None:
        commit(root, "pre-refresh commit")

    url = root.url

    summaries: list[dict[str, Any]] = []
    for entry in plan:
        lv = entry["level"]
        remove_resolution_level(root, lv)
        if session_for(root) is not None:
            commit(root, f"drop level {lv} for refresh")
        summary = coarsen_level(
            url,
            source_level=entry["parent_level"],
            target_level=lv,
            coarsen_factor=entry["coarsen_factor"],
            sparsity_factor=entry["sparsity_factor"],
            chunk_scale_factor=entry["chunk_scale_factor"],
        )
        summaries.append(summary)
    return summaries
