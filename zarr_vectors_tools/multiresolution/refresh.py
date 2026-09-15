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

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from zarr_vectors.exceptions import EditError

if TYPE_CHECKING:
    from zarr_vectors.building import Group


#: ``LevelMetadata.coarsening_method`` -> the coarsener key that wrote it.
#:
#: A level must be re-coarsened by the strategy that produced it.  Without
#: this the refresh fell back to ``select_coarsener_key``'s geometry routing,
#: so a per-fragment or quadric-decimated level came back rebuilt by a
#: different strategy -- different fragment identity, different vertex count,
#: silently.
_COARSENER_FOR_METHOD: dict[str, str] = {
    "per_object": "per_object",
    "per_fragment": "per_fragment",
    "mesh_cluster": "mesh",
    "mesh_quadric_collapse": "mesh_decimate",
    "skeleton_simplify": "skeleton",
    "polyline_rdp": "polyline",
    "polyline_decimate": "polyline",
}

#: Methods whose vertex-reduction parameter is NOT recoverable from the store.
#:
#: The skeleton coarsener's stride leaves no trace: it writes
#: ``bin_shape = root bin`` and ``bin_ratio = 1`` whatever the stride was, so
#: the bin-ratio arithmetic every other method is recovered by reads back 1.0
#: -- which the coarsener then reads as "keep anchors only" and flattens the
#: level (measured: a stride-4 level rebuilt from 10 vertices to 8).  Refusing
#: is the only honest option; the caller knows the stride it built with and
#: passes it through ``coarsen_factors``.
_UNRECOVERABLE_FACTOR: dict[str, str] = {
    "skeleton_simplify": (
        "the decimation stride, which no field on the level records"
    ),
}


def rebuild_pyramid_from_level(
    root: Group,
    source_level: int,
    *,
    coarsen_factors: Mapping[int, float] | None = None,
    sparsity_strategy: str = "random",
    sparsity_seed: int | None = None,
    compressor: Any = None,
    executor: Any = None,
) -> list[dict[str, Any]]:
    """Re-coarsen every level above ``source_level`` from scratch.

    Reads each existing target level's metadata (``bin_ratio``,
    ``object_sparsity``, ``chunk_shape``, ``coarsening_method``, and the
    tools-owned coarsening record's ``rdp_tolerance``) and re-runs
    :func:`zarr_vectors_tools.multiresolution.coarsen.coarsen_level` with the
    same parameters, replacing the old level data in place.

    The method each level was built with is re-applied, including the
    ``rdp``/``decimate`` mode for polylines: recovering only the numeric
    factors rebuilt a decimated streamline pyramid with Douglas-Peucker
    instead (measured: 114 vertices became 42), which is a different level,
    not a refreshed one.  An ``rdp`` level built with an explicit tolerance
    is rebuilt at that tolerance, read back from the level's coarsening
    record (:func:`~zarr_vectors_tools.multiresolution.coarsen.read_coarsening_record`);
    a derived one is derived again from the recovered bin, which gives the
    same value.

    Args:
        root: Open store handle.
        source_level: Every level ABOVE this one is rebuilt from it.
        coarsen_factors: Per-level override of the vertex-reduction factor,
            ``{level: factor}``.  Required for levels whose factor the store
            does not record -- today that is skeleton levels, where the factor
            is the decimation stride.
        sparsity_strategy: Object-selection strategy, as
            :func:`~zarr_vectors_tools.multiresolution.coarsen.build_pyramid`
            takes it.  The per-level *fraction* is recovered from the store;
            the strategy is not recorded there, so pass the one the pyramid
            was built with.
        sparsity_seed: Seed for the random strategies.  Advanced per level to
            match ``build_pyramid``.
        compressor: Codec for the rebuilt levels' arrays.  Each level's codec
            is fixed when its arrays are created, so a refresh without this
            leaves raw levels under a compressed level 0.
        executor: ``map``-like callable for the chunk-local coarseners.

    Returns:
        The list of per-level coarsening summaries.

    Raises:
        EditError: If a level cannot be faithfully reproduced -- an unknown
            coarsening method, or one whose factor is unrecoverable and not
            supplied via ``coarsen_factors``.
    """
    from zarr_vectors.building import (
        commit,
        list_resolution_levels,
        read_level_metadata,
        read_root_metadata,
        remove_resolution_level,
        session_for,
    )

    from zarr_vectors_tools.multiresolution.coarsen import (
        coarsen_level,
        read_coarsening_record,
    )

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
    records: dict[int, dict[str, Any]] = {}
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
        # The tools-owned record sits in the same attrs the delete wipes, so
        # it is snapshotted here with everything else.
        records[lv] = read_coarsening_record(root, lv)
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
        method_tag = str(lm.coarsening_method or "")
        coarsener = _COARSENER_FOR_METHOD.get(method_tag)
        if coarsener is None:
            raise EditError(
                f"Cannot refresh level {lv}: it was written by "
                f"coarsening_method={method_tag!r}, which has no coarsener to "
                f"re-run it with (known: {sorted(_COARSENER_FOR_METHOD)}). "
                f"Rebuild the pyramid with build_pyramid and the parameters "
                f"you want."
            )
        override = None if coarsen_factors is None else coarsen_factors.get(lv)
        if override is not None:
            coarsen_factor = float(override)
        elif method_tag in _UNRECOVERABLE_FACTOR:
            raise EditError(
                f"Cannot refresh level {lv}: rebuilding a {method_tag!r} "
                f"level needs {_UNRECOVERABLE_FACTOR[method_tag]}. Pass it as "
                f"coarsen_factors={{{lv}: <stride>}}, or rebuild with "
                f"build_skeleton_pyramid."
            )
        # Only an EXPLICIT tolerance is passed back.  A derived one is a
        # function of the bin, which is recovered above exactly, so deriving
        # it again gives the same value -- and keeps the rebuilt level
        # labelled "derived" rather than turning it into a pinned tolerance
        # that no longer follows the factors.
        record = records.get(lv, {})
        rdp_tolerance = (
            record.get("rdp_tolerance")
            if method_tag == "polyline_rdp"
            and record.get("rdp_tolerance_source") == "explicit"
            else None
        )
        plan.append({
            "level": lv,
            "coarsen_factor": coarsen_factor,
            "sparsity_factor": (
                1.0 / lm.object_sparsity if lm.object_sparsity else 1.0
            ),
            "chunk_scale_factor": scale,
            "parent_level": parent,
            "method": coarsener,
            # The mode is not a stored field, but the method tag the polyline
            # coarsener stamps distinguishes the two modes exactly.
            "coarsen_mode": (
                "decimate" if method_tag == "polyline_decimate" else "rdp"
            ),
            "rdp_tolerance": rdp_tolerance,
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
            method=entry["method"],
            coarsen_mode=entry["coarsen_mode"],
            rdp_tolerance=entry["rdp_tolerance"],
            sparsity_strategy=sparsity_strategy,
            # Advanced per level exactly as build_pyramid does, so a seeded
            # refresh reproduces the seeded build rather than applying one
            # selection to every level.
            sparsity_seed=(
                None if sparsity_seed is None
                else int(sparsity_seed) + (lv - 1)
            ),
            compressor=compressor,
            executor=executor,
        )
        summaries.append(summary)
    return summaries
