"""Per-*fragment* coarsening — the reuse-preserving pyramid.

The per-object coarsener bins vertices globally and then re-cuts every
surviving object into per-(object, coarse-chunk) runs.  That is correct for
``implicit_sequential`` geometry, where connectivity is the adjacency of
consecutive vertices inside a fragment and no fragment can therefore carry two
objects' traversal orders — but it destroys fragment sharing.  Two objects that
referenced one fragment at level 0 get two private fragments at level 1, each
holding its own copy of the shared metavertices, so the fragment count scales
with *object×chunk runs* rather than with fragments and a coarse level can end
up larger on disk than the level it coarsens.

This strategy takes the opposite trade.  It never re-cuts anything:

* each source fragment is thinned **in place** — ``coarsen_factor=2`` keeps
  every second vertex, so a fragment of N vertices becomes ``ceil(N/2)``;
* the ``(chunk, fragment_index)`` map is explicit and order-preserving, so a
  manifest entry rewrites 1:1;
* two objects referencing one source fragment therefore reference the **same**
  coarsened fragment — reuse survives, and the fragment count can only shrink
  (unreferenced fragments are dropped), never spike.

Storage stays shared as well.  Thinning is computed on each fragment's own
*vertex indices*, then the chunk's kept indices are unioned into one coarsened
vertex array and every fragment is rewritten as an **index fragment** into it.
So a vertex two fragments have in common is stored once, exactly as at level 0
— the per-object path would have stored it twice.

Endpoints are always kept.  Fragment ends are where cross-chunk seams attach,
so dropping them would silently sever connectivity at chunk boundaries.

Trade-offs, stated plainly
--------------------------
* Vertex *count* reduction is per fragment, so the level's total reduction is
  only ``1/coarsen_factor`` when fragments are disjoint.  Where they overlap,
  the union keeps a vertex any fragment kept, so the realised reduction is
  weaker than the per-object binner's — this strategy trades compression for
  structure.
* There is no spatial re-binning: coarse vertices are a *subset* of the source
  positions, not bin centroids.  Positions are therefore exact rather than
  averaged, which is what makes an index fragment into a shared array possible
  at all.
* ``chunk_scale_factor`` merges source chunks into a coarser grid; fragments
  are concatenated in ``(source chunk, source fragment)`` order so the mapping
  stays deterministic.
* **No cross-level links are emitted.**  ``cross_level_depth`` /
  ``cross_level_storage`` are accepted and ignored, the same way the polyline
  and skeleton strategies ignore them — only the per-object coarsener emits
  inline ``±1`` arrays.  ``_finalize_cross_level_for_store`` still stamps those
  two values on root metadata afterwards, so a store built this way advertises
  a cross-level depth it has no arrays for.  Pass ``cross_level_depth=0,
  cross_level_storage="none"`` if that matters to a downstream reader.  What
  this strategy *does* give you instead is a stable ``(chunk, fragment_index)``
  identity across levels, which is a coarser-grained but cheaper way to relate
  a level to its parent.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from zarr_vectors.constants import (
    CAP_PRESERVED_OBJECT_IDS,
    CAP_SHARED_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.building import (
    LevelMetadata,
    create_object_index_array,
    create_resolution_level,
    create_vertices_array,
    get_level_chunk_shape,
    get_resolution_level,
    list_chunk_keys,
    open_store,
    read_all_object_manifests,
    read_chunk_vertices,
    read_level_metadata,
    read_root_metadata,
    read_vertex_fragment_index,
    write_chunk_fragments,
    write_chunk_vertices,
    write_object_index,
)
from zarr_vectors.exceptions import ArrayError

from ..groupings import propagate_groupings, surviving_oids_from
from ..object_selection import apply_sparsity

#: Level-metadata tag recorded for levels this strategy produced.
COARSEN_PER_FRAGMENT: str = "per_fragment"

#: Dispatch key for :func:`~zarr_vectors_tools.multiresolution.coarsen.coarsen_level`.
COARSENER_KEY: str = "per_fragment"


def _thin_indices(idx: np.ndarray, factor: float, mode: str) -> np.ndarray:
    """Keep ``ceil(len(idx) / factor)`` of ``idx``, endpoints always included.

    ``mode="decimate"`` takes a uniform stride; ``mode="rdp"`` is accepted and
    treated as ``"decimate"`` here, because Douglas-Peucker is defined by a
    distance tolerance rather than a target count and cannot honour the exact
    "``2x`` halves this fragment" contract this strategy exists to provide.
    (The polyline strategy is the one that does true RDP.)
    """
    n = int(idx.shape[0])
    if n <= 2 or factor <= 1.0:
        return idx
    keep_n = max(2, int(np.ceil(n / float(factor))))
    if keep_n >= n:
        return idx
    # linspace over positions, endpoints inclusive — uniform in index space and
    # guaranteed to include 0 and n-1.
    take = np.unique(np.round(np.linspace(0, n - 1, keep_n)).astype(np.int64))
    return idx[take]


def _fragment_source_indices(frag_index, f: int) -> np.ndarray:
    """The source vertex indices of fragment ``f``, range or explicit."""
    if frag_index.is_range(f):
        start, count = frag_index.range(f)
        return np.arange(int(start), int(start) + int(count), dtype=np.int64)
    return np.asarray(frag_index.indices(f), dtype=np.int64)


def _target_chunk(cc: tuple, scale: tuple[int, ...]) -> tuple:
    return tuple(int(c) // int(s) for c, s in zip(cc, scale))


def _normalise_scale(chunk_scale_factor, ndim: int) -> tuple[int, ...]:
    if isinstance(chunk_scale_factor, (list, tuple)):
        scale = tuple(int(v) for v in chunk_scale_factor)
        if len(scale) != ndim:
            raise ArrayError(
                f"chunk_scale_factor rank {len(scale)} != sid_ndim {ndim}"
            )
    else:
        scale = (int(chunk_scale_factor),) * ndim
    if any(s < 1 for s in scale):
        raise ArrayError(f"chunk_scale_factor entries must be >= 1, got {scale}")
    return scale


def coarsen_fragments_level(
    store_path: str | Path,
    source_level: int,
    target_level: int,
    *,
    coarsen_factor: float = 1.0,
    sparsity_factor: float = 1.0,
    chunk_scale_factor: int | tuple[int, ...] = 1,
    sparsity_strategy: str = "random",
    sparsity_seed: int | None = None,
    compressor: Any = None,
    coarsen_mode: str = "decimate",
    **_ignored: Any,
) -> dict[str, Any]:
    """Coarsen one level fragment-wise, preserving fragment identity and reuse.

    See the module docstring for the contract. Returns the usual summary dict
    plus ``fragments_in`` / ``fragments_out`` / ``fragment_reuse`` so a caller
    can assert the count did not spike.
    """
    root = open_store(str(store_path), mode="r+")
    root_meta = read_root_metadata(root)
    ndim = len(root_meta.chunk_shape)
    scale = _normalise_scale(chunk_scale_factor, ndim)
    src_group = get_resolution_level(root, source_level)

    # ``coarsen_factor`` is a ratio against the SOURCE level, matching the
    # per-object coarsener. No spatial binning happens here, but every level
    # >= 1 must still declare a bin_shape (it drives the NGFF scale), so the
    # nominal along-path scale is tracked the same way.
    try:
        _src_lm = read_level_metadata(root, source_level)
    except Exception:  # noqa: BLE001
        _src_lm = None
    _src_bin = getattr(_src_lm, "bin_shape", None) if _src_lm else None
    root_bin = tuple(float(b) for b in root_meta.effective_bin_shape)
    source_bin = tuple(float(b) for b in _src_bin) if _src_bin else root_bin
    target_bin_shape = tuple(b * float(coarsen_factor) for b in source_bin)

    # ---- objects + sparsity ------------------------------------------
    try:
        src_manifests = read_all_object_manifests(src_group)
    except Exception:  # noqa: BLE001
        src_manifests = []
    has_objects = len(src_manifests) > 0
    n_src_objects = len(src_manifests)

    if has_objects and sparsity_factor > 1.0 and n_src_objects > 1:
        alive = np.array(
            [len(m) > 0 for m in src_manifests], dtype=bool,
        )
        # apply_sparsity takes the FRACTION to keep, not the drop divisor:
        # passing sparsity_factor straight through asks it to keep 200% and it
        # silently keeps everything. Same 1/factor the per-object coarsener uses.
        keep_oids = sorted(int(o) for o in apply_sparsity(
            n_src_objects,
            1.0 / float(sparsity_factor),
            sparsity_strategy,
            seed=sparsity_seed,
            alive_mask=alive,
            relative_to="alive",
        ))
    else:
        keep_oids = list(range(n_src_objects))

    # Fragments referenced by a surviving object. With no object index every
    # fragment is kept — there is nothing to say otherwise.
    referenced: set[tuple[tuple, int]] | None = None
    if has_objects:
        referenced = {
            (tuple(int(x) for x in cc), int(f))
            for oid in keep_oids
            for cc, f in src_manifests[oid]
        }

    # ---- thin every referenced fragment, chunk by chunk ---------------
    # target chunk -> ordered list of (index array into that chunk's new
    # vertex array). Source order is preserved so fragment ids are stable.
    out_fragments: dict[tuple, list[np.ndarray]] = {}
    out_vertices: dict[tuple, list[np.ndarray]] = {}
    out_cursor: dict[tuple, int] = {}
    frag_map: dict[tuple[tuple, int], tuple[tuple, int]] = {}
    n_in = n_out = 0

    for cc in sorted(list_chunk_keys(src_group, VERTICES)):
        cc = tuple(int(x) for x in cc)
        try:
            frag_index = read_vertex_fragment_index(src_group, cc)
        except ArrayError:
            continue
        # Materialised per-fragment positions, aligned 1:1 with frag_index.
        try:
            groups = read_chunk_vertices(src_group, cc, dtype=np.float32, ndim=ndim)
        except ArrayError:
            continue
        if not groups:
            continue

        # Reconstruct the chunk's vertex rows, indexed by SOURCE vertex id.
        # ``read_chunk_vertices`` hands back materialised per-fragment blocks,
        # not the underlying array, and a store need not carry a range fragment
        # spanning everything — so scatter each fragment back to its own
        # indices rather than assuming a contiguous layout.
        frag_indices = [
            _fragment_source_indices(frag_index, f)
            for f in range(frag_index.num_fragments)
        ]
        n_rows = max(
            (int(a.max()) + 1 for a in frag_indices if a.size), default=0,
        )
        source_positions = np.zeros((n_rows, ndim), dtype=np.float32)
        for f, idx in enumerate(frag_indices):
            if f >= len(groups) or idx.size == 0:
                continue
            arr = np.asarray(groups[f], dtype=np.float32)
            if arr.shape[0] == idx.size:
                source_positions[idx] = arr

        cc_t = _target_chunk(cc, scale)
        out_fragments.setdefault(cc_t, [])
        out_vertices.setdefault(cc_t, [])
        out_cursor.setdefault(cc_t, 0)

        # Pass A: thin each referenced fragment, collect its kept source ids.
        kept_per_fragment: list[tuple[int, np.ndarray]] = []
        for f in range(frag_index.num_fragments):
            if referenced is not None and (cc, f) not in referenced:
                continue
            n_in += 1
            idx = frag_indices[f]
            if idx.size == 0:
                kept_per_fragment.append((f, idx))
                continue
            kept_per_fragment.append((f, _thin_indices(idx, coarsen_factor, coarsen_mode)))

        if not kept_per_fragment:
            continue

        # Pass B: union the kept source ids -> this chunk's coarsened vertex
        # rows. A vertex several fragments kept is stored ONCE.
        union = np.unique(np.concatenate(
            [k for _f, k in kept_per_fragment if k.size]
            or [np.empty(0, dtype=np.int64)],
        ))
        base = out_cursor[cc_t]
        if union.size:
            out_vertices[cc_t].append(source_positions[union])
            out_cursor[cc_t] = base + int(union.size)

        # Pass C: rewrite each fragment as an index fragment into the union.
        for f, kept in kept_per_fragment:
            if kept.size:
                local = base + np.searchsorted(union, kept)
            else:
                local = np.empty(0, dtype=np.int64)
            frag_map[(cc, f)] = (cc_t, len(out_fragments[cc_t]))
            out_fragments[cc_t].append(local.astype(np.int64, copy=False))
            n_out += 1

    # ---- write the target level ---------------------------------------
    # Scale the SOURCE level's chunk shape, not the root's.  ``_target_chunk``
    # divides SOURCE coords by ``scale``, so the grid this level's coords live
    # on is the source grid scaled once.  Deriving the declared shape from the
    # root instead made every level past the first contradict its own
    # coordinates -- at level 2 of a [2, 2] pyramid the store declared 20 while
    # the coords had been divided by 4, so a bbox query resolved to the wrong
    # chunks.  (``strategies.meshes`` already scales from the source; this is
    # the same fix.)
    src_chunk_shape = get_level_chunk_shape(root_meta, _src_lm)
    target_chunk_shape = tuple(
        float(c) * float(s) for c, s in zip(src_chunk_shape, scale)
    )
    target_chunk_shape_override = (
        None
        if all(
            abs(t - r) < 1e-9
            for t, r in zip(target_chunk_shape, root_meta.chunk_shape)
        )
        else target_chunk_shape
    )

    total_vertices = sum(
        sum(int(a.shape[0]) for a in arrs) for arrs in out_vertices.values()
    )
    arrays_present = [VERTICES, "object_index"] if has_objects else [VERTICES]
    level_meta = LevelMetadata(
        level=target_level,
        vertex_count=int(total_vertices),
        arrays_present=arrays_present,
        # No spatial binning happens here — coarse vertices are a SUBSET of the
        # source positions, not centroids. But every level >= 1 must declare a
        # bin_shape (it drives the NGFF scale transform), so this records the
        # nominal along-path scale the decimation corresponds to, following the
        # polyline strategy's convention of root bin x coarsen_factor.
        bin_shape=target_bin_shape,
        # Fold-change relative to LEVEL 0 — this becomes the NGFF ``scale``.
        # With per-level factors it is not ``coarsen_factor``: [2, 2] is
        # ratio 2 then 4.
        bin_ratio=tuple(
            max(1, int(round(float(t) / float(r))))
            for t, r in zip(target_bin_shape, root_bin)
        ),
        chunk_shape=target_chunk_shape_override,
        object_sparsity=(1.0 / sparsity_factor),
        coarsening_method=COARSEN_PER_FRAGMENT,
        parent_level=source_level,
        preserves_object_ids=has_objects,
        inherited_num_objects=n_src_objects if has_objects else 0,
        # The whole point: a fragment may serve many objects, exactly as at
        # the source level.
        shared_fragments=True,
    )
    level_group = create_resolution_level(root, target_level, level_meta)

    ctx = (
        level_group.batched_writes(compressor=compressor)
        if compressor else nullcontext()
    )
    with ctx:
        create_vertices_array(level_group, dtype="float32")
        if has_objects:
            create_object_index_array(level_group)

    for cc_t in sorted(out_vertices):
        arrs = out_vertices[cc_t]
        if not arrs:
            continue
        stacked = np.concatenate(arrs, axis=0)
        # One range fragment covering the chunk's coarsened vertices, then the
        # real fragments as explicit index entries on top — the same shape the
        # source level has, so a reader sees no structural change.
        write_chunk_vertices(level_group, cc_t, [stacked], dtype=np.float32)
        write_chunk_fragments(
            level_group, cc_t, list(out_fragments[cc_t]),
            target="vertex", mode="replace",
        )

    # ---- remap manifests ----------------------------------------------
    if has_objects:
        new_manifests: dict[int, list] = {}
        for oid in keep_oids:
            entries = []
            for cc, f in src_manifests[oid]:
                ref = frag_map.get((tuple(int(x) for x in cc), int(f)))
                if ref is not None:
                    entries.append(ref)
            new_manifests[oid] = entries
        write_object_index(
            level_group, new_manifests, sid_ndim=ndim,
            total_objects=n_src_objects,
        )
        # Object ids are preserved, so the source level's group taxonomy is
        # meaningful here unchanged — carry it forward so ``<N>/groups/<id>``
        # answers the same question ``0/groups/<id>`` does.
        propagate_groupings(
            src_group, level_group,
            surviving_oids=surviving_oids_from(keep_oids, sparsity_factor),
        )

    # ---- root capability tokens ---------------------------------------
    # Both are genuinely true here and a reader may branch on them: OIDs are
    # carried across unchanged, and a coarse fragment really is referenced by
    # every object that referenced its source. Leaving CAP_SHARED_FRAGMENTS
    # unstamped would tell a consumer it cannot dedupe by fragment identity
    # when it can — the exact opposite of the per-object path's situation.
    # Imported here, not at module scope: ``coarsen`` imports THIS module at
    # the bottom of its own body to register the strategy, so a module-level
    # import back into it only works while ``_stamp_root_capability`` happens
    # to be defined above that line. A lazy import removes the cycle.
    from ..coarsen import _stamp_root_capability

    if has_objects:
        _stamp_root_capability(root, CAP_PRESERVED_OBJECT_IDS)
    _stamp_root_capability(root, CAP_SHARED_FRAGMENTS)

    return {
        "vertex_count": int(total_vertices),
        "object_count": len(keep_oids),
        "objects_kept": len(keep_oids),
        "method": COARSEN_PER_FRAGMENT,
        "preserves_object_ids": has_objects,
        "shared_fragments": True,
        "fragments_in": int(n_in),
        "fragments_out": int(n_out),
    }


def _per_fragment_coarsener(
    store_path, source_level, target_level, **kwargs,
) -> dict[str, Any]:
    """Registry adapter — drops the kwargs this strategy has no use for."""
    return coarsen_fragments_level(
        store_path, source_level, target_level,
        coarsen_factor=kwargs.get("coarsen_factor", 1.0),
        sparsity_factor=kwargs.get("sparsity_factor", 1.0),
        chunk_scale_factor=kwargs.get("chunk_scale_factor", 1),
        sparsity_strategy=kwargs.get("sparsity_strategy", "random"),
        sparsity_seed=kwargs.get("sparsity_seed"),
        compressor=kwargs.get("compressor"),
        coarsen_mode=kwargs.get("coarsen_mode", "decimate"),
    )


__all__ = [
    "COARSENER_KEY",
    "COARSEN_PER_FRAGMENT",
    "_per_fragment_coarsener",
    "coarsen_fragments_level",
]
