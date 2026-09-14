"""Taking one store apart along the lines already in it.

The inverse of :mod:`~zarr_vectors_tools.compose.merge`, and useful for
the same reason: a store that holds a whole-brain tractogram plus
forty-three labelled bundles is the right thing to *serve* and the wrong
thing to hand someone who wants one bundle.  Splitting it should not mean
re-running the conversion with a filter.

**A split is a merge, run N times.**  Each output is a new store fed by
the source restricted to one set of object ids, which means it inherits
everything :func:`~zarr_vectors_tools.compose.merge.merge_stores` already
does correctly — id allocation, attribute columns, group taxonomy,
headers, provenance, pyramid — rather than reimplementing five of them
slightly differently.  The parts differ only in how the id sets are
chosen, which is what :func:`split_store` is really about.

**Grids stay aligned by default.**  Each output keeps the parent's bounds
and cell size, so a bundle store's cells are the same cells as the
tractogram's.  That is what lets the pieces be read together, compared
cell-for-cell, or merged back.  ``bounds="fit"`` tightens each output to
its own contents instead, which is smaller and no longer comparable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from zarr_vectors.exceptions import IngestError, StoreError

from zarr_vectors_tools.compose.sources import StoreSource

__all__ = ["plan_split", "split_store", "split_parts"]


def split_parts(
    source: Any,
    *,
    by: Literal["groups", "attribute", "objects", "provenance"] = "groups",
    attribute: str | None = None,
    names: Mapping[Any, str] | None = None,
    parts: Mapping[str, Sequence[int]] | None = None,
    level: int = 0,
) -> dict[str, npt.NDArray[np.int64]]:
    """Work out which object ids belong to which output.

    Separated from the writing so a caller can inspect, filter or reorder
    the parts before committing to N stores — and so a split that would
    produce four hundred single-object outputs can be noticed first.

    Args:
        source: Store path or open ``Dataset``.
        by: How to cut.  ``"groups"`` uses the named object groups;
            ``"attribute"`` uses the distinct values of a per-object
            attribute; ``"objects"`` takes explicit id lists;
            ``"provenance"`` uses the id offsets a previous merge
            recorded, which is what makes a merge reversible.
        attribute: Object attribute to cut on, for ``by="attribute"``.
        names: Optional ``{value: name}`` lookup for ``by="attribute"``.
        parts: Explicit ``{name: [ids]}``, for ``by="objects"``.
        level: Level to read the taxonomy from.

    Returns:
        ``{part_name: object_ids}``, insertion-ordered.
    """
    import zarr_vectors as zv

    dataset = source if hasattr(source, "level") else zv.open(str(source), mode="r")
    lvl = dataset.level(level)

    if by == "objects":
        if not parts:
            raise IngestError("by='objects' needs parts={name: [object_ids]}")
        return {
            str(name): np.asarray(ids, dtype=np.int64) for name, ids in parts.items()
        }

    if by == "groups":
        catalog = lvl.groups
        out: dict[str, npt.NDArray[np.int64]] = {}
        for name in catalog.names():
            members = np.asarray(catalog[name].members, dtype=np.int64)
            if len(members):
                out[str(name)] = members
        if not out:
            raise IngestError(
                f"{dataset.url} has no named object groups to split on. Use "
                "by='attribute' with the column that holds the labels, or "
                "by='objects' with explicit ids."
            )
        return out

    if by == "attribute":
        if not attribute:
            raise IngestError("by='attribute' needs attribute=<name>")
        from zarr_vectors.building import read_object_attributes

        try:
            column = np.asarray(read_object_attributes(lvl.store, attribute))
        except Exception as exc:  # noqa: BLE001
            raise IngestError(
                f"no object attribute {attribute!r} in {dataset.url}; it has "
                f"{list(lvl.attribute_names('object')) or 'none'}"
            ) from exc
        if column.ndim > 1:
            column = column.reshape(len(column), -1)[:, 0]
        lookup = {_key(k): str(v) for k, v in (names or {}).items()}
        present = np.asarray(lvl.objects.present_mask())[: len(column)]
        out = {}
        # One grouping pass rather than a full-column comparison per
        # distinct value -- see readers._group_members.
        from zarr_vectors_tools.compose.readers import _group_members

        for value, members in _group_members(column).items():
            if isinstance(value, float) and np.isnan(value):
                continue
            members = members[present[members]]
            if not len(members):
                continue
            label = lookup.get(_key(value), f"{attribute}_{_key(value)}")
            out[label] = members.astype(np.int64)
        if not out:
            raise IngestError(f"attribute {attribute!r} has no non-empty values")
        return out

    if by == "provenance":
        return _parts_from_provenance(dataset, lvl)

    raise ValueError(
        f"by={by!r} is not one of 'groups', 'attribute', 'objects', 'provenance'"
    )


def _parts_from_provenance(dataset: Any, level: Any) -> dict[str, npt.NDArray[np.int64]]:
    """Reconstruct the pieces a previous merge combined.

    Each merge records the id offset it wrote at, so consecutive offsets
    bracket exactly the objects one source contributed.  Nothing else on
    disk says where the seam is, which is the whole reason
    :func:`~zarr_vectors_tools.compose._carry.record_provenance` exists.
    """
    from zarr_vectors_tools.compose._carry import read_provenance

    entries = [
        h for h in read_provenance(dataset) if h.get("operation") == "merge"
    ]
    if not entries:
        raise IngestError(
            f"{dataset.url} records no merge history to undo. It was either "
            "not built by merge_stores, or its metadata was not carried."
        )

    seams: list[tuple[str, int]] = []
    for entry in entries:
        labels = list(entry.get("sources") or [])
        offsets = list(entry.get("id_offsets") or [])
        for label, offset in zip(labels, offsets):
            seams.append((str(label), int(offset)))
    seams.sort(key=lambda pair: pair[1])

    total = int(len(level.objects))
    out: dict[str, npt.NDArray[np.int64]] = {}
    if seams and seams[0][1] > 0:
        out["original"] = np.arange(0, seams[0][1], dtype=np.int64)
    for i, (label, start) in enumerate(seams):
        stop = seams[i + 1][1] if i + 1 < len(seams) else total
        if stop > start:
            out[_slug(label)] = np.arange(start, stop, dtype=np.int64)
    return out


def plan_split(source: Any, **kw: Any) -> dict[str, Any]:
    """The parts a split would produce, and how big each is."""
    parts = split_parts(source, **kw)
    return {
        "source": str(getattr(source, "url", source)),
        "part_count": len(parts),
        "parts": [
            {"name": name, "objects": int(len(ids))} for name, ids in parts.items()
        ],
        "total_objects": int(sum(len(ids) for ids in parts.values())),
    }


def split_store(
    source: Any,
    output: str | Path,
    *,
    by: Literal["groups", "attribute", "objects", "provenance"] = "groups",
    attribute: str | None = None,
    names: Mapping[Any, str] | None = None,
    parts: Mapping[str, Sequence[int]] | None = None,
    level: int = 0,
    bounds: Literal["source", "fit"] = "source",
    suffix: str = ".zarrvectors",
    stem: str | None = None,
    pyramid: Literal["rebuild", "drop", "keep"] = "drop",
    pyramid_factors: Sequence[tuple[float, float]] | None = None,
    overwrite: bool = False,
    min_objects: int = 1,
    progress: bool = False,
    **merge_options: Any,
) -> dict[str, Any]:
    """Split ``source`` into one store per part, under ``output``.

    Args:
        source: Store path or open ``Dataset``.
        output: Directory to write the parts into; created if absent.
        by: How to cut — see :func:`split_parts`.
        attribute: Object attribute to cut on, for ``by="attribute"``.
        names: ``{value: name}`` lookup for naming attribute parts.
        parts: Explicit ``{name: [ids]}``, for ``by="objects"``.
        level: Level to read from.  Splitting a coarse level gives stores
            of coarse geometry, which is occasionally what you want and
            usually not.
        bounds: ``"source"`` keeps the parent's grid, so the outputs stay
            cell-aligned with it and with each other.  ``"fit"`` sizes
            each output to its own contents.
        suffix: Extension for each output store.
        stem: Filename prefix; defaults to the source store's own name.
        pyramid: What pyramid each output gets.  ``"drop"`` by default —
            a split usually produces many small stores, and building a
            pyramid on each is rarely wanted and never cheap.
        overwrite: Replace an output that already exists.
        min_objects: Skip parts with fewer objects than this.
        progress: Print each part as it is written.

    Returns:
        A summary dict with one entry per part written, and the names of
        any that were skipped.
    """
    import zarr_vectors as zv

    dataset = source if hasattr(source, "level") else zv.open(str(source), mode="r")
    selection = split_parts(
        dataset, by=by, attribute=attribute, names=names, parts=parts, level=level,
    )

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = stem or Path(str(dataset.url).rstrip("/")).stem

    written: list[dict[str, Any]] = []
    skipped: list[str] = []

    for name, ids in selection.items():
        if len(ids) < min_objects:
            skipped.append(name)
            continue
        target = out_dir / f"{base}_{_slug(name)}{suffix}"
        if target.exists():
            if not overwrite:
                raise StoreError(
                    f"{target} already exists; pass overwrite=True to replace it"
                )
            import shutil

            shutil.rmtree(target)

        part_source = StoreSource(
            dataset, level=level, objects=ids, label=name,
        )
        explicit = (
            _fit_bounds(part_source) if bounds == "fit" else _source_bounds(dataset)
        )
        from zarr_vectors_tools.compose.merge import merge_stores

        summary = merge_stores(
            str(target), [part_source],
            create=True,
            bounds=explicit,
            cell_size=tuple(dataset.level(level).scale),
            pyramid=pyramid,
            pyramid_factors=pyramid_factors,
            **merge_options,
        )
        if progress:
            print(
                f"  {name}: {summary['objects_added']:,} objects -> {target.name}",
                flush=True,
            )
        written.append({
            "name": name,
            "path": str(target),
            "objects": summary["objects_added"],
            "vertices": summary["vertices_added"],
        })

    return {
        "source": str(dataset.url),
        "output": str(out_dir),
        "by": by,
        "written": written,
        "skipped": skipped,
        "part_count": len(written),
    }


def _source_bounds(dataset: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    lo, hi = dataset.bounds
    return tuple(float(v) for v in lo), tuple(float(v) for v in hi)


def _fit_bounds(source: StoreSource) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The extent of just this part, by reading it.

    Costs an extra pass over the part's geometry, which is why
    ``bounds="source"`` is the default: the parent's box is already known
    and needs no read at all.
    """
    lo: npt.NDArray[Any] | None = None
    hi: npt.NDArray[Any] | None = None
    for batch in source.iter_batches():
        box = batch.bounds()
        if box is None:
            continue
        lo = box[0] if lo is None else np.minimum(lo, box[0])
        hi = box[1] if hi is None else np.maximum(hi, box[1])
    if lo is None or hi is None:
        raise StoreError(f"part {source.info.label!r} holds no vertices to size a grid from")
    return tuple(float(v) for v in lo), tuple(float(v) for v in hi)


def _key(value: Any) -> Any:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    number = float(value)
    return int(number) if number.is_integer() else number


def _slug(name: str) -> str:
    keep = [c if (c.isalnum() or c in "-_.") else "-" for c in str(name)]
    return "".join(keep).strip("-") or "part"
