"""``zvtools merge`` / ``zvtools split`` — moving objects between stores.

Thin argument plumbing over
:func:`~zarr_vectors_tools.compose.merge.merge_stores` and
:func:`~zarr_vectors_tools.compose.split.split_store`.  Two things get
decided here rather than in the library, because they are terminal
concerns and not API ones: ``--dry-run``, which prints the plan and
writes nothing, and the ``--group-by`` / ``--lut`` pair, which turns a
label column plus a JSON lookup table into named groups on the way in —
the shape a bundle atlas actually ships in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _print(title: str, payload: Any) -> None:
    print(title)
    print(json.dumps(payload, indent=2, default=str))


def _build_sources(args) -> list[Any]:
    """Turn the positional sources into :class:`Source` objects.

    A transform, a label-derived grouping or a non-default TRK space each
    need the source constructed rather than named, so this is where a
    bare path stops being enough.
    """
    from zarr_vectors_tools.compose import GeometrySource, open_source
    from zarr_vectors_tools.compose.readers import (
        derive_groups,
        has_native_reader,
        load_lut,
        read_geometry,
        resolve_reader_format,
    )

    transform = None
    if args.transform:
        transform = _load_transform(args.transform)

    names = load_lut(args.lut) if args.lut else None
    built: list[Any] = []
    for spec in args.sources:
        path = Path(spec)
        fmt = None
        if path.is_file():
            fmt = resolve_reader_format(path, args.format)

        if not (fmt and has_native_reader(fmt)):
            # Only a natively read file can be grouped or re-spaced on the
            # way in; a store or a staged file would merge without it.
            ignored = [
                flag for flag, given in (
                    ("--group-by", bool(args.group_by)),
                    ("--lut", bool(args.lut)),
                    ("--space", args.space != "voxmm"),
                ) if given
            ]
            if ignored:
                verb = "apply" if len(ignored) > 1 else "applies"
                raise SystemExit(
                    f"error: {' / '.join(ignored)} {verb} to TRK, TCK and TRX "
                    f"files, not to {spec}"
                )

        if fmt and has_native_reader(fmt) and (args.group_by or args.space != "voxmm"):
            options: dict[str, Any] = {}
            if fmt == "trk":
                options["space"] = args.space
            geometry = read_geometry(path, fmt, **options)
            if args.group_by:
                derive_groups(geometry, args.group_by, names=names)
            built.append(
                GeometrySource(geometry, transform=transform, label=path.name)
            )
        elif fmt and not has_native_reader(fmt):
            # Staged through a scratch store, whose ingester needs a chunk
            # shape.  The merge re-bins onto the target's grid, so the
            # target's own chunk shape is as good as any and needs no flag.
            built.append(open_source(
                spec, transform=transform, format=fmt,
                chunk_shape=_staging_chunk_shape(args),
            ))
        else:
            built.append(open_source(spec, transform=transform))
    return built


def _staging_chunk_shape(args) -> tuple[float, ...]:
    """The chunk shape to stage a file source with: the target's, else ``--cell-size``."""
    root = Path(args.target) / "zarr.json"
    if root.exists():
        attrs = json.loads(root.read_text()).get("attributes", {})
        chunk = attrs.get("zarr_vectors", {}).get("chunk_shape")
        if chunk:
            return tuple(float(c) for c in chunk)
    if args.cell_size:
        return tuple(args.cell_size)
    raise SystemExit(
        f"error: {args.target} does not exist yet, so a file source has no "
        f"grid to stage on; pass --cell-size X,Y,Z with --create"
    )


def _load_transform(spec: str):
    """A 4x4 affine from a ``.npy``, a ``.json``, or 16 comma-separated numbers."""
    import numpy as np

    path = Path(spec)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".json":
        return np.asarray(json.loads(path.read_text()), dtype=float)
    values = [float(v) for v in spec.replace(";", ",").split(",") if v.strip()]
    side = int(round(len(values) ** 0.5))
    if side * side != len(values):
        raise SystemExit(
            f"error: --transform needs a square matrix; got {len(values)} numbers"
        )
    return np.asarray(values, dtype=float).reshape(side, side)


def run_merge(args) -> int:
    from zarr_vectors_tools.compose import merge_stores, plan_merge

    from ._args import build_factors

    factors = build_factors(
        args.pyramid_coarsen, args.pyramid_sparsity,
        flags=("--pyramid-coarsen", "--pyramid-sparsity"),
    )
    if factors and args.pyramid != "rebuild":
        raise SystemExit(
            f"error: --pyramid-coarsen / --pyramid-sparsity describe a rebuilt "
            f"pyramid; they do nothing with --pyramid {args.pyramid}"
        )

    sources = _build_sources(args)
    if args.dry_run:
        try:
            _print("merge plan:", plan_merge(str(args.target), sources))
        finally:
            for source in sources:
                source.close()
        return 0
    try:
        summary = merge_stores(
            str(args.target), sources,
            create=args.create,
            cell_size=args.cell_size,
            pyramid=args.pyramid,
            pyramid_factors=factors,
            pyramid_options={
                "coarsen_mode": args.coarsen_mode,
                "sparsity_strategy": args.sparsity_strategy,
            },
            group_prefix=args.group_prefix,
            source_attribute=args.source_attr,
            on_out_of_bounds=args.on_out_of_bounds,
            progress=True,
        )
    finally:
        for source in sources:
            source.close()

    _print("merged:", summary)
    return 0


def run_split(args) -> int:
    from zarr_vectors_tools.compose import plan_split, split_store
    from zarr_vectors_tools.compose.readers import load_lut

    from ._args import build_factors

    factors = build_factors(
        args.pyramid_coarsen, args.pyramid_sparsity,
        flags=("--pyramid-coarsen", "--pyramid-sparsity"),
    )
    # Asking for levels is asking for a pyramid, so the factors switch the
    # default from drop to rebuild; only an explicit --pyramid drop refuses.
    pyramid = args.pyramid or ("rebuild" if factors else "drop")
    if factors and pyramid != "rebuild":
        raise SystemExit(
            "error: --pyramid-coarsen / --pyramid-sparsity describe a rebuilt "
            "pyramid; they do nothing with --pyramid drop"
        )

    names = load_lut(args.lut) if args.lut else None
    common = {
        "by": args.by,
        "attribute": args.attribute,
        "names": names,
        "level": args.level,
    }
    if args.dry_run:
        _print("split plan:", plan_split(str(args.store), **common))
        return 0

    summary = split_store(
        str(args.store), args.output,
        bounds=args.bounds,
        pyramid=pyramid,
        pyramid_factors=factors,
        pyramid_options={
            "coarsen_mode": args.coarsen_mode,
            "sparsity_strategy": args.sparsity_strategy,
        },
        overwrite=args.overwrite,
        min_objects=args.min_objects,
        progress=True,
        **common,
    )
    _print("split:", summary)
    return 0
