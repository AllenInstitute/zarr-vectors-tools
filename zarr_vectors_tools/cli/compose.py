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
        else:
            built.append(open_source(spec, transform=transform))
    return built


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

    if args.dry_run:
        _print("merge plan:", plan_merge(str(args.target), list(args.sources)))
        return 0

    factors = None
    if args.pyramid_coarsen or args.pyramid_sparsity:
        coarsen = args.pyramid_coarsen or []
        sparsity = args.pyramid_sparsity or []
        if len(coarsen) != len(sparsity):
            raise SystemExit(
                f"error: --pyramid-coarsen has {len(coarsen)} entries but "
                f"--pyramid-sparsity has {len(sparsity)}; they must match"
            )
        factors = [(float(c), float(s)) for c, s in zip(coarsen, sparsity)]

    sources = _build_sources(args)
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
        pyramid=args.pyramid,
        overwrite=args.overwrite,
        min_objects=args.min_objects,
        progress=True,
        **common,
    )
    _print("split:", summary)
    return 0
