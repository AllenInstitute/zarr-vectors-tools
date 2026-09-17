"""``zvtools pyramid`` / ``validate`` / ``info`` subcommands."""

from __future__ import annotations

from ._args import build_factors, check_rdp_tolerances, executor_ctx


def _refuse_rdp_tolerance_for_store(store) -> None:
    """Refuse ``--rdp-tolerance`` on a store whose coarsener has no tolerance.

    ``build_pyramid`` refuses it too, but names its own parameter; checking
    here names the flag the user typed.
    """
    from zarr_vectors.building import open_store, read_root_metadata

    from zarr_vectors_tools.multiresolution.coarsen import select_coarsener_key

    meta = read_root_metadata(open_store(str(store), mode="r"))
    key = select_coarsener_key(meta)
    if key != "polyline":
        raise SystemExit(
            f"error: --rdp-tolerance applies to streamline stores, whose "
            f"coarsener simplifies by distance; {store} has geometry "
            f"{list(meta.geometry_types or [])} and is coarsened by {key!r}, "
            f"which has no tolerance"
        )


def run_pyramid(args) -> int:
    from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

    factors = build_factors(args.coarsen, args.sparsity)
    if factors is None:
        raise SystemExit("error: pyramid requires --coarsen and --sparsity")
    tolerances = check_rdp_tolerances(args.rdp_tolerance, factors, args.coarsen_mode)
    if tolerances is not None:
        _refuse_rdp_tolerance_for_store(args.store)

    extra: dict = {}
    if args.cross_level_storage is not None:
        extra["cross_level_storage"] = args.cross_level_storage
    if args.cross_level_depth is not None:
        extra["cross_level_depth"] = args.cross_level_depth

    with executor_ctx(args.workers, args.workers_backend) as ex:
        result = build_pyramid(
            str(args.store),
            factors=factors,
            chunk_scale_factors=args.chunk_scale,
            sparsity_strategy=args.sparsity_strategy,
            coarsen_mode=args.coarsen_mode,
            rdp_tolerances=tolerances,
            compressor=(None if args.compressor == "none" else args.compressor),
            executor=ex,
            **extra,
        )
    print(
        f"built {result.get('levels_created', '?')} coarser level(s) "
        f"(method={result.get('method')})"
    )
    return 0


def run_validate(args) -> int:
    from zarr_vectors.validate import validate

    result = validate(str(args.store), level=args.level)
    ok = bool(getattr(result, "ok", False))
    print(f"validation (level {args.level}): {'OK' if ok else 'FAILED'}")
    for attr in ("errors", "messages", "issues"):
        for item in (getattr(result, attr, None) or []):
            print(f"  - {item}")
    return 0 if ok else 1


def run_info(args) -> int:
    from zarr_vectors.building import list_resolution_levels, open_store, read_root_metadata

    root = open_store(str(args.store))
    md = read_root_metadata(root)
    levels = list_resolution_levels(root)
    print(f"store: {args.store}")
    print(f"  zv_version:        {md.zv_version}")
    print(f"  geometry_types:    {md.geometry_types}")
    print(f"  links_convention:  {md.links_convention}")
    print(f"  chunk_shape:       {md.chunk_shape}")
    print(f"  bounds:            {md.bounds}")
    print(f"  resolution levels: {levels}")
    print(f"  cross_level:       depth={md.cross_level_depth} storage={md.cross_level_storage}")
    print(f"  capabilities:      {md.format_capabilities}")
    return 0


def run_bundles(args) -> int:
    """``zvtools bundles STORE``."""
    import pandas as pd

    from zarr_vectors_tools.algorithms.bundles import bundle_summary

    table = bundle_summary(
        str(args.store), level=args.level, write=not args.no_write,
        orient_endpoints=not args.as_stored,
    )
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(table)
    if args.csv:
        table.to_csv(args.csv)
    written = table.attrs["levels_written"]
    print(
        f"summarised {len(table)} group(s) from level {table.attrs['source_level']}; "
        + (f"written to level(s) {written}" if written else "not written")
    )
    return 0
