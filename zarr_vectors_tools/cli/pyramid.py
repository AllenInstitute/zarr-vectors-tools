"""``zvtools pyramid`` / ``validate`` / ``info`` subcommands."""

from __future__ import annotations

from ._args import build_factors, executor_ctx


def run_pyramid(args) -> int:
    from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

    factors = build_factors(args.coarsen, args.sparsity)
    if factors is None:
        raise SystemExit("error: pyramid requires --coarsen and --sparsity")

    extra: dict = {}
    if args.cross_level_storage is not None:
        extra["cross_level_storage"] = args.cross_level_storage
    if args.cross_level_depth is not None:
        extra["cross_level_depth"] = args.cross_level_depth

    with executor_ctx(args.workers) as ex:
        result = build_pyramid(
            str(args.store),
            factors=factors,
            chunk_scale_factors=args.chunk_scale,
            sparsity_strategy=args.sparsity_strategy,
            coarsen_mode=args.coarsen_mode,
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
    from zarr_vectors.core.store import (
        list_resolution_levels,
        open_store,
        read_root_metadata,
    )

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
