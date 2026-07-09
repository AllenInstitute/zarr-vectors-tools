"""``zvtools convert`` — ingest a file into a new zarr-vectors store."""

from __future__ import annotations

import shutil
from pathlib import Path

from ._args import (
    build_factors,
    executor_ctx,
    load_ingest_func,
    resolve_format,
)


def _maybe_overwrite(output, overwrite: bool) -> None:
    """Remove an existing output store when ``--overwrite`` is set.

    Only deletes a directory that looks like a zarr store (has ``zarr.json`` or
    ``.zattrs``) — including a partial store from a failed ingest, whose root is
    created before level-0 data — so an unrelated directory is never wiped.
    """
    p = Path(output)
    if not p.exists():
        return
    if not overwrite:
        return  # the ingester's own "Store already exists" error still fires
    if p.is_dir():
        looks_like_store = (p / "zarr.json").exists() or (p / ".zattrs").exists()
        if not looks_like_store:
            raise SystemExit(
                f"error: refusing to overwrite {output}: not a zarr-vectors store"
            )
        shutil.rmtree(p)
    else:
        p.unlink()
    print(f"overwrite: removed existing store at {output}")


def _print_summary(action: str, summary: dict) -> None:
    print(action)
    for k in (
        "streamline_count", "vertex_count", "object_count",
        "chunk_count", "cross_chunk_link_count", "chunk_shape", "bounds",
    ):
        if k in summary:
            print(f"  {k}: {summary[k]}")


def _build_pyramid_post(args, factors, chunk_scale) -> None:
    """Build a sparsity pyramid on the just-written level-0 store."""
    from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

    with executor_ctx(args.workers) as ex:
        result = build_pyramid(
            str(args.output),
            factors=factors,
            chunk_scale_factors=chunk_scale,
            sparsity_strategy=args.sparsity_strategy,
            coarsen_mode=args.coarsen_mode,
            executor=ex,
        )
    print(f"  pyramid: {result.get('levels_created', '?')} coarser level(s) built")


def _convert_trk(args, factors, chunk_scale) -> int:
    """Streamlines via the memory-bounded parallel ingester (inline pyramid)."""
    from zarr_vectors_tools.ingest.trk_parallel import ingest_trk_parallel

    with executor_ctx(args.workers) as ex:
        summary = ingest_trk_parallel(
            str(args.input),
            str(args.output),
            num_chunks=args.num_chunks,
            n_parts=args.n_parts,
            workers=(args.workers or 1),
            executor=ex,
            dtype=args.dtype,
            compute_length=args.compute_length,
            compute_endpoints=args.compute_endpoints,
            build_multiscale=factors is not None,
            pyramid_factors=factors,
            chunk_scale_factors=chunk_scale,
            sparsity_strategy=args.sparsity_strategy,
            pyramid_coarsen_mode=args.coarsen_mode,
            progress=True,
        )
    _print_summary("ingested trk (streamlines)", summary)
    if factors is not None:
        print(f"  pyramid: {len(factors)} coarser level(s) built")
    return 0


def run(args) -> int:
    fmt = resolve_format(args.input, args.format)
    factors = build_factors(args.coarsen, args.sparsity)
    chunk_scale = args.chunk_scale

    _maybe_overwrite(args.output, args.overwrite)

    # The "length" pyramid strategy ranks by per-object length, which must be
    # computed at ingest; auto-enable it for streamlines so the pyramid step
    # doesn't fail with "requires object_attributes/length".
    if args.sparsity_strategy == "length" and fmt.geometry == "streamlines":
        if not args.compute_length:
            args.compute_length = True
            print("note: --sparsity-strategy length → enabling --compute-length")

    # trk has its own streaming path with an inline pyramid + num_chunks knob.
    if fmt.name == "trk":
        return _convert_trk(args, factors, chunk_scale)

    if args.chunk_shape is None:
        raise SystemExit(f"error: --chunk-shape X,Y,Z is required for format {fmt.name!r}")

    ingest = load_ingest_func(fmt)
    kwargs: dict = {"bin_shape": args.bin_shape, "dtype": args.dtype}
    if fmt.geometry == "streamlines":  # trx / tck
        kwargs["compute_length"] = args.compute_length
        kwargs["compute_endpoints"] = args.compute_endpoints
    if fmt.geometry == "points" and args.knn_distance_k is not None:  # ply / las / csv
        kwargs["knn_distance_k"] = args.knn_distance_k

    try:
        if fmt.name == "edgelist":
            if not args.nodes:
                raise SystemExit("error: --nodes NODES.csv is required for format 'edgelist'")
            summary = ingest(
                str(args.input), str(args.nodes), str(args.output),
                tuple(args.chunk_shape), bin_shape=args.bin_shape, dtype=args.dtype,
            )
        else:
            summary = ingest(str(args.input), str(args.output), tuple(args.chunk_shape), **kwargs)
    except ImportError as exc:  # a heavy reader dep was missing at call time
        hint = (
            f" — install it with: pip install 'zarr-vectors-tools[{fmt.extra}]'"
            if fmt.extra else ""
        )
        raise SystemExit(f"error: {fmt.name} ingest failed ({exc}){hint}")

    _print_summary(f"ingested {fmt.name} ({fmt.geometry})", summary)

    if factors is not None:
        _build_pyramid_post(args, factors, chunk_scale)
    return 0
