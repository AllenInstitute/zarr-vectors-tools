"""``zvtools convert`` — move data between a file and a zarr-vectors store.

Both directions, chosen by what ``INPUT`` is:

* a FILE in, a store out — ingest, optionally building a pyramid (a
  precomputed layer, a directory or URL rather than a file, counts as one);
* a STORE in, a file out — export.

One command rather than two because it is one operation with a direction,
and because a pipeline that ends in an export otherwise has to drop out of
the CLI to a Python script for its last step.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ._args import (
    build_factors,
    executor_ctx,
    load_export_func,
    load_ingest_func,
    looks_like_store,
    resolve_export_format,
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
        "spatial_key", "n_obs", "n_vars", "obs_columns_stored", "genes_stored",
        "columns_stored", "dropped_na", "key_column",
        # export-side counters
        "node_count", "root_count", "face_count", "attributes_carried",
        # cortical surfaces
        "hemispheres", "geometry", "space", "c_ras", "scalars", "labels",
        "alternates", "filled",
        # precomputed skeleton layers
        "layer", "bounds_nm", "frag_chunks_read", "source_segment_pieces",
        "objects", "level0_fragments", "level0_cross_chunk_edges",
    ):
        if k in summary:
            print(f"  {k}: {summary[k]}")


def _build_pyramid_post(args, factors, chunk_scale) -> None:
    """Build a sparsity pyramid on the just-written level-0 store."""
    from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

    with executor_ctx(args.workers, args.workers_backend) as ex:
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

    with executor_ctx(args.workers, args.workers_backend) as ex:
        summary = ingest_trk_parallel(
            str(args.input),
            str(args.output),
            num_chunks=args.num_chunks,
            n_parts=args.n_parts,
            workers=(args.workers or 1),
            executor=ex,
            dtype=args.dtype,
            # "none" is the argparse spelling of "store raw"; resolve_compressor
            # accepts it, but pass None so the no-compressor path stays the
            # byte-for-byte default it has always been.
            compressor=(None if args.compressor == "none" else args.compressor),
            compute_length=args.compute_length,
            compute_endpoints=args.compute_endpoints,
            object_attrs=set(args.object_attrs or []),
            vertex_attrs=set(args.vertex_attrs or []),
            attr_seed=args.attr_seed,
            register_to_rasmm=args.apply_affine,
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


def _convert_precomputed(args, fmt, factors, chunk_scale) -> None:
    """A precomputed skeleton layer, with the pyramid built inside the ingest.

    Which of the two precomputed ingesters runs is only known once the
    layer's ``info`` is read, so the options that belong to one of them are
    checked there, not here.
    """
    refused = {
        "--bin-shape": args.bin_shape is not None,
        "--dtype": args.dtype != "float32",
        "--compressor": args.compressor != "none",
    }
    rejected = sorted(flag for flag, given in refused.items() if given)
    if rejected:
        raise SystemExit(
            f"error: {', '.join(rejected)} "
            f"{'do' if len(rejected) > 1 else 'does'} not apply to precomputed "
            f"input: the skeleton ingesters store float32 positions, raw, "
            f"with no sub-binning"
        )
    for flag, value in (("--anchor", args.anchor), ("--counts", args.counts)):
        if value is not None and len(value) != 3:
            raise SystemExit(f"error: {flag} takes three integers, X,Y,Z")

    strides: list[int] = []
    sparsity: list[float] | None = None
    strategy = args.sparsity_strategy
    if factors is not None:
        # Skeletons decimate by stride; round a fractional --coarsen the way
        # ``zvtools pyramid`` does on a skeleton store.
        strides = [max(1, int(round(c))) for c, _ in factors]
        sparsity = [s for _, s in factors]
        if strategy == "random":
            strategy = "length"
            if any(s > 1 for s in sparsity):
                print("note: skeleton pyramids have no random sparsity; "
                      "using --sparsity-strategy length")

    ingest = load_ingest_func(fmt)
    with executor_ctx(args.workers, args.workers_backend) as ex:
        try:
            summary = ingest(
                args.input, str(args.output), args.chunk_shape,
                frags_dir=args.frags_dir,
                anchor=args.anchor,
                counts=args.counts,
                segment_ids=args.segment_ids,
                strides=strides,
                chunk_scale_factors=chunk_scale,
                sparsity_factors=sparsity,
                sparsity_strategy=strategy,
                drop_interior_below=args.drop_interior_below,
                executor=ex,
            )
        except ImportError as exc:  # cloud-files / cloud-volume / mapbuffer
            raise SystemExit(
                f"error: precomputed ingest failed ({exc}) — install it with: "
                f"pip install 'zarr-vectors-tools[{fmt.extra}]'"
            )

    _print_summary("ingested precomputed (skeleton)", summary)
    if strides:
        print(f"  pyramid: {len(strides)} coarser level(s) built")


#: CLI option -> the exporter keyword it fills, for the options an exporter
#: may or may not take.  ``ExportFmt.accepts`` decides which of these a given
#: format gets; anything the user set that is not accepted is refused by name.
_EXPORT_OPTIONS: dict[str, tuple[str, str]] = {
    # cli attribute: (exporter keyword, the flag that sets it)
    "export_object_ids": ("object_ids", "--object-id"),
    "export_group_ids": ("group_ids", "--group-id"),
    "export_bbox": ("bbox", "--bbox"),
    "export_attributes": ("attribute_names", "--attribute"),
}


def run_export(args) -> int:
    """``zvtools convert STORE OUT.ext`` — write a store back out to a file."""
    fmt = resolve_export_format(args.output, args.format)

    if args.chunk_shape is not None or args.num_chunks is not None:
        raise SystemExit(
            "error: --chunk-shape / --num-chunks describe how to BUILD a "
            "store; they do not apply when exporting one"
        )
    if args.coarsen or args.sparsity:
        raise SystemExit(
            "error: --coarsen / --sparsity build pyramid levels; to export an "
            "existing one pass --level N"
        )

    kwargs: dict[str, Any] = {"level": args.level}
    rejected: list[str] = []
    if args.export_bbox is not None and len(args.export_bbox) % 2 != 0:
        raise SystemExit(
            f"error: --bbox needs a min corner and a max corner, so an even "
            f"number of values; got {len(args.export_bbox)}"
        )
    for attribute, (keyword, flag) in _EXPORT_OPTIONS.items():
        value = getattr(args, attribute, None)
        if value is None:
            continue
        if keyword not in fmt.accepts:
            rejected.append(flag)
            continue
        if keyword == "bbox":
            # The exporters take a (min corner, max corner) pair; the flag is
            # flat so it reads the way a bounding box is usually written.
            half = len(value) // 2
            value = (tuple(value[:half]), tuple(value[half:]))
        kwargs[keyword] = value
    # Format-specific options that have a default, so "was it set?" is a
    # comparison rather than a None check.
    if args.delimiter != ",":
        if "delimiter" in fmt.accepts:
            kwargs["delimiter"] = args.delimiter
        else:
            rejected.append("--delimiter")
    if rejected:
        raise SystemExit(
            f"error: {', '.join(sorted(rejected))} "
            f"{'do' if len(rejected) > 1 else 'does'} not apply to "
            f"{fmt.name!r} export (it takes: "
            f"{', '.join(sorted(fmt.accepts - {'level', 'chunks'})) or 'no options'})"
        )

    export = load_export_func(fmt)
    try:
        summary = export(str(args.input), str(args.output), **kwargs)
    except ImportError as exc:
        hint = (
            f" — install it with: pip install 'zarr-vectors-tools[{fmt.extra}]'"
            if fmt.extra else ""
        )
        raise SystemExit(f"error: {fmt.name} export failed ({exc}){hint}")

    _print_summary(
        f"exported {fmt.name} ({fmt.geometry}) from level {args.level}",
        dict(summary or {}),
    )
    print(f"  wrote: {args.output}")
    return 0


def run(args) -> int:
    # A store on the input side means the user is exporting out of it.  The
    # ingest options below are all about building a store and none of them
    # mean anything in that direction, so the branch comes first.
    if looks_like_store(args.input):
        return run_export(args)

    fmt = resolve_format(args.input, args.format)
    factors = build_factors(args.coarsen, args.sparsity)
    chunk_scale = args.chunk_scale

    # --apply-affine only applies to trk (the one format read in raw,
    # unregistered voxmm coordinates).  Everything else is either already in RAS
    # world space (trx/tck) or has no source affine (meshes/points/graphs), so
    # reject the flag there rather than accept-and-ignore it.
    if getattr(args, "apply_affine", False) and fmt.name != "trk":
        why = (
            "its streamlines are already read in RAS world space"
            if fmt.geometry == "streamlines"
            else "this format has no source affine to bake"
        )
        raise SystemExit(
            f"error: --apply-affine only applies to trk input, not {fmt.name!r} "
            f"({why})"
        )

    # The synthetic attribute generators are wired into the trk parallel path
    # only.  trx/tck carry their own native per-vertex/per-object scalars, and
    # non-streamline formats have no streamlines to derive them from — so reject
    # rather than accept-and-ignore (mirrors --apply-affine above).
    if (getattr(args, "object_attrs", None) or getattr(args, "vertex_attrs", None)) \
            and fmt.name != "trk":
        raise SystemExit(
            f"error: --object-attr/--vertex-attr only apply to trk input, "
            f"not {fmt.name!r}"
        )

    # The h5ad and table groups address side tables / named columns that the
    # other formats have no equivalent of — reject rather than
    # accept-and-ignore, as above.  Each flag lists the formats that read it.
    flag_owners = {
        "--spatial-key": ({"h5ad"}, getattr(args, "spatial_key", "auto") != "auto"),
        "--spatial-columns": ({"h5ad"}, getattr(args, "spatial_columns", None) is not None),
        "--obs-column": ({"h5ad"}, bool(getattr(args, "obs_columns", None))),
        "--no-obs": ({"h5ad"}, bool(getattr(args, "no_obs", False))),
        "--gene": ({"h5ad"}, bool(getattr(args, "genes", None))),
        "--layer": ({"h5ad"}, getattr(args, "layer", None) is not None),
        "--backed": ({"h5ad"}, bool(getattr(args, "backed", False))),
        "--position-columns": ({"table"}, getattr(args, "position_columns", None) is not None),
        "--key-column": ({"table"}, getattr(args, "key_column", None) is not None),
        "--column": ({"table"}, bool(getattr(args, "columns", None))),
        "--delimiter": ({"table"}, getattr(args, "delimiter", ",") != ","),
        "--object-id-column": ({"h5ad", "table"},
                               getattr(args, "object_id_column", None) is not None),
        "--drop-na": ({"h5ad", "table"}, bool(getattr(args, "drop_na", False))),
        "--geometry": ({"gifti", "freesurfer"}, getattr(args, "geometry", None) is not None),
        "--hemisphere": ({"gifti", "freesurfer"}, bool(getattr(args, "hemispheres", None))),
        "--space": ({"freesurfer"}, getattr(args, "space", "auto") != "auto"),
        "--surface": ({"freesurfer"}, bool(getattr(args, "surfaces", None))),
        "--morph": ({"freesurfer"}, bool(getattr(args, "morph", None))),
        "--annot": ({"freesurfer"}, bool(getattr(args, "annots", None))),
        "--anchor": ({"precomputed"}, getattr(args, "anchor", None) is not None),
        "--counts": ({"precomputed"}, getattr(args, "counts", None) is not None),
        "--frags-dir": ({"precomputed"}, bool(getattr(args, "frags_dir", ""))),
        "--segment-id": ({"precomputed"}, bool(getattr(args, "segment_ids", None))),
        "--drop-interior-below": ({"precomputed"},
                                  bool(getattr(args, "drop_interior_below", 0))),
    }
    rejected = [
        (flag, owners) for flag, (owners, given) in flag_owners.items()
        if given and fmt.name not in owners
    ]
    if rejected:
        detail = "; ".join(
            f"{flag} applies to {'/'.join(sorted(owners))} input" for flag, owners in rejected
        )
        raise SystemExit(f"error: {detail} — not {fmt.name!r}")

    if fmt.name == "table" and not args.position_columns:
        raise SystemExit(
            "error: --position-columns X,Y[,Z] is required for --format table"
        )

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
        rc = _convert_trk(args, factors, chunk_scale)
        if rc != 0:
            return rc
    elif fmt.name == "precomputed":
        _convert_precomputed(args, fmt, factors, chunk_scale)
    else:
        if args.chunk_shape is None:
            raise SystemExit(f"error: --chunk-shape X,Y,Z is required for format {fmt.name!r}")

        ingest = load_ingest_func(fmt)
        kwargs: dict = {"bin_shape": args.bin_shape, "dtype": args.dtype}
        if fmt.geometry == "streamlines":  # trx / tck
            kwargs["compute_length"] = args.compute_length
            kwargs["compute_endpoints"] = args.compute_endpoints
        if fmt.geometry == "points" and args.knn_distance_k is not None:  # ply / las / csv
            kwargs["knn_distance_k"] = args.knn_distance_k
        if fmt.name == "table":
            kwargs["position_columns"] = args.position_columns
            kwargs["key_column"] = args.key_column
            kwargs["columns"] = args.columns
            kwargs["delimiter"] = args.delimiter
            kwargs["object_id_column"] = args.object_id_column
            if args.drop_na:
                kwargs["drop_na"] = True
        if fmt.name == "h5ad":
            kwargs["spatial_key"] = args.spatial_key
            kwargs["spatial_columns"] = args.spatial_columns
            # None = every obs column; [] = none.  --no-obs is the only way to
            # spell the empty list on a repeatable append flag.
            kwargs["obs_columns"] = [] if args.no_obs else args.obs_columns
            kwargs["genes"] = args.genes
            kwargs["layer"] = args.layer
            kwargs["object_id_column"] = args.object_id_column
            kwargs["backed"] = args.backed
            kwargs["drop_na"] = args.drop_na
        if fmt.name == "gifti":
            kwargs["geometry"] = args.geometry
            hemis = args.hemispheres or []
            if len(hemis) > 1:
                raise SystemExit(
                    "error: gifti takes one --hemisphere, for files that do "
                    "not say which hemisphere they are; the rest is read from "
                    "the files"
                )
            kwargs["hemisphere"] = _hemisphere_name(hemis[0]) if hemis else None
        if fmt.name == "freesurfer":
            if args.geometry is not None:
                kwargs["geometry"] = args.geometry
            kwargs["space"] = args.space
            kwargs["hemispheres"] = args.hemispheres
            kwargs["alternates"] = args.surfaces
            kwargs["morphometry"] = args.morph
            kwargs["annotations"] = args.annots

        try:
            if fmt.name == "edgelist":
                if not args.nodes:
                    raise SystemExit("error: --nodes NODES.csv is required for format 'edgelist'")
                summary = ingest(
                    str(args.input), str(args.nodes), str(args.output),
                    tuple(args.chunk_shape), bin_shape=args.bin_shape, dtype=args.dtype,
                )
            else:
                summary = ingest(
                    str(args.input), str(args.output),
                    tuple(args.chunk_shape), **kwargs,
                )
        except ImportError as exc:  # a heavy reader dep was missing at call time
            hint = (
                f" — install it with: pip install 'zarr-vectors-tools[{fmt.extra}]'"
                if fmt.extra else ""
            )
            raise SystemExit(f"error: {fmt.name} ingest failed ({exc}){hint}")

        _print_summary(f"ingested {fmt.name} ({fmt.geometry})", summary)

        if factors is not None:
            _build_pyramid_post(args, factors, chunk_scale)

    # Sharding is a single-process post-pass (safe: parallel workers can't
    # co-write a shard file).  Applies to every level's per-chunk arrays.
    _maybe_shard(getattr(args, "shard", None), args.output)
    return 0


def _hemisphere_name(value: str) -> str:
    return {"lh": "left", "rh": "right"}.get(value.lower(), value.lower())


def _maybe_shard(shard_shape, output) -> None:
    """Pack per-chunk cells into shards after the store is fully written.

    ``shard_shape`` is the outer-chunk size in *inner-chunk* (spatial-chunk)
    units — an int broadcasts to every axis (``8`` → ``8×8×8`` ≈ 512
    chunks/shard).  ``None`` leaves the store unsharded (one file per chunk).
    """
    if shard_shape is None:
        return
    from zarr_vectors.building import shard_store

    print(f"sharding store (shard_shape={shard_shape}) ...")
    stats = shard_store(str(output), shard_shape=shard_shape)
    print(
        f"  packed {stats['chunks_packed']} chunks across "
        f"{stats['arrays_sharded']} arrays into shards of {stats['shard_shape']}"
    )


def run_shard(args) -> int:
    """``zvtools shard`` — (re)shard or unshard an existing store."""
    from zarr_vectors.building import reshard

    store = str(args.store)
    shape = None if args.unshard else args.shard_shape
    verb = "unsharding" if shape is None else f"sharding (shard_shape={shape})"
    print(f"{verb} {store} ...")
    stats = reshard(store, shape)
    print("  " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    return 0
