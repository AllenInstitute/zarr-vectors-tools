"""``zvtools`` — flag-based terminal CLI for zarr-vectors-tools.

Subcommands:
    convert   Ingest a file into a new zarr-vectors store (+ optional pyramid).
    pyramid   Build a sparsity pyramid on an existing store.
    validate  Run core conformance validation on a store.
    info      Print a store's geometry, resolution levels, and metadata.

Also runnable as ``python -m zarr_vectors_tools``.
"""

from __future__ import annotations

import argparse
import sys

from . import convert as _convert
from . import pyramid as _pyramid
from ._args import (
    FORMAT_REGISTRY,
    parse_float_list,
    parse_int_list,
    parse_num_chunks,
    parse_shape,
)

_SPARSITY_STRATEGIES = (
    "random", "length", "spatial_coverage", "attribute", "point_thinning",
)


def _pkg_version() -> str:
    try:
        from importlib.metadata import version

        return version("zarr-vectors-tools")
    except Exception:
        return "unknown"


def _add_pyramid_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("pyramid (coarser levels)")
    g.add_argument(
        "--coarsen", type=parse_float_list, default=None, metavar="C1,C2,...",
        help="per-level vertex coarsen factor / decimation stride",
    )
    g.add_argument(
        "--sparsity", type=parse_float_list, default=None, metavar="S1,S2,...",
        help="per-level object-drop divisor (1=keep all, 2=half, 8=1/8, ...)",
    )
    g.add_argument(
        "--chunk-scale", type=parse_int_list, dest="chunk_scale", default=None,
        metavar="K1,K2,...", help="per-level chunk-size multiplier",
    )
    g.add_argument(
        "--sparsity-strategy", dest="sparsity_strategy",
        choices=_SPARSITY_STRATEGIES, default="random",
        help="which objects survive sparsification (default: random)",
    )
    g.add_argument(
        "--coarsen-mode", dest="coarsen_mode", choices=("rdp", "decimate"),
        default="rdp", help="streamline/polyline vertex reduction (default: rdp)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zvtools",
        description="Convert files into zarr-vectors stores with optional sparsity pyramids.",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"zvtools (zarr-vectors-tools {_pkg_version()})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- convert -----------------------------------------------------------
    c = sub.add_parser(
        "convert", help="ingest a file into a new store (+ optional pyramid)",
        description="Ingest INPUT into a new zarr-vectors store at OUTPUT.",
    )
    c.add_argument("input", help="input file (format auto-detected from extension)")
    c.add_argument("output", help="output .zarrvectors store path")
    c.add_argument(
        "--format", choices=("auto", *FORMAT_REGISTRY), default="auto",
        help="input format (default: auto from extension)",
    )
    c.add_argument(
        "--chunk-shape", type=parse_shape, dest="chunk_shape", default=None,
        metavar="X,Y,Z", help="level-0 spatial chunk size (required for non-trk formats)",
    )
    c.add_argument(
        "--num-chunks", type=parse_num_chunks, dest="num_chunks", default=None,
        metavar="N|X,Y,Z", help="trk only: target total chunk count, or per-axis counts",
    )
    c.add_argument(
        "--bin-shape", type=parse_shape, dest="bin_shape", default=None,
        metavar="X,Y,Z", help="optional intra-chunk sub-binning",
    )
    c.add_argument("--dtype", default="float32", help="stored position dtype (default: float32)")
    c.add_argument("--overwrite", action="store_true",
                   help="replace an existing output store")
    _add_pyramid_args(c)
    c.add_argument("--compressor", choices=("none", "zstd", "blosc"),
                   default="none",
                   help="codec for per-chunk arrays (default: none = raw). "
                        "zstd/blosc roughly halve streamline stores (~2.4x on "
                        "HCP tracts) at the cost of a slower write path")
    c.add_argument("--shard", type=parse_num_chunks, dest="shard", default=None,
                   metavar="N|X,Y,Z",
                   help="after conversion, pack per-chunk cells into shards of N "
                        "chunks per axis (one file per shard instead of one per "
                        "chunk) — far fewer files for cloud upload. 8 = 8x8x8 "
                        "~512 chunks/shard. Omit to leave unsharded")
    c.add_argument("--workers", type=int, default=None,
                   help="parallel worker processes (default backend needs no extra)")
    c.add_argument("--workers-backend", dest="workers_backend",
                   choices=("process", "dask"), default="process",
                   help="parallel backend: 'process' (stdlib, default) or "
                        "'dask' (needs the [parallel] extra)")
    c.add_argument("--n-parts", type=int, dest="n_parts", default=None,
                   help="trk: file-split granularity")
    c.add_argument("--apply-affine", action="store_true", dest="apply_affine",
                   help="trk only (rejected for other inputs): bake the "
                        "vox_to_ras affine into vertex positions (RAS world "
                        "space) and store an identity crs affine. Simpler for "
                        "viewers that can't apply an affine themselves (e.g. "
                        "neuroglancer). Default: keep raw voxmm coordinates + "
                        "the real affine in crs")
    c.add_argument("--compute-length", action="store_true", dest="compute_length",
                   help="streamlines: store per-object length")
    c.add_argument("--compute-endpoints", action="store_true", dest="compute_endpoints",
                   help="streamlines: store per-object endpoints")
    c.add_argument("--object-attr", action="append", dest="object_attrs",
                   default=None, metavar="NAME",
                   choices=("length", "endpoints", "orientation",
                            "tortuosity", "vertex_count"),
                   help="trk: generate a per-object (per-streamline) attribute "
                        "for color-by-object testing (repeatable). Choices: "
                        "length, endpoints, orientation (start→end unit vector, "
                        "3ch DEC), tortuosity, vertex_count")
    c.add_argument("--vertex-attr", action="append", dest="vertex_attrs",
                   default=None, metavar="NAME",
                   choices=("arc_length", "x", "y", "z", "random",
                            "index", "tangent"),
                   help="trk: generate a per-vertex (per-point) attribute for "
                        "color-by-vertex testing (repeatable). Choices: "
                        "arc_length (0→1 along each streamline), x/y/z "
                        "(coordinate), random, index (0→1 within streamline), "
                        "tangent (per-vertex unit direction, 3ch DEC)")
    c.add_argument("--attr-seed", type=int, dest="attr_seed", default=0,
                   help="seed for the 'random' attribute generators (default: 0)")
    c.add_argument("--nodes", default=None,
                   help="edgelist: path to the node CSV (second input)")
    c.add_argument("--knn-distance-k", type=int, dest="knn_distance_k", default=None,
                   help="points: k for kNN-distance enrichment (needs [points-enrichment])")
    c.set_defaults(func=_convert.run)

    # ---- pyramid -----------------------------------------------------------
    p = sub.add_parser(
        "pyramid", help="build a sparsity pyramid on an existing store",
        description="Build coarser resolution levels on an existing store (auto geometry routing).",
    )
    p.add_argument("store", help="existing zarr-vectors store path")
    _add_pyramid_args(p)
    p.add_argument("--cross-level-storage", dest="cross_level_storage",
                   choices=("none", "implicit", "explicit"), default=None)
    p.add_argument("--cross-level-depth", dest="cross_level_depth", type=int, default=None)
    p.add_argument("--compressor", choices=("none", "zstd", "blosc"),
                   default="none",
                   help="codec for the coarser levels' per-chunk arrays "
                        "(default: none = raw). Match the value level 0 was "
                        "written with to keep the store uniform")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel worker processes (default backend needs no extra)")
    p.add_argument("--workers-backend", dest="workers_backend",
                   choices=("process", "dask"), default="process",
                   help="parallel backend: 'process' (stdlib, default) or "
                        "'dask' (needs the [parallel] extra)")
    p.set_defaults(func=_pyramid.run_pyramid)

    # ---- validate ----------------------------------------------------------
    v = sub.add_parser("validate", help="run conformance validation on a store")
    v.add_argument("store", help="zarr-vectors store path")
    v.add_argument("--level", type=int, default=3, help="conformance level 1-5 (default: 3)")
    v.set_defaults(func=_pyramid.run_validate)

    # ---- info --------------------------------------------------------------
    i = sub.add_parser("info", help="print store geometry, levels, and metadata")
    i.add_argument("store", help="zarr-vectors store path")
    i.set_defaults(func=_pyramid.run_info)

    # ---- shard -------------------------------------------------------------
    s = sub.add_parser(
        "shard", help="(re)shard or unshard an existing store's per-chunk arrays",
        description="Pack per-chunk cells into shards (few large files, good for "
                    "cloud upload) or reverse it. Data and coordinates are "
                    "unchanged; only the on-disk file layout differs.",
    )
    s.add_argument("store", help="zarr-vectors store path")
    s.add_argument("--shape", type=parse_num_chunks, dest="shard_shape", default=8,
                   metavar="N|X,Y,Z",
                   help="shard size in chunks per axis (default 8 = 8x8x8). "
                        "Ignored with --unshard")
    s.add_argument("--unshard", action="store_true",
                   help="remove sharding (back to one file per chunk)")
    s.set_defaults(func=_convert.run_shard)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("aborted", file=sys.stderr)
        return 130
    except Exception as exc:  # ingest/coarsen/validate errors → clean message
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
