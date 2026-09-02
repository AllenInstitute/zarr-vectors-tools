"""``zvtools`` — flag-based terminal CLI for zarr-vectors-tools.

Subcommands:
    convert   Ingest a file into a new zarr-vectors store (+ optional pyramid).
    merge     Add a file's or another store's objects to an existing store.
    split     Cut a store into one store per group, label or merged source.
    pyramid   Build a sparsity pyramid on an existing store.
    validate  Run core conformance validation on a store.
    info      Print a store's geometry, resolution levels, and metadata.
    attach    Stage a table's columns onto a store's existing vertices.
    shard     Repack per-chunk cells into shards, or undo it.

Also runnable as ``python -m zarr_vectors_tools``.
"""

from __future__ import annotations

import argparse
import sys

from zarr_vectors_tools.ingest.attach import DEFAULT_KEY_ATTRIBUTE

from . import attach as _attach
from . import compose as _compose
from . import convert as _convert
from . import pyramid as _pyramid
from ._args import (
    FORMAT_REGISTRY,
    parse_float_list,
    parse_int_list,
    parse_num_chunks,
    parse_shape,
    parse_str_list,
)

_SPARSITY_STRATEGIES = (
    "random", "length", "spatial_coverage", "attribute", "point_thinning",
    "group",
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
    h = c.add_argument_group("h5ad (AnnData)")
    h.add_argument("--spatial-key", dest="spatial_key", default="auto",
                   metavar="KEY",
                   help="h5ad: obsm key holding the coordinates (default: "
                        "auto — tries spatial, X_spatial, spatial_fov, "
                        "X_umap, X_tsne, X_pca)")
    h.add_argument("--spatial-columns", type=parse_int_list,
                   dest="spatial_columns", default=None, metavar="I,J[,K]",
                   help="h5ad: which columns of the embedding to use as "
                        "positions (default: the first up-to-3)")
    h.add_argument("--obs-column", action="append", dest="obs_columns",
                   default=None, metavar="NAME",
                   help="h5ad: obs column to store as a vertex attribute "
                        "(repeatable; default: all of them). Use --no-obs to "
                        "store none")
    h.add_argument("--no-obs", action="store_true", dest="no_obs",
                   help="h5ad: do not store any obs columns")
    h.add_argument("--gene", action="append", dest="genes", default=None,
                   metavar="NAME",
                   help="h5ad: store this gene's expression as a vertex "
                        "attribute (repeatable; default: none)")
    h.add_argument("--layer", dest="layer", default=None, metavar="NAME",
                   help="h5ad: read expression from layers[NAME] instead of X")
    h.add_argument("--object-id-column", dest="object_id_column", default=None,
                   metavar="NAME",
                   help="h5ad: obs column grouping cells into Zarr Vectors objects "
                        "(e.g. cell_type, sample)")
    h.add_argument("--backed", action="store_true", dest="backed",
                   help="h5ad: leave X on disk (AnnData backed mode); helps "
                        "on large files when --gene selects few columns")
    h.add_argument("--drop-na", action="store_true", dest="drop_na",
                   help="h5ad: drop cells whose coordinates contain NaN "
                        "(table input drops them by default)")
    t = c.add_argument_group("table (keyed delimited table)")
    t.add_argument("--position-columns", type=parse_str_list,
                   dest="position_columns", default=None, metavar="X,Y[,Z]",
                   help="table: column names holding the coordinates, in axis "
                        "order (required for --format table)")
    t.add_argument("--key-column", dest="key_column", default=None, metavar="NAME",
                   help="table: column holding the row identifier. Hashed into "
                        "the join key so later files can be staged in with "
                        "'zvtools attach'")
    t.add_argument("--column", action="append", dest="columns", default=None,
                   metavar="NAME",
                   help="table: metadata column to store as a vertex attribute "
                        "(repeatable; default: every non-position, non-key column)")
    t.add_argument("--delimiter", default=",",
                   help="table: column delimiter (default: ,)")
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

    # ---- attach ------------------------------------------------------------
    a = sub.add_parser(
        "attach", help="stage a file's columns into an existing store",
        description="Add per-vertex attributes to an existing store, matching "
                    "rows to points on a join key. Lets a dataset split across "
                    "several files be imported one file at a time instead of "
                    "being merged into a single large file first.",
    )
    a.add_argument("store", help="existing zarr-vectors store path (modified in place)")
    a.add_argument("input", help="file to stage in (.h5ad, or a delimited table)")
    a.add_argument("--format", choices=("auto", "h5ad", "table"), default="auto",
                   help="source format (default: auto from extension)")
    a.add_argument("--column", action="append", dest="columns", default=None,
                   metavar="NAME",
                   help="column to stage in (repeatable). h5ad: an obs column. "
                        "table: any column. Default for tables: every non-key column")
    a.add_argument("--gene", action="append", dest="genes", default=None,
                   metavar="NAME",
                   help="h5ad only: gene whose expression to stage in "
                        "(repeatable). Matched against var_names, then gene_symbol")
    a.add_argument("--gene-by", dest="gene_by", default=None, metavar="COL",
                   help="h5ad only: var column to match --gene against")
    a.add_argument("--layer", dest="layer", default=None, metavar="NAME",
                   help="h5ad only: read expression from layers[NAME] instead of X")
    a.add_argument("--key-column", dest="key_column", default=None, metavar="NAME",
                   help="column holding the identifier to join on. Required for "
                        "tables; for h5ad defaults to the obs index")
    a.add_argument("--key-attribute", dest="key_attribute",
                   default=DEFAULT_KEY_ATTRIBUTE, metavar="NAME",
                   help=f"store attribute holding the join key "
                        f"(default: {DEFAULT_KEY_ATTRIBUTE})")
    a.add_argument("--delimiter", default=",",
                   help="table only: column delimiter (default: ,)")
    a.add_argument("--level", type=int, default=0,
                   help="resolution level to write into (default: 0)")
    a.add_argument("--missing", choices=("fill", "error"), default="fill",
                   help="what to do for points absent from the incoming file "
                        "(default: fill with NaN / -1)")
    a.add_argument("--overwrite", action="store_true",
                   help="replace attributes that already exist")
    a.add_argument("--shard", type=parse_num_chunks, dest="shard", default=None,
                   metavar="N|X,Y,Z",
                   help="create the new attribute arrays sharded, packing N "
                        "chunks per axis into one file. Set this when staging "
                        "many columns: unsharded costs one file per chunk per "
                        "attribute, and resharding afterwards has to reread "
                        "every one. Match the value across attaches")
    a.set_defaults(func=_attach.run_attach)

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

    # ---- merge -------------------------------------------------------------
    m = sub.add_parser(
        "merge", help="add a file's or another store's objects to an existing store",
        description="Append objects to a store that already exists, carrying "
                    "their attributes, group taxonomy and format headers, and "
                    "rebuilding the pyramid the write invalidates. Unlike "
                    "convert, nothing already in the target is replaced.",
    )
    m.add_argument("target", help="store to merge into")
    m.add_argument("sources", nargs="+",
                   help="files and/or stores to merge in")
    m.add_argument("--format", default=None, choices=(*FORMAT_REGISTRY, "auto"),
                   help="input format for file sources (default: from extension)")
    m.add_argument("--create", action="store_true",
                   help="create the target if absent, sized from the sources")
    m.add_argument("--cell-size", type=parse_shape, dest="cell_size", default=None,
                   metavar="X,Y,Z", help="grid cell size for a created target")
    m.add_argument("--space", default="voxmm", choices=("voxmm", "ras"),
                   help="trk only: keep stored coordinates (voxmm, default) or "
                        "apply the header affine to reach RAS mm")
    m.add_argument("--transform", default=None, metavar="FILE|CSV",
                   help="affine to apply to every source, as a .npy, a .json, "
                        "or comma-separated numbers in row-major order")
    m.add_argument("--group-by", dest="group_by", default=None, metavar="ATTR",
                   help="turn this per-object attribute into named groups")
    m.add_argument("--lut", default=None, metavar="FILE",
                   help="JSON {name: code} lookup naming the --group-by values")
    m.add_argument("--group-prefix", dest="group_prefix", default=None,
                   help="namespace incoming group names (default: same name "
                        "extends the existing group)")
    m.add_argument("--source-attr", dest="source_attr", default=None, metavar="NAME",
                   help="stamp a per-object attribute with each source's index")
    m.add_argument("--on-out-of-bounds", dest="on_out_of_bounds", default="raise",
                   choices=("raise", "skip", "expand"),
                   help="geometry outside the target's grid: refuse (default), "
                        "drop it, or grow the grid upwards to fit")
    m.add_argument("--pyramid", default="rebuild", choices=("rebuild", "drop", "keep"),
                   help="what to do with the coarse levels a merge invalidates "
                        "(default: rebuild)")
    m.add_argument("--pyramid-coarsen", type=parse_float_list,
                   dest="pyramid_coarsen", default=None, metavar="C1,C2,...",
                   help="per-level coarsen factors for the rebuild (default: "
                        "inferred from the existing levels)")
    m.add_argument("--pyramid-sparsity", type=parse_float_list,
                   dest="pyramid_sparsity", default=None, metavar="S1,S2,...",
                   help="per-level sparsity divisors for the rebuild")
    m.add_argument("--sparsity-strategy", dest="sparsity_strategy",
                   choices=_SPARSITY_STRATEGIES, default="random",
                   help="which objects survive sparsification (default: "
                        "random). 'group' thins each named group by the same "
                        "factor with a floor of one, so no group is ever lost")
    m.add_argument("--coarsen-mode", dest="coarsen_mode",
                   choices=("rdp", "decimate"), default="rdp",
                   help="streamline/polyline vertex reduction (default: rdp). "
                        "Note rdp's tolerance is min(chunk_shape)*0.5*factor, "
                        "so it does NOT compound across levels unless "
                        "--chunk-scale grows the cells too; 'decimate' strides "
                        "do compound")
    m.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="print the plan — ids, offsets, whether it fits — and stop")
    m.set_defaults(func=_compose.run_merge)

    # ---- split -------------------------------------------------------------
    sp = sub.add_parser(
        "split", help="cut a store into one store per group, label or source",
        description="The inverse of merge. Each part keeps the parent's grid by "
                    "default, so the pieces stay cell-aligned with it and with "
                    "each other.",
    )
    sp.add_argument("store", help="store to split")
    sp.add_argument("output", help="directory to write the parts into")
    sp.add_argument("--by", default="groups",
                    choices=("groups", "attribute", "objects", "provenance"),
                    help="how to cut (default: the store's named object groups)")
    sp.add_argument("--attribute", default=None, metavar="NAME",
                    help="per-object attribute to cut on, for --by attribute")
    sp.add_argument("--lut", default=None, metavar="FILE",
                    help="JSON {name: code} lookup naming the attribute values")
    sp.add_argument("--level", type=int, default=0,
                    help="resolution level to read from (default: 0)")
    sp.add_argument("--bounds", default="source", choices=("source", "fit"),
                    help="keep the parent's grid (default) or size each part "
                         "to its own contents")
    sp.add_argument("--pyramid", default="drop", choices=("rebuild", "drop", "keep"),
                    help="pyramid for each part (default: drop)")
    sp.add_argument("--min-objects", type=int, dest="min_objects", default=1,
                    metavar="N", help="skip parts with fewer than N objects")
    sp.add_argument("--overwrite", action="store_true",
                    help="replace parts that already exist")
    sp.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="print the parts and their sizes, and stop")
    sp.set_defaults(func=_compose.run_split)

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
