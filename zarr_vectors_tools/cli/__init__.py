"""``zvtools`` — flag-based terminal CLI for zarr-vectors-tools.

Subcommands:
    convert   Ingest a file into a new zarr-vectors store (+ optional pyramid).
    merge     Add a file's or another store's objects to an existing store.
    split     Cut a store into one store per group, label or merged source.
    pyramid   Build a sparsity pyramid on an existing store.
    validate  Run core conformance validation on a store.
    info      Print a store's geometry, resolution levels, and metadata.
    attach    Stage a table's columns (or CIFTI maps) onto a store's vertices.
    shard     Repack per-chunk cells into shards, or undo it.
    bundles   Summarise each group of a streamline store.
    synapses  Join a synapse table to an EM skeleton store by segment id.
    run       Run a recipe: several of the above in order, resumable.

Also runnable as ``python -m zarr_vectors_tools``.
"""

from __future__ import annotations

import argparse
import sys

from zarr_vectors_tools.convert.ingest.attach import DEFAULT_KEY_ATTRIBUTE

from . import attach as _attach
from . import compose as _compose
from . import convert as _convert
from . import pipeline as _pipeline
from . import pyramid as _pyramid
from ._args import (
    EXPORT_REGISTRY,
    FORMAT_REGISTRY,
    parse_float_list,
    parse_int_list,
    parse_num_chunks,
    parse_shape,
    parse_str_list,
)

#: What ``convert --format`` accepts.  The union of both directions, because
#: the same command does both and the name means the same thing either way --
#: ``trk`` is TRK whether it is being read or written.  Sorted so the help
#: text does not depend on dict order.
_CONVERT_FORMATS = tuple(sorted(set(FORMAT_REGISTRY) | set(EXPORT_REGISTRY)))

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
    g.add_argument(
        "--rdp-tolerance", type=parse_float_list, dest="rdp_tolerance",
        default=None, metavar="T1,T2,...",
        help="streamlines, rdp mode: per-level Douglas-Peucker tolerance, the "
             "furthest a level may stray from the level below, in store units "
             "(e.g. mm). One per --coarsen entry. Default: half the smallest "
             "edge of each level's bin",
    )
    # Shared by convert and pyramid.  On a point, graph or line store these
    # links are most of the build: a 300k-point pyramid took 80 s with the
    # default explicit links and 5 s with none.
    g.add_argument(
        "--cross-level-storage", dest="cross_level_storage",
        choices=("none", "implicit", "explicit"), default=None,
        help="links between pyramid levels: 'explicit' writes both directions "
             "(the default when omitted), 'implicit' fine-to-coarse only, "
             "'none' skips them",
    )
    g.add_argument(
        "--cross-level-depth", dest="cross_level_depth", type=int, default=None,
        metavar="N",
        help="largest level gap to link (default 1; -1 = every pair of levels)",
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
        "convert",
        help="file -> store (ingest), or store -> file (export)",
        description=(
            "Move data between a file and a zarr-vectors store. The direction "
            "comes from INPUT: a file is ingested into the store at OUTPUT, a "
            "store is exported to the file at OUTPUT (format from its "
            "extension)."
        ),
    )
    c.add_argument(
        "input",
        help="file to ingest (or a precomputed layer's directory or URL), or "
             "store to export (decides the direction)",
    )
    c.add_argument(
        "output",
        help="store to write (ingest), or file to write (export)",
    )
    c.add_argument(
        "--format", choices=("auto", *_CONVERT_FORMATS), default="auto",
        help=(
            "format to use, instead of guessing from the extension: on ingest "
            "the INPUT's, on export the OUTPUT's (default: auto)"
        ),
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
                        "chunk) -- far fewer files for cloud upload. 8 = 8x8x8 "
                        "~512 chunks/shard. Omit to leave unsharded")
    c.add_argument("--workers", type=int, default=None,
                   help="parallel worker processes (default backend needs no extra)")
    c.add_argument("--workers-backend", dest="workers_backend",
                   choices=("process", "dask"), default="process",
                   help="parallel backend: 'process' (stdlib, default) or "
                        "'dask' (needs the [parallel] extra)")
    c.add_argument("--n-parts", type=int, dest="n_parts", default=None,
                   help="trk: file-split granularity")
    c.add_argument("--scratch-dir", dest="scratch_dir", default=None, metavar="DIR",
                   help="trk: keep the intermediate part files and a progress "
                        "record here instead of a temporary directory")
    c.add_argument("--resume", action="store_true",
                   help="trk: reuse what an earlier run with the same options "
                        "finished in --scratch-dir (the file scan, Phase A, "
                        "level 0, finished pyramid levels)")
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
                        "length, endpoints, orientation (start->end unit vector, "
                        "3ch DEC), tortuosity, vertex_count")
    c.add_argument("--vertex-attr", action="append", dest="vertex_attrs",
                   default=None, metavar="NAME",
                   choices=("arc_length", "x", "y", "z", "random",
                            "index", "tangent"),
                   help="trk: generate a per-vertex (per-point) attribute for "
                        "color-by-vertex testing (repeatable). Choices: "
                        "arc_length (0->1 along each streamline), x/y/z "
                        "(coordinate), random, index (0->1 within streamline), "
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
                        "auto -- tries spatial, X_spatial, spatial_fov, "
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
                   help="table: column delimiter (default: ,), and CSV export")

    sf = c.add_argument_group("cortical surfaces (gifti, freesurfer)")
    sf.add_argument(
        "--geometry", default=None, metavar="SURFACE",
        help="surface to chunk: midthickness, pial, white, smoothwm, orig "
             "(default: midthickness; gifti falls back to the first of those "
             "it finds). Every other surface is kept as coords_<name>",
    )
    sf.add_argument(
        "--hemisphere", action="append", dest="hemispheres", default=None,
        choices=("left", "right", "lh", "rh"),
        help="freesurfer: hemispheres to read (repeatable; default both). "
             "gifti: the hemisphere of files that do not name one. gifti "
             "export: which hemispheres to write",
    )
    sf.add_argument(
        "--space", choices=("auto", "scanner", "surface"), default="auto",
        help="freesurfer: scanner RAS (add c_ras, lines up with volumes and "
             "tracts) or FreeSurfer surface RAS. auto = scanner when the "
             "files record c_ras (default: auto)",
    )
    sf.add_argument(
        "--surface", action="append", dest="surfaces", default=None,
        metavar="NAME",
        help="freesurfer: other surfaces to keep (repeatable; default white, "
             "pial, inflated, sphere when present). gifti export: which "
             "surfaces to write (default all)",
    )
    sf.add_argument(
        "--morph", action="append", dest="morph", default=None, metavar="NAME",
        help="freesurfer: morphometry maps from surf/ (repeatable; default "
             "thickness, curv, sulc, area when present)",
    )
    sf.add_argument(
        "--annot", action="append", dest="annots", default=None, metavar="NAME",
        help="freesurfer: parcellations from label/ (repeatable; default "
             "aparc when present), e.g. aparc.a2009s",
    )

    pc = c.add_argument_group(
        "precomputed (Neuroglancer skeleton or mesh layer)",
        "A skeleton layer with a spatial index keeps its own chunk grid, so "
        "--chunk-shape is left out; one without needs --chunk-shape, in nm, "
        "and --coarsen gives each level's decimation stride. A mesh layer "
        "(the mesh directory, not the segmentation above it) needs "
        "--chunk-shape too, and builds its pyramid like any mesh input.",
    )
    pc.add_argument(
        "--anchor", type=parse_int_list, default=None, metavar="X,Y,Z",
        help="spatial index: voxel corner of one .frags chunk; ingest the "
             "block from there instead of every chunk in the layer",
    )
    pc.add_argument(
        "--counts", type=parse_int_list, default=None, metavar="NX,NY,NZ",
        help="spatial index: .frags chunks per axis from --anchor "
             "(default: 1,1,1)",
    )
    pc.add_argument(
        "--frags-dir", dest="frags_dir", default="", metavar="DIR",
        help="spatial index: subdirectory holding the .frags files "
             "(default: the layer root)",
    )
    pc.add_argument(
        "--segment-id", action="append", type=int, dest="segment_ids",
        default=None, metavar="ID",
        help="no spatial index, or a mesh layer: ingest only this segment "
             "(repeatable; default: every segment in the layer)",
    )
    pc.add_argument(
        "--lod", type=int, default=None, metavar="N",
        help="multi-resolution mesh layer: level of detail to read "
             "(default: 0, the finest)",
    )
    pc.add_argument(
        "--drop-interior-below", type=int, dest="drop_interior_below",
        default=0, metavar="N",
        help="at each coarser level, drop objects of at most N vertices that "
             "touch no chunk boundary (default: 0, keep all)",
    )

    e = c.add_argument_group("export (store -> file)")
    e.add_argument(
        "--level", type=int, default=0,
        help="resolution level to export (default: 0, the finest)",
    )
    e.add_argument(
        "--object-id", action="append", type=int, dest="export_object_ids",
        default=None, metavar="ID",
        help="export only these objects (repeatable)",
    )
    e.add_argument(
        "--group-id", action="append", type=int, dest="export_group_ids",
        default=None, metavar="ID",
        help="streamlines: export only these groups/bundles (repeatable)",
    )
    e.add_argument(
        "--bbox", type=parse_float_list, dest="export_bbox", default=None,
        metavar="X0,Y0,Z0,X1,Y1,Z1",
        help="export only what falls in this box",
    )
    e.add_argument(
        "--attribute", action="append", dest="export_attributes",
        default=None, metavar="NAME",
        help="per-vertex attributes to include (repeatable). csv/ply/h5ad "
             "write none by default; trk/trx write every numeric one unless "
             "this names them",
    )
    e.add_argument(
        "--object-attribute", action="append", dest="export_object_attributes",
        default=None, metavar="NAME",
        help="trk/trx: per-streamline attributes to include (repeatable; "
             "default: every numeric one)",
    )
    e.add_argument(
        "--unit", dest="export_unit", default=None, metavar="UNIT",
        help="precomputed: the store's coordinate unit (nanometer, micrometer, "
             "millimeter) when its metadata does not record one",
    )
    e.add_argument(
        "--prefix", dest="export_prefix", default=None, metavar="TEXT",
        help="gifti: leading file-name part, e.g. sub-01 gives "
             "sub-01_hemi-L_pial.surf.gii",
    )
    c.set_defaults(func=_convert.run)

    # ---- pyramid -----------------------------------------------------------
    p = sub.add_parser(
        "pyramid", help="build a sparsity pyramid on an existing store",
        description="Build coarser resolution levels on an existing store (auto geometry routing).",
    )
    p.add_argument("store", help="existing zarr-vectors store path")
    _add_pyramid_args(p)
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

    # ---- synapses ----------------------------------------------------------
    sy = sub.add_parser(
        "synapses", help="join a synapse table to an EM skeleton store by segment id",
        description="Write a CAVE-style synapse table as a point store whose object "
                    "ids are the skeleton store's, and count each neuron's pre- and "
                    "post-synaptic sites onto it.",
    )
    sy.add_argument("table", help="synapse table: .csv, .tsv, or .parquet with pyarrow")
    sy.add_argument("skeletons", help="skeleton store with per-object segment ids")
    sy.add_argument("output", help="point store to write")
    sy.add_argument("--side", choices=("post", "pre"), default="post",
                    help="whose segment owns each synapse (default: post)")
    sy.add_argument("--position", default="ctr_pt_position", metavar="COLUMN",
                    help="position column, '[x y z]' strings or COLUMN_x/_y/_z "
                         "(default: ctr_pt_position)")
    sy.add_argument("--resolution", type=parse_float_list, default=[1.0, 1.0, 1.0],
                    metavar="X,Y,Z", help="nanometres per voxel of the positions, "
                                          "e.g. 4,4,40 (default: 1,1,1)")
    sy.add_argument("--chunk-shape", type=parse_shape, dest="chunk_shape", default=None,
                    metavar="X,Y,Z", help="chunk size in nm (default: the skeleton store's)")
    sy.add_argument("--column", action="append", dest="columns", default=None,
                    metavar="NAME", help="further numeric column to keep (repeatable)")
    sy.add_argument("--unmatched", choices=("keep", "drop", "error"), default="keep",
                    help="synapses whose segment the skeleton store lacks (default: keep "
                         "as one extra object)")
    sy.add_argument("--no-counts", action="store_true", dest="no_counts",
                    help="do not write synapse counts to the skeleton store")
    sy.set_defaults(func=_convert.run_synapses)

    # ---- bundles -----------------------------------------------------------
    b = sub.add_parser(
        "bundles", help="summarise each object group of a streamline store",
        description="One row per group: streamline count, length statistics, "
                    "mean tortuosity and endpoint centroids, written to the "
                    "store as group attributes unless --no-write.",
    )
    b.add_argument("store", help="zarr-vectors streamline store path")
    b.add_argument("--level", type=int, default=None,
                   help="summarise this level from its own geometry and write it "
                        "there only (default: level 0, written to every level)")
    b.add_argument("--no-write", action="store_true", dest="no_write",
                   help="print the table without storing it")
    b.add_argument("--as-stored", action="store_true", dest="as_stored",
                   help="average endpoints as stored instead of orienting each bundle")
    b.add_argument("--csv", default=None, metavar="FILE",
                   help="also write the table to this CSV file")
    b.set_defaults(func=_pyramid.run_bundles)

    # ---- attach ------------------------------------------------------------
    a = sub.add_parser(
        "attach", help="stage a file's columns into an existing store",
        description="Add per-vertex attributes to an existing store, matching "
                    "rows to points on a join key. Lets a dataset split across "
                    "several files be imported one file at a time instead of "
                    "being merged into a single large file first.",
    )
    a.add_argument("store", help="existing zarr-vectors store path (modified in place)")
    a.add_argument("input", help="file to stage in (.h5ad, a delimited table, or "
                                 "a .dscalar/.dlabel/.dtseries.nii onto a surface store)")
    a.add_argument("--format", choices=("auto", "h5ad", "table", "cifti"), default="auto",
                   help="source format (default: auto from extension)")
    a.add_argument("--name", default=None, metavar="NAME",
                   help="cifti only: attribute name (default: from the filename). "
                        "On a multi-map dscalar, keeps the maps together as one "
                        "multi-column attribute")
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
                   help="print the plan -- ids, offsets, whether it fits -- and stop")
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

    # ---- run ---------------------------------------------------------------
    r = sub.add_parser(
        "run", help="run a recipe of zvtools steps in order, resumably",
        description="Run the steps a recipe lists (convert, pyramid, attach, "
                    "merge, split, validate, info, shard), each with the same "
                    "options as its own command. Progress is kept next to the "
                    "recipe, so a rerun skips the steps that already finished.",
    )
    r.add_argument("recipe", help="recipe file: .yaml, .toml or .json")
    r.add_argument("--force", action="store_true",
                   help="rerun every step, even ones that finished")
    r.add_argument("--from", type=int, dest="start", default=None, metavar="N",
                   help="rerun from step N (1-based) onward")
    r.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="print each step's zvtools command, and stop")
    r.set_defaults(func=_pipeline.run)

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
    except Exception as exc:  # ingest/coarsen/validate errors -> clean message
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
