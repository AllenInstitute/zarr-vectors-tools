"""Shared argument helpers and the format registry for the ``zvtools`` CLI."""

from __future__ import annotations

import argparse
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

# ===================================================================
# Comma-list parsers (for argparse ``type=``)
# ===================================================================

def parse_float_list(s: str) -> list[float]:
    """``"8,2,2"`` -> ``[8.0, 2.0, 2.0]``."""
    try:
        return [float(x) for x in s.split(",") if x.strip() != ""]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {s!r}") from exc


def parse_int_list(s: str) -> list[int]:
    """``"2,2,2"`` -> ``[2, 2, 2]``."""
    try:
        return [int(x) for x in s.split(",") if x.strip() != ""]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {s!r}") from exc


def parse_str_list(s: str) -> list[str]:
    """``"x,y,z"`` -> ``["x", "y", "z"]`` (column names)."""
    names = [x.strip() for x in s.split(",") if x.strip() != ""]
    if not names:
        raise argparse.ArgumentTypeError("expected at least one column name")
    return names


def parse_shape(s: str) -> tuple[float, ...]:
    """``"100,100,100"`` -> ``(100.0, 100.0, 100.0)`` (spatial chunk/bin size)."""
    vals = parse_float_list(s)
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one value")
    return tuple(vals)


def parse_num_chunks(s: str) -> int | tuple[int, ...]:
    """``"5000"`` -> ``5000`` (total) or ``"5,13,8"`` -> ``(5, 13, 8)`` (per-axis)."""
    vals = parse_int_list(s)
    if len(vals) == 1:
        return vals[0]
    return tuple(vals)


def build_factors(
    coarsen: list[float] | None,
    sparsity: list[float] | None,
) -> list[tuple[float, float]] | None:
    """Zip ``--coarsen``/``--sparsity`` into per-level ``(coarsen, sparsity)`` tuples.

    Returns ``None`` when neither is given (no pyramid requested). Raises when
    only one is given or their lengths differ.
    """
    if not coarsen and not sparsity:
        return None
    coarsen = coarsen or []
    sparsity = sparsity or []
    if len(coarsen) != len(sparsity):
        raise SystemExit(
            f"error: --coarsen has {len(coarsen)} entries but --sparsity has "
            f"{len(sparsity)}; they must match (one per coarser pyramid level)"
        )
    return [(float(c), float(s)) for c, s in zip(coarsen, sparsity)]


def check_rdp_tolerances(
    tolerances: list[float] | None,
    factors: list[tuple[float, float]] | None,
    coarsen_mode: str,
) -> list[float] | None:
    """Check ``--rdp-tolerance`` against the pyramid it is meant to shape.

    Everything that can be decided without the store: a pyramid to apply to,
    one tolerance per ``--coarsen`` entry, positive finite distances, and rdp
    mode.  Run before any ingest, so a bad list does not cost a conversion.
    Whether the store's geometry has a tolerance is checked by the caller,
    which knows the format or has the store.
    """
    if tolerances is None:
        return None
    if factors is None:
        raise SystemExit(
            "error: --rdp-tolerance sets each coarser level's tolerance, but "
            "no pyramid was requested; pass --coarsen and --sparsity too"
        )
    if coarsen_mode == "decimate":
        raise SystemExit(
            "error: --rdp-tolerance does not apply with --coarsen-mode "
            "decimate, which thins by a stride and has no distance tolerance"
        )
    if len(tolerances) != len(factors):
        raise SystemExit(
            f"error: --rdp-tolerance has {len(tolerances)} entries but "
            f"--coarsen has {len(factors)}; they must match (one per coarser "
            f"pyramid level)"
        )
    bad = [t for t in tolerances if not math.isfinite(t) or t <= 0.0]
    if bad:
        raise SystemExit(
            f"error: --rdp-tolerance values must be distances > 0 in store "
            f"units; got {', '.join(repr(t) for t in bad)}"
        )
    return [float(t) for t in tolerances]


@contextmanager
def executor_ctx(workers: int | None, backend: str = "process"):
    """Yield a parallel executor when ``workers > 1``, else ``None``.

    ``backend="process"`` (default) uses the stdlib process pool — no extra
    dependency.  ``backend="dask"`` uses a local Dask cluster and needs the
    ``parallel`` extra (it adds ``scatter`` broadcast of the shared payload,
    which helps dense levels with a large shared object).
    """
    if workers and workers > 1:
        if backend == "dask":
            from zarr_vectors_tools.convert.ingest._parallel import dask_executor

            with dask_executor(workers) as ex:
                yield ex
        else:
            from zarr_vectors_tools.convert.ingest._parallel import process_pool_executor

            with process_pool_executor(workers) as ex:
                yield ex
    else:
        yield None


# ===================================================================
# Format registry
# ===================================================================

def looks_like_store(path: str | Path) -> bool:
    """Is this a zarr-vectors store rather than a file to ingest?

    What decides the DIRECTION of ``zvtools convert``: a store on the input
    side means the user is exporting out of it.  The test is the same one
    ``--overwrite`` uses to decide whether a directory is safe to replace --
    a directory carrying Zarr's own metadata file -- so the two cannot
    disagree about what a store is.
    """
    p = Path(path)
    if not p.is_dir():
        return False
    return (p / "zarr.json").exists() or (p / ".zattrs").exists()


@dataclass(frozen=True)
class Fmt:
    """One convertible file format."""

    name: str
    exts: tuple[str, ...]          # extensions that auto-detect to this format
    module: str                    # zarr_vectors_tools.convert.ingest.<module>
    func: str                      # entry function name
    extra: str | None              # optional-dependency extra needed (for hints)
    geometry: str                  # what it produces (for messages)
    inline_pyramid: bool = False   # builds the pyramid inside the ingest call


FORMAT_REGISTRY: dict[str, Fmt] = {
    "trk":      Fmt("trk", (".trk",), "trk_parallel", "ingest_trk_parallel",
                    "parallel", "streamlines", inline_pyramid=True),
    "trx":      Fmt("trx", (".trx",), "trx", "ingest_trx", "streamlines", "streamlines"),
    "tck":      Fmt("tck", (".tck",), "tck", "ingest_tck", "streamlines", "streamlines"),
    "swc":      Fmt("swc", (".swc",), "swc", "ingest_swc", None, "skeleton"),
    "obj":      Fmt("obj", (".obj",), "obj", "ingest_obj", None, "mesh"),
    "stl":      Fmt("stl", (".stl",), "stl", "ingest_stl", None, "mesh"),
    "ply":      Fmt("ply", (".ply",), "ply", "ingest_ply", "ply", "points"),
    "las":      Fmt("las", (".las", ".laz"), "las", "ingest_las", "las", "points"),
    "csv":      Fmt("csv", (".csv", ".xyz"), "csv_points", "ingest_csv", None, "points"),
    "h5ad":     Fmt("h5ad", (".h5ad",), "h5ad", "ingest_h5ad", "h5ad", "points"),
    # No extension of its own: ``.csv`` auto-detects to the numeric-only
    # ``csv`` ingester, so the keyed/mixed-type table path is opt-in via
    # ``--format table``.
    "table":    Fmt("table", (), "cell_table", "ingest_table", None, "points"),
    "lines":    Fmt("lines", (), "lines", "ingest_lines_csv", None, "lines"),
    "edgelist": Fmt("edgelist", (), "edgelist", "ingest_edgelist", "graph", "graph"),
    "graphml":  Fmt("graphml", (".graphml",), "graphml", "ingest_graphml", "graph", "graph"),
    # Cortical surfaces.  A subject is a SET of files, so both also accept a
    # directory: resolve_format recognises a FreeSurfer subject (surf/lh.white)
    # and a directory of .gii files without --format.
    "gifti":    Fmt("gifti", (".gii",), "gifti", "ingest_gifti", "surfaces", "surface"),
    "freesurfer": Fmt("freesurfer", (), "freesurfer", "ingest_freesurfer",
                      "surfaces", "surface"),
    # A Neuroglancer precomputed skeleton layer is a directory or a bucket
    # prefix, never a file: resolve_format recognises a URL, or a directory
    # holding an ``info`` file, without --format.
    "precomputed": Fmt("precomputed", (), "precomputed", "ingest_precomputed",
                       "precomputed", "skeleton", inline_pyramid=True),
}

# extension -> format name (only unambiguous extensions; .csv defaults to points)
_EXT_TO_FORMAT: dict[str, str] = {
    ext: fmt.name for fmt in FORMAT_REGISTRY.values() for ext in fmt.exts
}


@dataclass(frozen=True)
class ExportFmt:
    """One format a store can be written back out to.

    ``accepts`` names the exporter's own optional parameters, so the CLI can
    pass only what a given exporter takes and refuse the rest by name rather
    than dropping it silently.
    """

    name: str
    exts: tuple[str, ...]
    module: str                    # zarr_vectors_tools.convert.export.<module>
    func: str
    extra: str | None              # optional-dependency extra, for hints
    geometry: str                  # what it expects to find in the store
    accepts: frozenset[str]


#: Every exporter the package has.  The names match the ingest registry's
#: wherever both directions exist, so ``--format trk`` means TRK either way.
_COMMON = frozenset({"level", "chunks"})

EXPORT_REGISTRY: dict[str, ExportFmt] = {
    "trk": ExportFmt(
        "trk", (".trk",), "trk", "export_trk", "trk", "streamlines",
        _COMMON | {
            "object_ids", "group_ids", "affine",
            "attribute_names", "object_attribute_names",
        },
    ),
    "trx": ExportFmt(
        "trx", (".trx",), "trx", "export_trx", "trx", "streamlines",
        _COMMON | {"object_ids", "group_ids", "attribute_names", "object_attribute_names"},
    ),
    # With object_ids the output is a directory of one .swc per object, so
    # it has no extension to resolve from: pass --format swc.
    "swc": ExportFmt(
        "swc", (".swc",), "swc", "export_swc", None, "skeleton",
        _COMMON | {"object_ids"},
    ),
    # A surface store goes out as a DIRECTORY of .gii files (one per surface
    # and per map, per hemisphere), so it has no extension to resolve from:
    # pass --format gifti, or give a surface store an extensionless OUTPUT.
    # Not _COMMON: this exporter takes no chunks.
    "gifti": ExportFmt(
        "gifti", (), "gifti", "export_gifti", "surfaces", "surface",
        frozenset({"level", "hemispheres", "surfaces", "attribute_names", "prefix"}),
    ),
    # A precomputed layer is a directory or bucket prefix, never a file, so it
    # has no extension to resolve from: a URL output picks it, or pass
    # --format precomputed.  Skeleton and graph stores become skeletons, mesh
    # stores legacy meshes.
    "precomputed": ExportFmt(
        "precomputed", (), "precomputed", "export_precomputed", "precomputed",
        "skeleton, graph or mesh",
        frozenset({
            "level", "object_ids", "segment_ids", "attribute_names",
            "object_attribute_names", "unit",
        }),
    ),
    "obj": ExportFmt(
        "obj", (".obj",), "obj", "export_obj", None, "mesh",
        _COMMON | {"bbox", "object_ids"},
    ),
    "ply": ExportFmt(
        "ply", (".ply",), "ply", "export_ply", "ply", "points",
        _COMMON | {"bbox", "object_ids", "attribute_names", "binary"},
    ),
    "csv": ExportFmt(
        "csv", (".csv", ".xyz"), "csv_points", "export_csv", None, "points",
        _COMMON | {"bbox", "object_ids", "attribute_names", "delimiter"},
    ),
    "h5ad": ExportFmt(
        "h5ad", (".h5ad",), "h5ad", "export_h5ad", "h5ad", "points",
        _COMMON | {"bbox", "object_ids", "attribute_names"},
    ),
}

_EXPORT_EXT_TO_FORMAT: dict[str, str] = {
    ext: fmt.name for fmt in EXPORT_REGISTRY.values() for ext in fmt.exts
}


def resolve_export_format(
    output_path: str | Path, explicit: str | None,
) -> ExportFmt:
    """Return the :class:`ExportFmt` for ``--format``, or the OUTPUT extension.

    The output names the format when exporting, which is the mirror of the
    ingest rule where the input does.
    """
    if explicit and explicit != "auto":
        try:
            return EXPORT_REGISTRY[explicit]
        except KeyError:
            raise SystemExit(
                f"error: cannot export to {explicit!r}; zvtools exports "
                f"{{{','.join(EXPORT_REGISTRY)}}}"
            ) from None
    if "://" in str(output_path):
        # Only a precomputed layer is written to a URL; every other exporter
        # writes a local file.
        return EXPORT_REGISTRY["precomputed"]
    ext = Path(output_path).suffix.lower()
    name = _EXPORT_EXT_TO_FORMAT.get(ext)
    if name is None:
        raise SystemExit(
            f"error: cannot tell what to export from the output extension "
            f"{ext or '(none)'!r}; pass --format "
            f"{{{','.join(EXPORT_REGISTRY)}}}"
        )
    return EXPORT_REGISTRY[name]


def load_export_func(fmt: ExportFmt):
    """Import the export entry function for ``fmt`` (lazy, with an install hint)."""
    import importlib

    try:
        mod = importlib.import_module(f"zarr_vectors_tools.convert.export.{fmt.module}")
        return getattr(mod, fmt.func)
    except ImportError as exc:
        hint = (
            f" — install it with: pip install 'zarr-vectors-tools[{fmt.extra}]'"
            if fmt.extra
            else ""
        )
        raise SystemExit(f"error: cannot load the {fmt.name} exporter ({exc}){hint}")


def _directory_format(path: Path) -> str:
    """The format of a directory input: the formats that come as sets of files."""
    from zarr_vectors_tools.convert.ingest.freesurfer import find_freesurfer_surf_dir

    if (path / "info").is_file():
        return "precomputed"
    if find_freesurfer_surf_dir(path) is not None:
        return "freesurfer"
    if any(p.suffix.lower() == ".gii" for p in path.iterdir()):
        return "gifti"
    raise SystemExit(
        f"error: {path} is a directory, but neither a FreeSurfer subject "
        f"(surf/lh.white), a directory of .gii files, nor a precomputed layer "
        f"(info); pass a file, or --format"
    )


def resolve_format(input_path: str | Path, explicit: str | None) -> Fmt:
    """Return the :class:`Fmt` for ``--format`` (if given) or the input extension.

    A directory input is resolved by what is in it, since the cortical surface
    formats describe one subject as many files and a precomputed layer is a
    directory.  A URL can only be a precomputed layer: every other ingester
    reads a local file.
    """
    from zarr_vectors_tools.convert.ingest.precomputed import is_url

    if explicit and explicit != "auto":
        return FORMAT_REGISTRY[explicit]
    if is_url(input_path):
        return FORMAT_REGISTRY["precomputed"]
    if Path(input_path).is_dir():
        return FORMAT_REGISTRY[_directory_format(Path(input_path))]
    ext = Path(input_path).suffix.lower()
    name = _EXT_TO_FORMAT.get(ext)
    if name is None:
        raise SystemExit(
            f"error: cannot auto-detect format from extension {ext!r}; "
            f"pass --format {{{','.join(FORMAT_REGISTRY)}}}"
        )
    return FORMAT_REGISTRY[name]


def load_ingest_func(fmt: Fmt):
    """Import the ingest entry function for ``fmt`` (lazy, with an install hint)."""
    import importlib

    try:
        mod = importlib.import_module(f"zarr_vectors_tools.convert.ingest.{fmt.module}")
        return getattr(mod, fmt.func)
    except ImportError as exc:
        hint = (
            f" — install it with: pip install 'zarr-vectors-tools[{fmt.extra}]'"
            if fmt.extra
            else ""
        )
        raise SystemExit(f"error: cannot load the {fmt.name} ingester ({exc}){hint}")
