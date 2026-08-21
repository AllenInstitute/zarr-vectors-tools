"""``zvtools attach`` — stage a file's columns into an existing store.

The counterpart to ``convert``: where ``convert`` builds a store from one
file, ``attach`` adds per-vertex attributes to a store that already exists,
matching rows to points on a join key. Datasets split across several files
are imported one file at a time rather than being merged up-front.
"""

from __future__ import annotations

from pathlib import Path

_H5AD_EXTS = {".h5ad"}


def _resolve_source(input_path: str, explicit: str | None) -> str:
    """Decide whether the incoming file is an ``.h5ad`` or a table."""
    if explicit and explicit != "auto":
        return explicit
    return "h5ad" if Path(input_path).suffix.lower() in _H5AD_EXTS else "table"


def run_attach(args) -> int:
    source = _resolve_source(args.input, args.format)

    if source == "h5ad":
        from zarr_vectors_tools.ingest.h5ad import attach_h5ad

        if not args.genes and not args.columns:
            raise SystemExit(
                "error: attach from h5ad needs --gene NAME and/or --column NAME"
            )
        summary = attach_h5ad(
            str(args.store),
            str(args.input),
            genes=args.genes,
            obs_columns=args.columns,
            layer=args.layer,
            gene_by=args.gene_by,
            key_obs_column=args.key_column,
            key_attribute=args.key_attribute,
            level=args.level,
            missing=args.missing,
            overwrite=args.overwrite,
            shard_shape=args.shard,
            progress=True,
        )
    else:
        from zarr_vectors_tools.ingest.cell_table import attach_table

        if not args.key_column:
            raise SystemExit(
                "error: attach from a table needs --key-column NAME "
                "(the column holding the identifier to join on)"
            )
        summary = attach_table(
            str(args.store),
            str(args.input),
            key_column=args.key_column,
            columns=args.columns,
            delimiter=args.delimiter,
            key_attribute=args.key_attribute,
            level=args.level,
            missing=args.missing,
            overwrite=args.overwrite,
            shard_shape=args.shard,
            progress=True,
        )

    print(f"attached {source} -> {args.store}")
    for key in (
        "attributes_written", "genes_attached", "obs_columns_attached",
        "columns_attached", "vertices_matched", "vertices_unmatched",
        "rows_available", "duplicate_keys", "chunk_count",
    ):
        if key in summary:
            print(f"  {key}: {summary[key]}")

    unmatched = summary.get("vertices_unmatched", 0)
    if unmatched:
        total = unmatched + summary.get("vertices_matched", 0)
        print(
            f"  note: {unmatched}/{total} point(s) had no row in this file "
            f"and were filled"
        )
    return 0
