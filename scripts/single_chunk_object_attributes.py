#!/usr/bin/env python
"""Collapse ``object_attributes/<name>`` columns to a single Zarr chunk.

The ZVF spec chunks a per-object attribute column at 65,536 rows
(``docs/spec/layout/chunk_arrays.md``), so a store with more objects than
that splits the column across ``c/0``, ``c/1``, ....  Neuroglancer's
zarr-vectors reader reads only ``c/0`` and then fails a byte-count check
against the column's declared row count; the throw escapes
``buildSkeletonMetadata`` and takes the whole store open down with it, so
the layer never loads at all.

Collapsing the column to one chunk keeps it a valid Zarr v3 array — Zarr
allows a chunk larger than 65,536 rows, and the writer's own comment notes
the bucket is a write-amplification choice, not a correctness one — and
makes the column readable by that reader.  Nothing else in the store
changes: dtype, values, fill value, codecs and the ``zv_array`` attribute
block are all carried over verbatim.

This works around a reader bug rather than fixing the format.  Once the
reader pages every chunk of the column, spec-chunked stores load untouched
and this script becomes unnecessary.

Usage::

    python scripts/single_chunk_object_attributes.py STORE [--dry-run]
    python scripts/single_chunk_object_attributes.py STORE --attribute label_id
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors import open_store
from zarr_vectors.core.group import Group

# Mirrors ``zarr_vectors.core.arrays.OBJECT_ATTRIBUTE_ROW_BUCKET``; imported
# by value rather than by name so the script still reports sensibly against a
# core that renames it.
SPEC_ROW_BUCKET = 65_536


def _levels(root: Group) -> list[str]:
    """Resolution-level child names, in numeric order."""
    return sorted(
        (name for name in root.children() if name.isdigit()),
        key=int,
    )


def _columns(level: Group) -> list[str]:
    """Per-object attribute names under this level's ``object_attributes/``."""
    if "object_attributes" not in level:
        return []
    return sorted(level["object_attributes"].children())


def _chunk_count(shape: tuple[int, ...], chunks: tuple[int, ...]) -> int:
    return math.prod(
        math.ceil(s / c) for s, c in zip(shape, chunks, strict=True)
    )


def _backup(store_path: Path, level: str, name: str, backup_root: Path) -> None:
    src = store_path / level / "object_attributes" / name
    dst = backup_root / level / "object_attributes" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def rewrite_column(
    root: Group,
    level_name: str,
    name: str,
    *,
    dry_run: bool,
) -> dict[str, Any] | None:
    """Rewrite one column as a single chunk.  ``None`` if already single-chunk."""
    level = root[level_name]
    path = f"object_attributes/{name}"
    arr = level.zarr_group[path]
    shape = tuple(int(s) for s in arr.shape)
    chunks = tuple(int(c) for c in arr.chunks)
    n_chunks = _chunk_count(shape, chunks)
    if n_chunks == 1:
        return None

    # Read through zarr so every chunk is decoded and the fill-padded tail of
    # the last one is dropped — the on-disk chunks are full-width, so a raw
    # concatenation would be longer than the column.
    values = arr[...]
    attributes = dict(arr.attrs)
    fill_value = arr.fill_value

    summary = {
        "level": level_name,
        "name": name,
        "shape": shape,
        "dtype": str(arr.dtype),
        "chunks_before": chunks,
        "n_chunks_before": n_chunks,
    }
    if dry_run:
        return summary

    # ``shape`` in the zv attribute block is what the reader sizes its buffer
    # from; it must keep describing the same rows, so restamp it rather than
    # trusting whatever the writer left there.
    attributes["shape"] = list(shape)
    level.write_array(
        path,
        values,
        chunks=shape,
        fill_value=fill_value,
        attributes=attributes,
    )

    # Verify against a freshly resolved handle: the write invalidated the
    # cached node, and a silent no-op here is exactly the failure this script
    # exists to prevent.
    after = open_store(root.path, mode="r")[level_name].zarr_group[path]
    if tuple(int(c) for c in after.chunks) != shape:
        raise RuntimeError(
            f"{level_name}/{path}: chunk shape is {after.chunks}, expected {shape}"
        )
    round_tripped = after[...]
    if not np.array_equal(round_tripped, values, equal_nan=True):
        raise RuntimeError(f"{level_name}/{path}: values changed on rewrite")
    summary["n_chunks_after"] = 1
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("store", type=Path, help="path to the .zarrvectors store")
    ap.add_argument(
        "--attribute",
        action="append",
        dest="attributes",
        metavar="NAME",
        help="limit to this column (repeatable); default is every column",
    )
    ap.add_argument(
        "--level",
        action="append",
        dest="levels",
        metavar="N",
        help="limit to this resolution level (repeatable); default is all",
    )
    ap.add_argument(
        "--backup",
        type=Path,
        metavar="DIR",
        help="copy each column's directory here before rewriting it",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be rewritten, change nothing",
    )
    args = ap.parse_args()

    store_path = args.store.resolve()
    root = open_store(str(store_path), mode="r" if args.dry_run else "r+")

    levels = args.levels or _levels(root)
    changed: list[dict[str, Any]] = []
    skipped = 0
    for level_name in levels:
        names = args.attributes or _columns(root[level_name])
        for name in names:
            if args.backup and not args.dry_run:
                _backup(store_path, level_name, name, args.backup.resolve())
            result = rewrite_column(
                root, level_name, name, dry_run=args.dry_run
            )
            if result is None:
                skipped += 1
                continue
            changed.append(result)
            verb = "would rewrite" if args.dry_run else "rewrote"
            print(
                f"{verb} {level_name}/object_attributes/{name}: "
                f"{result['dtype']}{list(result['shape'])}, "
                f"{result['n_chunks_before']} chunks "
                f"(chunk {list(result['chunks_before'])}) -> 1"
            )

    print(
        f"\n{len(changed)} column(s) {'to rewrite' if args.dry_run else 'rewritten'}, "
        f"{skipped} already single-chunk"
    )
    if not changed:
        return 0
    if args.dry_run:
        return 0
    over = [c for c in changed if c["shape"][0] > SPEC_ROW_BUCKET]
    if over:
        print(
            f"note: {len(over)} column(s) now exceed the spec's "
            f"{SPEC_ROW_BUCKET}-row chunk bucket; appends to them rewrite the "
            "whole column."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
