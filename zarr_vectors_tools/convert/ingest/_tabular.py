"""Shared helpers for ingesting tabular per-vertex data.

Zarr Vectors attributes are numeric arrays with Zarr-path names, while ``obs``
tables and CSV columns are arbitrarily named and arbitrarily typed. These
two helpers are the bridge, and they live here so the ``.h5ad`` and
delimited-table ingesters encode identically — a column staged in from a
CSV must land in the same representation as the same column read from an
``.h5ad``, or the two would not round-trip through one export.
"""

from __future__ import annotations

import re

import numpy as np

# Zarr path segments: keep to characters that are safe in a key.
_UNSAFE = re.compile(r"[^0-9A-Za-z_.-]")


def sanitise_name(name: object, used: set[str]) -> str:
    """Turn a column label into a unique, Zarr-safe attribute name.

    ``used`` is updated in place so repeated calls stay collision-free.
    """
    cleaned = _UNSAFE.sub("_", str(name)).strip("_") or "attr"
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    candidate, n = cleaned, 1
    while candidate in used:
        candidate = f"{cleaned}_{n}"
        n += 1
    used.add(candidate)
    return candidate


def encode_column(series) -> tuple[np.ndarray, list[str] | None]:
    """Encode one table column as a numeric array (+ category labels).

    Returns ``(values, categories)``; ``categories`` is None for columns
    that are natively numeric and therefore need no decode on export.
    """
    import pandas as pd

    dtype = series.dtype

    if isinstance(dtype, pd.CategoricalDtype):
        codes = np.asarray(series.cat.codes, dtype=np.int32)
        return codes, [str(c) for c in series.cat.categories]

    if pd.api.types.is_bool_dtype(dtype):
        return np.asarray(series, dtype=np.uint8), None

    if pd.api.types.is_numeric_dtype(dtype):
        # Nullable extension dtypes (Int64, Float64) carry pd.NA, which
        # numpy cannot hold -- go through float so NA becomes NaN.
        if pd.api.types.is_extension_array_dtype(dtype):
            return np.asarray(series.astype("float64")), None
        return np.asarray(series), None

    if pd.api.types.is_datetime64_any_dtype(dtype):
        # Nanoseconds since epoch; int64 is exact for the full range.
        return series.to_numpy(dtype="datetime64[ns]").astype(np.int64), None

    # Strings and mixed object columns -> factorise into codes + labels.
    codes, uniques = pd.factorize(series, use_na_sentinel=True)
    return np.asarray(codes, dtype=np.int32), [str(u) for u in uniques]


def dictionary_meta(categories: list[str], *, ordered: bool = False) -> dict:
    """The array metadata that marks a column of codes as categorical.

    The dictionary-encoding convention core writes for string attributes
    (``encoding``, ``categories``, ``ordered``, ``_FillValue``), and what the
    Neuroglancer viewer reads to show a code's label rather than its number.
    ``_FillValue`` is always -1: the pandas missing-code convention, which is
    also what a staged attach fills unmatched rows of a code column with.
    """
    return {
        "encoding": "dictionary",
        "categories": [str(c) for c in categories],
        "ordered": bool(ordered),
        "_FillValue": -1,
    }


def is_ordered(series) -> bool:
    """Whether a table column is an ordered ``pandas.Categorical``."""
    return bool(getattr(series.dtype, "ordered", False))


def mark_dictionary_encoded(store_path, level: int, columns: dict[str, dict]) -> None:
    """Stamp :func:`dictionary_meta` onto attribute arrays already written.

    Core's point writer dictionary-encodes only the string columns it is
    handed, and sorts their categories as it does.  The table ingesters
    hand it codes instead -- in the source's category order, which the
    h5ad header and every staged attach rely on -- so the metadata that
    makes those codes categorical is added here, after the write.
    """
    if not columns:
        return
    from zarr_vectors.building import VERTEX_ATTRIBUTES, get_resolution_level, open_store

    level_group = get_resolution_level(open_store(str(store_path), mode="r+"), level)
    for name, extra in columns.items():
        # Merged into the array's own attributes; nothing else is touched.
        level_group.write_array_meta(f"{VERTEX_ATTRIBUTES}/{name}", extra)
