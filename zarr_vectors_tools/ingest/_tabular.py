"""Shared helpers for ingesting tabular per-vertex data.

ZVF attributes are numeric arrays with Zarr-path names, while ``obs``
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
