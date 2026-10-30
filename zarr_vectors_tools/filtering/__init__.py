"""TrackVis-style streamline filtering: a filter-tree format and its reference engines.

A filter is a tree of leaves -- passes through a region, a column
comparison, group membership -- combined with AND, OR and NOT
(:mod:`.predicates`, which also defines the JSON form engines exchange).
Two engines evaluate it over a :class:`Tractogram` held in memory:

- :mod:`.oracle` tests every segment, brute force.  It is the ground truth.
- :class:`IndexedEngine` files segments under a grid, tests only the
  segments near a region, and caches each leaf, so an operator change is
  only boolean algebra.  Tested to agree with the oracle exactly.

:func:`load_tractogram` reads a store's level 0 into a :class:`Tractogram`
in the RAS millimetres :func:`~zarr_vectors_tools.algorithms.select_streamlines`
uses, and :mod:`.synthetic` makes whole-brain-sized test sets.

The subpackage imports only numpy (the store reader is imported when
:func:`load_tractogram` is called), and its modules import one another
relatively, so it can be copied into a Pyodide worker as it is.
"""

from .indexed import IndexedEngine
from .oracle import evaluate, histogram, summarize
from .predicates import (
    All,
    And,
    Box,
    Column,
    Disk,
    InGroup,
    Mask,
    Not,
    Or,
    Passes,
    Slab,
    Sphere,
    from_json,
    to_json,
)
from .synthetic import whole_brain_like
from .tractogram import Tractogram, load_tractogram

__all__ = [
    "All",
    "And",
    "Box",
    "Column",
    "Disk",
    "InGroup",
    "IndexedEngine",
    "Mask",
    "Not",
    "Or",
    "Passes",
    "Slab",
    "Sphere",
    "Tractogram",
    "evaluate",
    "from_json",
    "histogram",
    "load_tractogram",
    "summarize",
    "to_json",
    "whole_brain_like",
]
