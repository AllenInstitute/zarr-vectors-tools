"""Per-object face groups, so a reader can fetch one object's faces.

A level stores each chunk's intra-chunk links (a mesh's faces) as flat rows
plus a ``link_fragments`` sidecar giving the row range of each link group.
The format lets groups be anything.  The mesh coarseners write exactly one
group per vertex fragment, in fragment order (empty where a fragment has no
intra-chunk faces), so an object's manifest -- which names vertex
fragments -- also names its faces, and a reader drawing one object can
range-read just its rows instead of the whole cell.

zarr-vectors-py owns the claim that a level is laid out this way: it checks
every chunk and stamps ``LevelMetadata.fragment_link_groups`` (and the root
``fragment_link_groups`` capability) only where the layout holds, and any
later write to the arrays it describes clears it.  Against a
zarr-vectors-py without that, the groups are still written -- they are
valid groups, and readers concatenate them -- but nothing is stamped.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

try:
    from zarr_vectors.building import (
        index_fragment_link_groups as _index_fragment_link_groups,
    )
    from zarr_vectors.building import (
        stamp_fragment_link_groups as _stamp_fragment_link_groups,
    )
except ImportError:  # zarr-vectors-py without fragment_link_groups
    _index_fragment_link_groups = None
    _stamp_fragment_link_groups = None

#: Whether the installed zarr-vectors-py can stamp ``fragment_link_groups``.
HAS_FRAGMENT_LINK_GROUPS = _stamp_fragment_link_groups is not None


def split_faces_by_fragment(
    faces: npt.NDArray[np.integer],
    fragment_of: npt.NDArray[np.int64],
    num_fragments: int,
) -> list[npt.NDArray[np.integer]] | None:
    """One group of rows per fragment, in fragment order (empty where a
    fragment has none), or ``None`` when some row's corners do not all lie in
    one fragment -- then no per-fragment grouping exists."""
    faces = np.asarray(faces)
    width = faces.shape[1] if faces.ndim == 2 else 3
    faces = faces.reshape(-1, width)
    if faces.size and (faces.min() < 0 or faces.max() >= len(fragment_of)):
        return None
    owner = fragment_of[faces] if faces.size else np.zeros((0, width), np.int64)
    if (owner < 0).any() or (owner != owner[:, :1]).any():
        return None
    order = np.argsort(owner[:, 0], kind="stable")
    sorted_faces = faces[order]
    bounds = np.searchsorted(owner[order, 0], np.arange(num_fragments + 1))
    return [sorted_faces[bounds[f]:bounds[f + 1]] for f in range(num_fragments)]


def stamp_face_groups(level_group: Any, root: Any) -> bool:
    """Have zarr-vectors-py check the level's face groups and stamp it if
    they follow its vertex fragments.  Call once the level's writes are done:
    a later write clears the stamp.  ``False`` when not stamped, including
    when the installed zarr-vectors-py cannot stamp at all."""
    if _stamp_fragment_link_groups is None:
        return False
    return bool(_stamp_fragment_link_groups(level_group, root))


def index_mesh_faces(
    store_path: str,
    levels: list[int] | None = None,
    *,
    dry_run: bool = False,
    verify: bool = False,
) -> list[dict[str, Any]]:
    """Give an existing store's levels one face group per vertex fragment and
    stamp them -- :func:`zarr_vectors.building.index_fragment_link_groups`,
    which rewrites as little as it can (usually just the ``link_fragments``
    sidecar) and leaves alone a level where a face joins two fragments."""
    if _index_fragment_link_groups is None:
        raise RuntimeError(
            "the installed zarr-vectors-py has no fragment_link_groups "
            "support (zarr_vectors.building.index_fragment_link_groups)"
        )
    return _index_fragment_link_groups(
        str(store_path), levels, dry_run=dry_run, verify=verify,
    )
