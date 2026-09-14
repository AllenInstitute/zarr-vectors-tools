"""Carrying the group taxonomy across pyramid levels.

Groups name *object ids* — "these OIDs are the streamlines", "this OID is the
whole network" — and every coarsener in this package preserves object ids
(``CAP_PRESERVED_OBJECT_IDS``).  A group's membership is therefore meaningful
at a coarse level unchanged, minus whatever sparsification dropped.

Historically no coarsener wrote ``groups/`` at all, so a pyramid's coarse levels
had object indices but no way to say what any object *was*: a reader asking for
"the network object" at level 2 got nothing, and had to reach back to level 0
for the taxonomy while reading manifests from level 2.  This module closes that
gap, so ``<N>/groups/<id>`` answers at every level the same way ``0/groups/<id>``
does.

Group attributes (``group_attributes/<name>``) ride along, since they are
indexed by group id and are what carry human-readable group names.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt
from zarr_vectors.building import (
    create_groupings_array,
    create_groupings_attributes_array,
    read_all_groupings,
    read_groupings_attributes,
    write_groupings,
    write_groupings_attributes,
)
from zarr_vectors.constants import GROUP_ATTRIBUTES

__all__ = ["group_labels_for", "propagate_groupings"]


def group_labels_for(src_group, n_objects: int) -> npt.NDArray[np.int64]:
    """One group id per object, for the ``"group"`` sparsity strategy.

    Objects in no group get ``-1`` and form their own stratum — thinned
    at the same rate as everything else, which is what a store holding a
    whole-brain tractogram *and* a labelled atlas wants: the unlabelled
    bulk coarsens normally while every named bundle survives.

    An object in more than one group is assigned to the highest-numbered
    one.  Stratification needs a partition and groups are not one; picking
    a rule and saying so beats sampling the same object twice.

    Lives here rather than in one strategy module because every coarsener
    that offers ``sparsity_strategy="group"`` needs it, and the CLI offers
    that strategy for every geometry.
    """
    labels = np.full(int(n_objects), -1, dtype=np.int64)
    try:
        groupings = list(read_all_groupings(src_group))
    except Exception:  # noqa: BLE001 - a level with no taxonomy at all
        groupings = []
    if not groupings:
        raise ValueError(
            "sparsity_strategy='group' needs the source level to have object "
            "groups, and this one has none. Build the taxonomy first (see "
            "zarr_vectors_tools.compose.derive_groups), or use a different "
            "strategy."
        )
    for gid, members in enumerate(groupings):
        # A contiguous group arrives as a range and can be a billion long;
        # slice it rather than materialising it.
        if isinstance(members, range):
            lo = max(0, int(members.start))
            hi = min(int(n_objects), int(members.stop))
            if hi > lo:
                labels[lo:hi] = gid
            continue
        ids = np.fromiter((int(o) for o in members), dtype=np.int64)
        if len(ids):
            labels[ids[(ids >= 0) & (ids < n_objects)]] = gid
    return labels


def _group_attribute_names(level_group) -> list[str]:
    try:
        grp = level_group.zarr_group[GROUP_ATTRIBUTES]
        return sorted(set(grp.array_keys()) | set(grp.group_keys()))
    except Exception:  # noqa: BLE001
        return []


def _carry_group_names(src_group, dst_group) -> list[str]:
    """Copy the row labels alongside the rows.

    Memberships live in the ``groups`` array; the names live in that
    array's own metadata, under ``group_names``.  Carrying only the first
    leaves a coarse level whose rows are addressable as ``group_0`` and
    nothing else — which is precisely the "a store cannot be understood
    without the writing application's source next to it" problem the
    names were added to fix, reintroduced one level up.

    Row ids are preserved by every coarsener here, so the labels transfer
    positionally with no remapping.
    """
    from zarr_vectors.constants import GROUPS

    try:
        names = list(src_group.read_array_meta(GROUPS).get("group_names") or [])
    except Exception:  # noqa: BLE001 - an unnamed source has nothing to carry
        return []
    if not names:
        return []
    try:
        meta = dict(dst_group.read_array_meta(GROUPS))
        meta["group_names"] = [str(n) for n in names]
        dst_group.write_array_meta(GROUPS, meta)
    except Exception:  # noqa: BLE001 - names are additive, never load-bearing
        return []
    return [str(n) for n in names]


def propagate_groupings(
    src_group,
    dst_group,
    *,
    surviving_oids: Iterable[int] | None = None,
) -> dict[int, int]:
    """Copy ``src_group``'s group taxonomy onto ``dst_group``.

    Args:
        src_group: Source level group (the level being coarsened).
        dst_group: Target level group (the level just written).
        surviving_oids: OIDs that still exist at the target level.  ``None``
            keeps every member — correct when ``sparsity_factor == 1``.  Members
            outside this set are dropped, because a group listing an OID that
            sparsification removed would hand readers an empty manifest and call
            it a member.

    Returns:
        ``{group_id: member_count}`` for what was written; ``{}`` when the
        source has no groupings (nothing to carry, not an error).
    """
    try:
        groupings = read_all_groupings(src_group)
    except Exception:  # noqa: BLE001
        return {}
    if not groupings:
        return {}

    keep = None if surviving_oids is None else {int(o) for o in surviving_oids}

    out: dict[int, list[int] | range] = {}
    for gid, members in enumerate(groupings):
        # A contiguous group comes back as a ``range`` and is stored O(1) as a
        # [start, stop) descriptor. Keep it that way when nothing is filtered
        # out — materialising a billion-member range to drop nothing would be a
        # pointless blow-up of both memory and the on-disk group blob.
        if keep is None and isinstance(members, range):
            out[gid] = members
            continue
        ids = [int(o) for o in members]
        if keep is not None:
            ids = [o for o in ids if o in keep]
        out[gid] = ids

    create_groupings_array(dst_group)
    write_groupings(dst_group, out)
    _carry_group_names(src_group, dst_group)

    # Group attributes are indexed by group id, and group ids are unchanged, so
    # they copy across verbatim.
    for name in _group_attribute_names(src_group):
        try:
            values = np.asarray(read_groupings_attributes(src_group, name))
        except Exception:  # noqa: BLE001
            continue
        try:
            create_groupings_attributes_array(
                dst_group, name,
                dtype=str(values.dtype),
                num_channels=int(values.shape[1]) if values.ndim > 1 else 1,
            )
            write_groupings_attributes(dst_group, name, values)
        except Exception:  # noqa: BLE001
            # An attribute that will not round-trip is not worth failing a
            # whole pyramid over; the memberships are the load-bearing part.
            continue

    return {gid: (len(v) if not isinstance(v, range) else len(v))
            for gid, v in out.items()}


def surviving_oids_from(keep_oids: Any, sparsity_factor: float) -> list[int] | None:
    """``keep_oids`` when sparsification actually dropped something, else ``None``.

    Passing ``None`` through to :func:`propagate_groupings` is what lets a
    contiguous ``range`` group stay a range.
    """
    if sparsity_factor and float(sparsity_factor) > 1.0:
        return list(keep_oids)
    return None
