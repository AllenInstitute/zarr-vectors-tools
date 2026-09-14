"""Tools-owned constants for the multiresolution layer.

These are values the coarsening strategies agree on but core has no
opinion about — they describe how *this* package schedules and tags its
work, not anything on disk.  They live here rather than in a strategy
module because both strategies and ``coarsen`` read them, and a shared
module is what keeps that from becoming a strategy↔strategy import.
"""

from __future__ import annotations

__all__ = [
    "COARSEN_SKELETON",
    "COARSEN_SKELETON_BINS",
    "CROSS_LINK_TASK_SHARD_AXIS",
]

#: Coarsening-method tag for skeleton-aware pyramids.  Recorded in level
#: metadata so a reader can tell how a level was produced.  Skeleton
#: coarsening is a tools-owned strategy, so the tag is tools-owned too.
COARSEN_SKELETON: str = "skeleton_simplify"

#: Coarsening-method tag for the connected-bin metavertex strategy.  Kept
#: distinct from :data:`COARSEN_SKELETON` because the two produce materially
#: different levels — stride decimation severs cross-chunk connectivity and
#: bounds nothing, bin contraction preserves components and bounds error by the
#: bin diagonal — and a reader must be able to tell which one made a level.
COARSEN_SKELETON_BINS: str = "skeleton_bin_contract"

#: Outer width per axis used to group cross-link cell writes into
#: race-safe task partitions — one worker owns each partition.
#:
#: Purely a task-granularity knob: it is NOT an on-disk value and has
#: nothing to do with zarr sharding.  Under the merged links layout a cell
#: is (offsets_segment, source_chunk) and that maps 1:1 to the endpoint
#: pair, so *any* partition of pairs across workers is already race-safe;
#: this only bounds how many pairs one task takes.
CROSS_LINK_TASK_SHARD_AXIS: int = 4
