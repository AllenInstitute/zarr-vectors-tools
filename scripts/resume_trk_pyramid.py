#!/usr/bin/env python
"""Resume a multiscale pyramid build that ran out of memory partway through.

Mirrors ingest_trk_parallel.py's Phase-6 pyramid config, but skips level 0
ingest entirely and skips whichever coarser levels already finished
successfully — it deletes the first INCOMPLETE level's directory (the one
that was partially written when the process died) and rebuilds it plus
every level after it.

Configure LEVELS_ALREADY_DONE below to the highest level that finished
successfully (NOT the one that crashed), then run:
    python scripts/resume_trk_pyramid.py
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — MUST match the PYRAMID_FACTORS / CHUNK_SCALE_FACTORS /
# SPARSITY_STRATEGY / PYRAMID_COARSEN_MODE / WORKERS used in the original
# ingest_trk_parallel.py run (copy them over if that script's config changes).
# ---------------------------------------------------------------------------

OUTPUT_STORE = Path("/tmp/wholebrain_tractogram.zarrvectors")

PYRAMID_FACTORS = [
    (8.0, 1.0),    # level 1
    (2.0, 2.0),    # level 2
    (2.0, 8.0),    # level 3
    (2.0, 64.0),   # level 4
    (1.0, 512.0),  # level 5
]
CHUNK_SCALE_FACTORS = [2, 2, 2, 2, 2]
SPARSITY_STRATEGY = "random"
PYRAMID_COARSEN_MODE = "decimate"
WORKERS = 8

# The highest level that finished successfully and should be KEPT as-is.
# The run died partway through building the NEXT level (LEVELS_ALREADY_DONE + 1)
# — that level's directory is incomplete/corrupt and gets deleted and rebuilt.
LEVELS_ALREADY_DONE = 1

# ---------------------------------------------------------------------------
# Entry point — DO NOT put logic at module level (Dask worker guard below)
# ---------------------------------------------------------------------------


def main() -> None:
    import shutil

    from zarr_vectors.core.store import list_resolution_levels, open_store
    from zarr_vectors_tools.multiresolution.coarsen import (
        _finalize_cross_level_for_store,
        coarsen_level,
    )

    crashed_level = LEVELS_ALREADY_DONE + 1
    existing = list_resolution_levels(open_store(str(OUTPUT_STORE)))
    print(f"Existing levels on disk: {existing}")
    if LEVELS_ALREADY_DONE not in existing:
        raise SystemExit(
            f"Level {LEVELS_ALREADY_DONE} (LEVELS_ALREADY_DONE) is not present "
            f"in the store — check the config before resuming."
        )

    crashed_dir = OUTPUT_STORE / str(crashed_level)
    if crashed_dir.exists():
        print(f"Removing incomplete level {crashed_level} at {crashed_dir} ...")
        shutil.rmtree(crashed_dir)
    for lvl in existing:
        if lvl > LEVELS_ALREADY_DONE:
            raise SystemExit(
                f"Level {lvl} exists on disk and is > LEVELS_ALREADY_DONE "
                f"({LEVELS_ALREADY_DONE}) but wasn't removed — refusing to "
                f"guess whether it's complete. Delete it manually first if "
                f"it's also incomplete, or raise LEVELS_ALREADY_DONE if it "
                f"actually finished."
            )

    # Only the (coarsen_factor, sparsity_factor, chunk_scale_factor) tuples
    # for the levels that still need building, e.g. LEVELS_ALREADY_DONE=1
    # skips the (8.0, 1.0) tuple that built level 1 and starts from (2.0, 2.0).
    remaining = list(
        zip(
            range(LEVELS_ALREADY_DONE, len(PYRAMID_FACTORS)),
            PYRAMID_FACTORS[LEVELS_ALREADY_DONE:],
            CHUNK_SCALE_FACTORS[LEVELS_ALREADY_DONE:],
        )
    )
    if not remaining:
        print("Nothing left to build — all configured levels already exist.")
        return

    def _run(executor) -> None:
        for source_level, (coarsen_factor, sparsity_factor), chunk_scale in remaining:
            target_level = source_level + 1
            print(
                f"Building level {target_level} from level {source_level} "
                f"(coarsen_factor={coarsen_factor}, sparsity_factor={sparsity_factor}, "
                f"chunk_scale_factor={chunk_scale}) ...",
                flush=True,
            )
            summary = coarsen_level(
                str(OUTPUT_STORE),
                source_level=source_level,
                target_level=target_level,
                coarsen_factor=coarsen_factor,
                sparsity_factor=sparsity_factor,
                chunk_scale_factor=chunk_scale,
                sparsity_strategy=SPARSITY_STRATEGY,
                coarsen_mode=PYRAMID_COARSEN_MODE,
                executor=executor,
            )
            print(f"  -> {summary}", flush=True)

        # Streamline/polyline coarsening doesn't emit inline ±1 cross-level
        # links (that's a per-object/skeleton-only feature), so there is
        # nothing for the ±N (N>=2) composition pass to compose from — it
        # would only do (wasted) O(all vertices in all levels) work
        # reconstructing per-level chunk assignments for no benefit. Stamp
        # root metadata as "no cross-level links" instead of calling
        # build_pyramid's default (cross_level_storage="explicit").
        _finalize_cross_level_for_store(
            str(OUTPUT_STORE), cross_level_depth=0, cross_level_storage="none",
        )

    if WORKERS > 1:
        from zarr_vectors_tools.ingest._parallel import dask_executor
        with dask_executor(WORKERS) as ex:
            _run(ex)
    else:
        _run(None)

    print("\nDone. Levels on disk:", list_resolution_levels(open_store(str(OUTPUT_STORE))))


# REQUIRED: Dask process workers re-import this module on macOS (spawn method).
# Without this guard every worker re-runs main() → recursive cluster spawn.
if __name__ == "__main__":
    main()
