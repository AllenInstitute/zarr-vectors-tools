#!/usr/bin/env python
"""Ingest a large TRX tractography file into a zarr-vectors streamline store.

Memory-bounded: streams the file in parallel per-streamline parts (reading
positions/dpv from the TRX memmap) rather than loading it whole. Mirrors the
pattern of ingest_trk_parallel.py, but for TRX — capturing named groups +
group attributes, per-streamline data (dps), per-vertex data (dpv), and the
VOXEL_TO_RASMM affine. Optionally builds a downsampled multiscale pyramid
after level 0, using the same coarsen/sparsity machinery (and the same
config knobs) as ingest_trk_parallel.py. Note: groups/group_attributes are
level-0-only — they are not propagated to coarser pyramid levels (the
neuroglancer frontend reads them once, from level 0, regardless of which
geometry level is currently displayed).

Configure the constants below, then run:
    python scripts/ingest_trx.py

Requires trx-python:  pip install trx-python
To use Dask parallel workers, set WORKERS > 1 (requires the 'parallel' extra).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------

# Path to the input .trx file (absolute or relative to repo root)
INPUT_TRX = Path(
    "sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_desc-bundles_tractography.trx"
)

# Output zarr-vectors store directory
OUTPUT_STORE = Path("/tmp/hbn_bundles.zarrvectors")

# For quick test runs: only ingest the first N streamlines (in on-disk order).
# Set to None to ingest all streamlines.
MAX_STREAMLINES = None

# Target total spatial chunk count; pipeline picks near-isotropic per-axis
# sizes. Pass a (nx, ny, nz) tuple to override per-axis.
NUM_CHUNKS = 5000

# How many pieces to split the streamlines into for Phase A binning. More
# parts = finer load balancing; the executor queues excess work so N_PARTS can
# safely be >> WORKERS. Does NOT change how many processes run at once.
N_PARTS = 20

# Number of Dask worker processes that run in parallel. Set to 1 for serial
# execution (useful for testing / low-memory machines).
WORKERS = 8

# Pyramid vertex-reduction mode:
#   "rdp"      - Douglas-Peucker simplification (dramatic, geometry-dependent).
#   "decimate" - uniform stride decimation: keep every Nth vertex plus
#                endpoints, for a gradual, predictable reduction per level.
PYRAMID_COARSEN_MODE = "decimate"

# Multiscale pyramid levels: list of (coarsen_factor, sparsity_factor).
#   In "rdp" mode, coarsen_factor drives the RDP epsilon
#       (epsilon = min_chunk * 0.5 * factor).
#   In "decimate" mode, coarsen_factor IS the stride (keep every Nth vertex).
# sparsity_factor drops that fraction of streamlines (1.0 = keep all).
PYRAMID_FACTORS = [
    (8.0, 1.0),   # level 1: stride-8 decimation, keep all streamlines
    (2.0, 2.0),   # level 2: stride-2 (16 total) decimation, keep half of total streamlines
    (2.0, 8.0),   # level 3: stride-2 (32 total) decimation, keep 1/8 of total streamlines
    (2.0, 64.0),  # level 4: stride-2 (64 total) decimation, keep 1/64 of total streamlines
    (1.0, 512.0)  # level 5: no decimation, keep only 1/512 of total streamlines
]

# Per-level chunk scale factor (multiply level-0 chunk size per level).
# Increase to cover more area per chunk at coarser levels.
CHUNK_SCALE_FACTORS = [2, 2, 2, 2, 2]

# Sparsity strategy for object dropping: "length" (keep longest) or "random".
SPARSITY_STRATEGY = "random"

# Whether to build the multiscale pyramid after level 0.
BUILD_MULTISCALE = True

# Keep intermediate .npz scratch files after ingest (useful for debugging).
KEEP_INTERMEDIATE = False

# Name of a per-streamline object_attributes column holding one uniform-
# random float32 in [0, 1) per streamline, for random-sampling filters in
# neuroglancer's segment-properties UI (e.g. "random_sample < 0.1" for a
# ~10% sample). Written before the pyramid phase so it propagates to
# coarser levels like any other object attribute. Set to None to skip it.
RANDOM_SAMPLE_ATTR = "random_sample"

# Seed for RANDOM_SAMPLE_ATTR; None for a fresh draw each run.
RANDOM_SAMPLE_SEED = None


# ---------------------------------------------------------------------------
# Entry point — DO NOT put logic at module level (Dask worker guard below)
# ---------------------------------------------------------------------------

def main() -> None:
    from zarr_vectors_tools.ingest.trx_parallel import ingest_trx_parallel

    if WORKERS > 1:
        from zarr_vectors_tools.ingest._parallel import dask_executor
        with dask_executor(WORKERS) as ex:
            summary = ingest_trx_parallel(
                INPUT_TRX,
                OUTPUT_STORE,
                num_chunks=NUM_CHUNKS,
                n_parts=N_PARTS,
                workers=WORKERS,
                executor=ex,
                max_streamlines=MAX_STREAMLINES,
                build_multiscale=BUILD_MULTISCALE,
                pyramid_factors=PYRAMID_FACTORS,
                chunk_scale_factors=CHUNK_SCALE_FACTORS,
                sparsity_strategy=SPARSITY_STRATEGY,
                pyramid_coarsen_mode=PYRAMID_COARSEN_MODE,
                keep_intermediate=KEEP_INTERMEDIATE,
                random_sample_attr=RANDOM_SAMPLE_ATTR,
                random_sample_seed=RANDOM_SAMPLE_SEED,
                progress=True,
            )
    else:
        summary = ingest_trx_parallel(
            INPUT_TRX,
            OUTPUT_STORE,
            num_chunks=NUM_CHUNKS,
            n_parts=N_PARTS,
            workers=1,
            max_streamlines=MAX_STREAMLINES,
            build_multiscale=BUILD_MULTISCALE,
            pyramid_factors=PYRAMID_FACTORS,
            chunk_scale_factors=CHUNK_SCALE_FACTORS,
            sparsity_strategy=SPARSITY_STRATEGY,
            pyramid_coarsen_mode=PYRAMID_COARSEN_MODE,
            keep_intermediate=KEEP_INTERMEDIATE,
            random_sample_attr=RANDOM_SAMPLE_ATTR,
            random_sample_seed=RANDOM_SAMPLE_SEED,
            progress=True,
        )

    print("\n=== Ingest summary ===")
    for k, v in summary.items():
        if k != "pyramid":
            print(f"  {k}: {v}")
    if summary.get("pyramid"):
        print(f"  pyramid levels built: {summary['pyramid'].get('levels_created', '?')}")


# REQUIRED: Dask process workers re-import this module on macOS (spawn method).
# Without this guard every worker re-runs main() → recursive cluster spawn.
if __name__ == "__main__":
    main()
