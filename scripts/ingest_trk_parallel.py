#!/usr/bin/env python
"""Ingest a large TRK tractography file into a zarr-vectors streamline store.

Memory-bounded: streams the file in parallel byte-range parts rather than
loading it whole.  Mirrors the pattern of ingest_cutout_parralel.py.

Configure the constants below, then run:
    python scripts/ingest_trk_parallel.py

To use Dask parallel workers, set WORKERS > 1 (requires the 'parallel' extra):
    pip install 'zarr-vectors-tools[parallel]'
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------

# Path to the input .trk file (can be absolute or relative to repo root)
INPUT_TRK = Path(
    "sub-Ha1_sample-RightHemi_space-orig_tract-wholebrain_track-ifod_tractogram.trk"
)

# Output zarr-vectors store directory
#OUTPUT_STORE = Path("/tmp/wholebrain_tractogram.zarrvectors")
OUTPUT_STORE = Path("/tmp/wholebrain_tractogram_subset.zarrvectors")

# For quick test runs: only ingest the first N streamlines from the file
# (in on-disk order). Set to None to ingest all streamlines.
MAX_STREAMLINES = 100_000

# Target total spatial chunk count; pipeline picks near-isotropic per-axis sizes.
# For ~70x184x111 mm extent, 125 gives ~5x13x8 mm chunks.
# Pass a (nx, ny, nz) tuple to override per-axis.
NUM_CHUNKS = 5000

# How many pieces to split the input file into for Phase A binning.
# More parts = finer load balancing; the executor queues excess work so
# N_PARTS can safely be >> WORKERS (e.g. 4-8× is a good rule of thumb).
# Increasing N_PARTS does NOT increase the number of processes running at
# once — that is controlled solely by WORKERS below.
N_PARTS = 4


# Number of Dask worker processes that run in parallel.
# This is the only knob that affects how many CPU cores are used at once.
# Set to 1 for serial execution (useful for testing / low-memory machines).
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
    (2.0, 2.0),   # level 2: stride-8 decimation, keep all streamlines
    (2.0, 8.0),   # level 3: stride-8 decimation, keep all streamlines
    (2.0, 64.0),   # level 4: stride-8 decimation, keep all streamlines
    (1.0, 512.0)
]

# PYRAMID_FACTORS = [
# ]
# Per-level chunk scale factor (multiply level-0 chunk size per level).
# Increase to cover more area per chunk at coarser levels.
CHUNK_SCALE_FACTORS = [2, 2, 2, 2, 2]
#CHUNK_SCALE_FACTORS = []


# Sparsity strategy for object dropping: "length" (keep longest) or "random".
SPARSITY_STRATEGY = "random"

# Whether to compute and store per-streamline length + endpoints.
COMPUTE_LENGTH = True
COMPUTE_ENDPOINTS = False

# Whether to build the multiscale pyramid after level 0.
BUILD_MULTISCALE = True

# Keep intermediate .npz scratch files after ingest (useful for debugging).
KEEP_INTERMEDIATE = False


# ---------------------------------------------------------------------------
# Entry point — DO NOT put logic at module level (Dask worker guard below)
# ---------------------------------------------------------------------------

def main() -> None:
    from zarr_vectors_tools.ingest.trk_parallel import ingest_trk_parallel

    if WORKERS > 1:
        from zarr_vectors_tools.ingest._parallel import dask_executor
        with dask_executor(WORKERS) as ex:
            summary = ingest_trk_parallel(
                INPUT_TRK,
                OUTPUT_STORE,
                num_chunks=NUM_CHUNKS,
                n_parts=N_PARTS,
                workers=WORKERS,
                executor=ex,
                max_streamlines=MAX_STREAMLINES,
                compute_length=COMPUTE_LENGTH,
                compute_endpoints=COMPUTE_ENDPOINTS,
                build_multiscale=BUILD_MULTISCALE,
                pyramid_factors=PYRAMID_FACTORS,
                chunk_scale_factors=CHUNK_SCALE_FACTORS,
                sparsity_strategy=SPARSITY_STRATEGY,
                pyramid_coarsen_mode=PYRAMID_COARSEN_MODE,
                keep_intermediate=KEEP_INTERMEDIATE,
                progress=True,
            )
    else:
        summary = ingest_trk_parallel(
            INPUT_TRK,
            OUTPUT_STORE,
            num_chunks=NUM_CHUNKS,
            n_parts=N_PARTS,
            workers=1,
            max_streamlines=MAX_STREAMLINES,
            compute_length=COMPUTE_LENGTH,
            compute_endpoints=COMPUTE_ENDPOINTS,
            build_multiscale=BUILD_MULTISCALE,
            pyramid_factors=PYRAMID_FACTORS,
            chunk_scale_factors=CHUNK_SCALE_FACTORS,
            sparsity_strategy=SPARSITY_STRATEGY,
            pyramid_coarsen_mode=PYRAMID_COARSEN_MODE,
            keep_intermediate=KEEP_INTERMEDIATE,
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
