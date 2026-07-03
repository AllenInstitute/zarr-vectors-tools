#!/usr/bin/env python
"""Ingest a 4x4x2 flywire .frags cutout centered on a mip_0 point, in parallel."""
import numpy as np
from zarr_vectors_tools.ingest.precomputed_skeletons import (
    PrecomputedFragsReader, enumerate_frag_keys, run_ingest,
)

URL = "gs://flywire_v141_m783/skeletons_mip_1"
CENTER_MIP0 = np.array([121851, 58059, 3495])      # flywire 4,4,40 nm voxels
COUNTS = np.array([8, 8, 4])                        # chunks per axis


def main():
    reader = PrecomputedFragsReader(URL)
    info = reader.info
    res = np.asarray(info.resolution_nm)            # (32,32,40) nm  = mip_1
    csv = np.asarray(info.chunk_size_voxels)        # (512,512,512) vox

    # mip_0 (4,4,40) -> spatial-index mip_1 voxels, then snap a centered block
    # onto the .frags grid (origin residue from a known-good chunk corner).
    center_mip1 = CENTER_MIP0 * np.array([4, 4, 40]) / res
    known = np.array([17398, 10448, 3088])          # any valid chunk corner
    anchor = (known + np.round((center_mip1 - COUNTS*csv/2 - known)/csv)*csv).astype(int)

    keys = enumerate_frag_keys(info, tuple(anchor), tuple(COUNTS))
    bounds = ([float(anchor[a]*res[a]) for a in range(3)],
              [float((anchor[a]+COUNTS[a]*csv[a])*res[a]) for a in range(3)])

    summary = run_ingest(
        reader, "/tmp/" \
        "", keys, bounds_nm=bounds,
        strides=[8, 8, 8], chunk_scale_factors=[2, 2, 2], sparsity_factors=[1, 1, 4],
        workers=8,            # <-- Dask local cluster of 8 process workers
        drop_interior_below=3,   # optional LOD: drop tiny fully-interior objects
    )
    print(summary)


# REQUIRED: Dask's process workers (macOS uses the 'spawn' start method) re-import
# this module in each worker.  Without this guard, every worker would re-run the
# ingest and recursively spawn more clusters → crash/fork-bomb.
if __name__ == "__main__":
    main()
