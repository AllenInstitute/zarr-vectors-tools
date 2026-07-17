#!/usr/bin/env python
"""Ingest the Allen Mouselight precomputed skeleton collection into a
multiscale zarr-vectors store with 1 mm³ spatial chunks.

Source layer:
    precomputed://gs://allen_neuroglancer_ccf/Mouselight

The layer does not contain .frags spatially-indexed files.  All skeletons
are read individually via the cloud-volume library, distributed spatially
across the 1 mm chunk grid, and written to a zarr-vectors store.  Segment
properties from the layer's segment_properties sub-directory are converted
to object attributes in the store.

The skeleton's per-vertex attributes (declared in ``skeleton/info``'s
``vertex_attributes``) are carried through to the store's ``vertex_attributes/``
arrays and exposed in neuroglancer as ``prop_<name>()`` in the skeleton
shader.  For Mouselight these are:

    - ``radius``       — float32 per-vertex radius
    - ``vertex_types`` — SWC compartment code: 1 = soma, 2 = axon,
                         3 = (basal) dendrite, 4 = apical dendrite

Example: keep the segment's own colour (hash / stated / segment-color user
shader) but shade dendrite a bit darker than axon/soma, in the skeleton
shader::

    void main() {
      vec4 color = segmentColor();
      float t = prop_vertex_types();
      if (t == 3.0 || t == 4.0) {
        color.rgb *= 0.6;  // dendrite (basal=3, apical=4): darker
      }
      emitRGBA(color);
    }

Customisation:
    - OUT_STORE        : local or cloud output path
    - CHUNK_SHAPE_NM   : spatial chunk size in nm (default 1 mm per axis)
    - STRIDES          : per-pyramid-level decimation factors
    - CHUNK_SCALE_FACTORS : per-level chunk grid scale-up
    - SPARSITY_FACTORS : per-level object-retention fractions (1.0 = keep all)
    - READ_WORKERS     : number of threads for parallel cloud skeleton reads
    - PYRAMID_WORKERS  : number of Dask process workers for pyramid building
    - SEG_IDS_SUBSET   : set to a list of int IDs to run on a small test subset
                         before committing to the full dataset
"""
import numpy as np
from zarr_vectors_tools.ingest.precomputed_plain_skeletons import (
    PlainPrecomputedReader,
    run_ingest_plain,
)

# ---- configuration -------------------------------------------------------

URL = "precomputed://gs://allen_neuroglancer_ccf/Mouselight"
OUT_STORE = "/tmp/mouselight_zarr_vectors"

# 1 mm³ spatial chunks (in nm)
CHUNK_SHAPE_NM = (1_000_000.0, 1_000_000.0, 1_000_000.0)

# Multi-resolution pyramid: 3 coarser levels.
# Level 0 → 1: keep every 8th vertex, 2× chunk grid, all objects.
# Level 1 → 2: keep every 8th vertex (of already-coarsened), 2× chunk grid, all objects.
# Level 2 → 3: keep every 8th vertex, 2× chunk grid, retain 25% of objects.
STRIDES = [8, 8, 8]
CHUNK_SCALE_FACTORS = [2, 2, 2]
SPARSITY_FACTORS = [1.0, 1.0, 0.25]

# IO / compute parallelism
READ_WORKERS = 16          # threads (network IO-bound)
PYRAMID_WORKERS = 8        # Dask processes (CPU-bound coarsening)

# Set to a list of int IDs for a quick test run; None = full dataset.
SEG_IDS_SUBSET = None      # e.g. [10, 20, 30, 40, 50] for a 5-neuron test


# --------------------------------------------------------------------------

def main():
    reader = PlainPrecomputedReader(URL)

    # Optionally override the segment ID list for a quick test.
    seg_ids = SEG_IDS_SUBSET
    if seg_ids is None:
        all_ids = reader.segment_ids
        print(f"Discovered {len(all_ids)} segment IDs in layer.", flush=True)
        seg_ids = all_ids
    else:
        print(f"Running subset of {len(seg_ids)} segment IDs.", flush=True)

    summary = run_ingest_plain(
        reader,
        OUT_STORE,
        chunk_shape_nm=CHUNK_SHAPE_NM,
        seg_ids=seg_ids,
        strides=STRIDES,
        chunk_scale_factors=CHUNK_SCALE_FACTORS,
        sparsity_factors=SPARSITY_FACTORS,
        sparsity_strategy="length",
        drop_interior_below=3,
        read_workers=READ_WORKERS,
        pyramid_workers=PYRAMID_WORKERS,
        progress=True,
    )

    print("\nIngest summary:")
    print(f"  objects:                  {summary['objects']}")
    print(f"  level-0 fragments:        {summary['level0_fragments']}")
    print(f"  level-0 cross-chunk edges:{summary['level0_cross_chunk_edges']}")
    if "pyramid" in summary:
        print(f"  pyramid levels:           {summary['pyramid']['num_levels']}")
    print(f"\nOutput store: {OUT_STORE}")


# REQUIRED: Dask process workers re-import this module on each worker process
# (macOS uses the 'spawn' start method).  Without this guard every worker
# would re-run the ingest, recursively spawning more clusters → fork-bomb.
if __name__ == "__main__":
    main()
