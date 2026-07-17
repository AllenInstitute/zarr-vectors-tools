#!/usr/bin/env python
"""Add a uniform-random float32 object attribute to an existing zarr-vectors
streamline store, for use as a random-sampling filter in neuroglancer's
segment-properties UI (e.g. filter to `random_sample < 0.1` for a ~10% sample).

Writes one value per object (drawn i.i.d. from Uniform[0, 1)) to
`object_attributes/<ATTR_NAME>` at level 0. The store must already exist —
this does not create a new store.

Configure the constants below, then run:
    python scripts/add_random_sample_attribute.py
"""

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------

STORE = Path("/tmp/hbn_bundles.zarrvectors")
ATTR_NAME = "random_sample"
SEED = None  # set to an int for reproducible values


def main() -> None:
    from zarr_vectors.core.arrays import (
        create_object_attributes_array,
        write_object_attributes,
    )
    from zarr_vectors.core.store import get_resolution_level, open_store

    root = open_store(str(STORE), mode="r+")
    level_group = get_resolution_level(root, 0)

    num_objects = level_group.read_array_meta("object_index")["num_objects"]
    print(f"store has {num_objects} objects")

    rng = np.random.default_rng(SEED)
    data = rng.random(num_objects, dtype=np.float32)

    create_object_attributes_array(level_group, ATTR_NAME)
    write_object_attributes(level_group, ATTR_NAME, data)

    print(f"wrote object_attributes/{ATTR_NAME}: {data.shape} {data.dtype}")
    print(f"  range [{data.min():.6f}, {data.max():.6f}]")


if __name__ == "__main__":
    main()
