# Quickstart

Convert a CSV of points into a store, check it, export it back and read it in
Python. [Install](install.md) first; [Concepts](concepts.md) defines the terms.

## 1. Make a test table

This writes `cells.csv`: 10,000 points in 20 clusters, with coordinates in µm, a
`group` column naming each point's cluster and an `intensity` value per point.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
group = np.repeat(np.arange(20), 500)                # 20 objects x 500 points
centre = rng.uniform(100, 900, size=(20, 3))[group]  # one centre per object, in µm
xyz = centre + rng.normal(0, 30, size=(group.size, 3))
pd.DataFrame({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
              "group": group, "intensity": rng.random(group.size)}
             ).to_csv("cells.csv", index=False)
```

## 2. Convert it

```bash
zvtools convert cells.csv cells.zv --format table \
    --position-columns x,y,z --object-id-column group \
    --chunk-shape 250,250,250 --bin-shape 10,10,10 \
    --coarsen 2,2 --sparsity 2,2 --cross-level-storage none
```

- `--format table --object-id-column group` reads named columns and makes each
  cluster one object (a plain `.csv` gets no object ids). Each non-position
  column, here `group` and `intensity`, becomes a vertex attribute.
- Chunk and bin shapes are in the table's units: 250 µm chunks, 10 µm bins. To
  choose your own, see [Store layout](store_layout.md).
- `--coarsen 2,2 --sparsity 2,2` adds two coarser levels, each keeping half the
  objects of the level below and doubling the bin width. See [Pyramids](pyramids.md).

## 3. Inspect and validate

```bash
zvtools info cells.zv
zvtools validate cells.zv --level 5
```

```text
store: cells.zv
  zv_version:        0.9.2
  geometry_types:    ['point_cloud']
  chunk_shape:       (250.0, 250.0, 250.0)
  resolution levels: [0, 1, 2]
validation (level 5): OK
```

`--level` runs cumulative checks from 1 (structure) to 5 (adds the pyramid),
default 3. A failed check exits with status 1 and says what is wrong.

## 4. Export back to CSV

```bash
zvtools convert cells.zv cells_out.csv --attribute group --attribute intensity
```

```text
dim0,dim1,dim2,group,intensity
246.525040,446.130249,100.982887,6.000000,0.454368
```

Because the input is a store, `convert` exports it, in the format the `.csv`
extension names. Rows come out in chunk order; `--level 1` exports a coarser level.

## 5. Read it in Python

```python
from zarr_vectors.types.points import read_points

pts = read_points("cells.zv", level=0, object_ids=[3], attribute_names=["intensity"])
print(pts["vertex_count"])                        # 500
print(pts["positions"][:2])                       # (N, 3) float32 array
print(pts["vertex_attributes"]["intensity"][:2])  # [0.44001247 0.59332326]
```

## Next steps

- [Convert](convert/index.md): every input format, and the options they share.
- [Store layout](store_layout.md): choosing chunk and bin shapes.
- [Pyramids](pyramids.md): choosing coarser levels.
- [Visualise](visualise.md): view the store in Neuroglancer.
