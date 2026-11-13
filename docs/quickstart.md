# Quickstart

Convert a CSV of points into a store, check it, export it and read it in
Python. [Install](install.md) first.

## 1. Make a test table

This writes `cells.csv`: 10,000 points in 20 clusters, in µm, with a `group`
column and an `intensity` value.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
group = np.repeat(np.arange(20), 500)                # 20 objects x 500 points
centre = rng.uniform(100, 900, size=(20, 3))[group]
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

- `--object-id-column group` makes each cluster one object. The other columns
  become vertex attributes.
- Chunk and bin shapes are in the table's units ([Store layout](store_layout.md)).
- `--coarsen 2,2 --sparsity 2,2` adds two coarser levels, each with half the
  objects of the level below ([Pyramids](pyramids.md)).

## 3. Inspect and validate

```bash
zvtools info cells.zv
zvtools validate cells.zv --level 5
```

```text
store: cells.zv
  zv_version:        0.9.4
  geometry_types:    ['point_cloud']
  ...
  resolution levels: [0, 1, 2]
  ...
validation (level 5): OK
```

`--level` runs checks from 1 (structure) to 5 (the pyramid); the default is 3.

## 4. Export back to CSV

```bash
zvtools convert cells.zv cells_out.csv --attribute group --attribute intensity
```

The input is a store, so `convert` exports it in the format the `.csv`
extension names.

## 5. Read it in Python

```python
import zarr_vectors as zv

ds = zv.open("cells.zv")
r = ds.read(objects=[3], attributes=["intensity"])
print(r.vertex_count)                   # 500
```

The read API belongs to the core package: see its
{zvpy}`quickstart <getting_started/quickstart.html>`.
