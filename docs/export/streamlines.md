# Streamlines

Two targets for the ZVF `polylines` geometry: TrackVis TRK and TRX. Both
need the `streamlines` extra:

```bash
pip install "zarr-vectors-tools[streamlines]"
```

Both exporters reassemble each object's stored segments into one
contiguous vertex array before writing, and both raise `ExportError` if
the filters leave nothing to export.

## TrackVis TRK — `export_trk`

```python
import numpy as np
from zarr_vectors_tools.export.trk import export_trk

summary = export_trk(
    "tracts.zv",                # store_path
    "tracts.trk",               # output_path
    level=0,                    # resolution level to read
    object_ids=[3, 5, 17],      # keep only these streamlines
    group_ids=None,             # keep only these groupings
    chunks=None,                # chunk whitelist — see the warning below
    affine=np.eye(4),           # 4x4 vox->RAS; None also means identity
)
summary["streamline_count"]     # → int
summary["vertex_count"]         # → int
```

### Recovering the affine

`affine=None` writes an **identity** matrix — it does not consult the
store. Ingest saves the source file's voxel-to-RAS matrix in a
`TRKHeader` under `/headers/trk/`, but the exporter never reads it, so
you must pass it in yourself:

```python
from zarr_vectors_tools.headers import HeaderRegistry
from zarr_vectors_tools.export.trk import export_trk

reg = HeaderRegistry("tracts.zv")
header = reg.get("trk")                 # KeyError if ingest did not preserve one
export_trk("tracts.zv", "out.trk", affine=header.affine)
```

`TRKHeader.affine` unflattens the stored 16-float `vox_to_ras` list into
a `(4, 4)` array, returning `None` when the source file carried no
affine — in which case the identity default is the correct behaviour
anyway.

:::{warning}
Exporting without the stored affine silently produces a geometrically
valid but **misregistered** tractogram: the streamlines are in voxel
space while the file claims RAS. Always pass `affine=` when the store
has a `trk` header.
:::

## TRX — `export_trx`

```python
from zarr_vectors_tools.export.trx import export_trx

summary = export_trx(
    "tracts.zv",
    "tracts.trx",
    level=0,
    object_ids=[3, 5, 17],
    group_ids=None,
    chunks=None,
)
summary["streamline_count"]
summary["vertex_count"]
```

`export_trx` takes no `affine` — TRX carries its own spatial metadata,
which the exporter leaves at the `TrxFile` defaults. The writer populates
the positions array and the per-streamline offsets only; per-vertex
(`dpv`), per-streamline (`dps`), and per-group (`dpg`) fields are not
written, so store attributes and groupings do not survive the trip.

## The segment-level `chunks` caveat

:::{warning}
On both `export_trk` and `export_trx` the `chunks` filter selects at the
**segment** level, not the object level. A streamline crossing a chunk
boundary is stored as several segments, and each surviving contiguous run
becomes its own streamline in the output. The returned
`streamline_count` can therefore exceed the number of source objects, and
individual streamlines are cut short at the boundary of the whitelist.
:::

```python
# Objects here are whole streamlines — counts match the source.
by_object = export_trk("tracts.zv", "a.trk", object_ids=[7])

# Chunks here cut object 7 into however many segments it spans.
by_chunk = export_trk("tracts.zv", "b.trk", chunks=[(0, 0, 0), (0, 0, 1)])
by_chunk["streamline_count"] >= by_object["streamline_count"]   # possibly much greater
```

Use `object_ids` or `group_ids` whenever you need whole objects; reach
for `chunks` only when you genuinely want a spatial slab and can tolerate
cut streamlines.

## See also

- [Export overview](index.md) — shared call shape and the `level=` parameter.
- [Ingest → tractography](../ingest/tractography.md) — the symmetric direction.
- [Tractography at scale](../ingest/tractography_at_scale.md) — the parallel TRK path.
- [Headers](../headers.md) — `TRKHeader` fields and the `HeaderRegistry` API.
