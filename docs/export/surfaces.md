# Cortical surfaces

A surface store (from `ingest_gifti` or `ingest_freesurfer`) writes out to a
directory of GIFTI files with `export_gifti`. Requires `nibabel`
(`pip install "zarr-vectors-tools[surfaces]"`).

```python
from zarr_vectors_tools.convert.export.gifti import export_gifti

summary = export_gifti(
    "sub-01_surfaces.zv",        # store_path
    "sub-01_gifti",              # output_dir, created if missing
    level=0,                     # 0 = the mesh as ingested
    hemispheres=None,            # default both; "lh"/"rh" accepted
    surfaces=None,               # default geometry + every alternate
    attribute_names=None,        # default every map and parcellation
    prefix="sub-01",             # sub-01_hemi-L_pial.surf.gii
)
summary["files"]; summary["skipped"]; summary["warnings"]
```

On the command line the output is a directory, so it has no extension to
name the format: a surface store exported to an extensionless path picks
GIFTI, or pass `--format gifti`.

```bash
zvtools convert sub-01_surfaces.zarrvectors sub-01_gifti --prefix sub-01
```

| Store content | File | Arrays |
| --- | --- | --- |
| geometry and each `coords_<name>` surface | `hemi-L_<surface>.surf.gii` | float32 `POINTSET`, int32 `TRIANGLE`, `GeometricType`, `AnatomicalStructureSecondary` |
| morphometry (thickness, curv, sulc, area...) | `hemi-L_<name>.shape.gii` | `NIFTI_INTENT_SHAPE`, one array per column |
| other continuous maps | `hemi-L_<name>.func.gii` | `NIFTI_INTENT_NONE`; a CIFTI time series keeps `TIME_SERIES` and `TimeStep` |
| parcellation with a label table | `hemi-L_<name>.label.gii` | int32 codes + label table (names, RGBA) |

Every file carries `AnatomicalStructurePrimary` (`CortexLeft`/`CortexRight`).
A name that is not a valid BIDS label is written after a dot
(`hemi-L.aparc.a2009s.label.gii`), so `ingest_gifti` reads the name back
unchanged; any file for which that fails is listed in `warnings`.

**Coordinates** are written as stored, and the header's `space` becomes the
GIFTI dataspace (`scanner`, `talairach`...; FreeSurfer surface RAS is
`unknown`). `c_ras` is neither re-applied nor removed, and FreeSurfer's
`VolGeom*` metadata is not written, because FreeSurfer tools would add
`c_ras` a second time.

A map measured on one hemisphere only is written for that hemisphere
(`skipped` lists the rest). `level=N > 0` writes the decimated mesh; its
vertex numbering is its own, so full-resolution metrics and CIFTI files do
not apply to it.

:::{warning}
`ingest_gifti` on a directory reads every `.gii` in it. Export into an empty
directory if you mean to ingest it again.
:::

Not preserved: float64 maps become float32; triangle order (the set and
winding are kept); the intent of a GIFTI time series ingested from `.func.gii`
(written as `NONE`). Label code `-1`, the store's fill for "no parcel", is
written as label key `-1`; nibabel reads it back, but Workbench and FreeView
use key `0` for unassigned, so check a parcellation there before relying on
the medial wall's colour.

## See also

- [Ingest → cortical surfaces](../ingest/surfaces.md) — the symmetric direction.
- [Export overview](index.md)
