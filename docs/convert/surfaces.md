# Cortical surfaces

```bash
zvtools convert subjects/bert bert.zv --chunk-shape 20,20,20
zvtools convert sub-01/anat sub-01.zv --chunk-shape 20,20,20
zvtools attach bert.zv sub-bert.MyelinMap.dscalar.nii
zvtools convert bert.zv bert_gifti --format gifti
```

A FreeSurfer subject and a directory of GIFTI files are recognised by what they hold.
`--chunk-shape` is in millimetres ([choosing it](../store_layout.md)). CIFTI files carry
no geometry, so they are attached to a surface store, not converted. Everything here
needs the `[surfaces]` extra (nibabel).

## What a surface store holds

- One mesh [object](../concepts.md) per hemisphere: left is object 0 and right object
  1 when both are present, also as groups named `left` and `right`.
- One anatomical surface is the geometry, which is chunked and answers bounding-box
  queries. Every other surface (pial, white, inflated, sphere...) is a three-column
  vertex attribute, `coords_<name>`, on the same vertices.
- Maps are float32 vertex attributes named after what they measure (`thickness`).
  Parcellations are int32 codes, with names and colours in the `surface` header.
- `zv_join_key` holds each vertex's hemisphere and source vertex number, which chunking
  would otherwise lose. CIFTI attach and GIFTI export use it.
- A map present on one hemisphere only is kept; the other gets NaN (-1 for labels).

The header also records the geometry, space, per-hemisphere vertex counts and `c_ras`
([Headers and attributes](../reference/headers_attributes.md)). `coords_<name>` and time
series are multi-column attributes: read [Visualise](../visualise.md) before viewing.

## FreeSurfer subject

```bash
zvtools convert subjects/bert bert_lh.zv --chunk-shape 20,20,20 \
    --hemisphere lh --space surface --surface pial --surface inflated \
    --morph thickness --annot aparc --annot aparc.a2009s
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--geometry NAME` | `midthickness` | surface to chunk: `midthickness`, `white`, `pial`, `smoothwm` or `orig` |
| `--hemisphere NAME` | both present | `lh` or `rh` (`left`, `right`) to read; repeatable |
| `--space MODE` | `auto` | `auto`, `scanner` or `surface`: see below |
| `--surface NAME` | white, pial, inflated, sphere | other surfaces kept as `coords_<name>`; repeatable |
| `--morph NAME` | thickness, curv, sulc, area | maps from `surf/?h.<name>`; repeatable |
| `--annot NAME` | aparc | parcellations from `label/?h.<name>.annot`; repeatable |

INPUT is the subject directory (holding `surf/` and `label/`) or its `surf/`. Defaults
take the files that exist; a name you pass must exist. Without a `?h.midthickness`
file, midthickness is computed as the mean of white and pial.

### Coordinates and c_ras

FreeSurfer writes surfaces in *surface RAS*, centred on the conformed volume; the T1,
diffusion data and tractography of the same subject are in *scanner RAS*. The two
differ by one translation, `c_ras`, read from the surface's volume footer when
FreeSurfer marked it valid, else from `mri/orig.mgz` (then `T1.mgz`, `brain.mgz`). An
invalid footer, as `fsaverage` ships, is never used.

`--space auto` (default) adds `c_ras` when it is known and keeps surface RAS
otherwise; `scanner` requires `c_ras`; `surface` leaves FreeSurfer's coordinates
unchanged. Only anatomical surfaces are shifted, not inflated or spherical ones, and
the summary prints the `space` and `c_ras` used. Keep scanner RAS to view or query the
surfaces beside the same subject's tractogram: in surface RAS they sit `c_ras` apart.

## GIFTI

```bash
zvtools convert sub-01/anat sub-01_pial.zv --chunk-shape 20,20,20 --geometry pial
zvtools convert cortex.surf.gii cortex.zv --chunk-shape 20,20,20 --hemisphere left
```

INPUT is one `.gii` file or a directory, all of whose `.gii` files are read. Each file
is sorted by its metadata first and its name second:

- **Kind.** `POINTSET` with `TRIANGLE` is a surface, `LABEL` a parcellation, anything
  else a map.
- **Hemisphere.** `AnatomicalStructurePrimary`, on the file or on an array (where
  `mris_convert` puts it), else the name (`hemi-L`, `.L.`, `lh.`, `left`).
  `--hemisphere`, given once, assigns the files that say neither.
- **Name.** The file name without subject, hemisphere, mesh and type parts (a BIDS
  `desc-` wins): `100307.L.MyelinMap_BC.32k_fs_LR.func.gii` gives `MyelinMap_BC`.

`--geometry` defaults to the first present of midthickness, pial, white, smoothwm and
orig. A map with a different vertex count from its surface is refused: resample it
first (for example with `wb_command -metric-resample`).

### GIFTI coordinates

A surface file stores positions in its *dataspace* and may carry a matrix,
`CoordinateSystemTransformMatrix`, to a *transformed space*. FreeSurfer's
`mris_convert` writes surface RAS positions (dataspace `unknown`) and a translation by
`c_ras` to `scanner`: read as stored, such a surface sits `c_ras` away from the same
subject's T1 and tractography.

`--space auto` (default) applies each anatomical surface's matrix when it leads to a
named space and is not the identity, and records that space; otherwise positions are
kept and the dataspace is recorded. `scanner` refuses a surface that does not reach
scanner RAS; `surface` keeps every position as the file stores it. Inflated, spherical
and flat surfaces are never moved. The summary prints the `space` and, when the
matrices are one translation into scanner RAS, the `c_ras`; the header keeps every
matrix applied (`applied_transforms`). To view a surface over the same subject's TRK
tractogram, convert the TRK with `--apply-affine` so both are in scanner RAS
([Streamlines → Coordinates](streamlines.md#coordinates)).

## CIFTI with zvtools attach

```bash
zvtools attach bert.zv sub-bert.parcels.dlabel.nii
zvtools attach bert.zv rfMRI_REST1_LR.dtseries.nii --name rest1
```

| File | Written as |
| --- | --- |
| `.dscalar.nii` | a float32 attribute per map, named after the map (after the file if it has one map); `--name` sets the name, and keeps several maps together as one multi-column attribute |
| `.dlabel.nii` | an int32 attribute per map, its label table in the header |
| `.dtseries.nii` | one float32 attribute, a column per time point; timing in the header |

Values find their vertex through the file's brain-model axis and `zv_join_key`.
Vertices the file leaves out (usually the medial wall) get NaN, or -1 for labels;
`--missing error` refuses instead, before anything is written. Subcortical voxels
are skipped. Attributes go to level 0: attach before building a pyramid, or rebuild
it after. `--overwrite` replaces existing attributes. The file must be on the
store's mesh: a 164k fs_LR map is refused on a native FreeSurfer subject, with
`CIFTI_STRUCTURE_CORTEX_LEFT is defined on a 163842-vertex mesh but the store's left hemisphere has 2562 vertices`.

## Export to GIFTI

```bash
zvtools convert bert.zv sub-bert_gifti --format gifti --prefix sub-bert
zvtools convert bert.zv bert_lh --hemisphere lh --surface pial --attribute thickness --attribute aparc
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--hemisphere NAME` | both | `lh` or `rh` to write; repeatable |
| `--surface NAME` | geometry and every alternate | surfaces to write; repeatable |
| `--attribute NAME` | every map and parcellation | maps to write; repeatable |
| `--prefix TEXT` | none | `sub-bert` gives `sub-bert_hemi-L_pial.surf.gii` |
| `--level N` | `0` | a coarser level is a decimated mesh with its own vertex numbering |

`--format gifti` is optional: a surface store exported to a path with no extension
writes GIFTI. Maps go to `.shape.gii` (morphometry), `.func.gii` or `.label.gii`.
Vertices keep the source numbering, so ingesting the directory again (export into an
empty one) rebuilds the store. Coordinates are written as stored, so a FreeSurfer store
in scanner RAS exports in scanner RAS.

## Pyramids, reading back, Python

Surface stores are mesh stores. Build their pyramid by edge-collapse decimation, which
carries maps, parcellation codes and `coords_<name>` to every level
([Meshes → Pyramids](meshes.md#pyramids)). Reading a hemisphere back in source vertex
order, and per-parcel summaries: [algorithms](../algorithms/streamlines_skeletons.md).
Python entry points: `zarr_vectors_tools.convert.ingest.freesurfer.ingest_freesurfer`,
`zarr_vectors_tools.convert.ingest.gifti.ingest_gifti`,
`zarr_vectors_tools.convert.ingest.cifti.attach_cifti` and
`zarr_vectors_tools.convert.export.gifti.export_gifti`.
