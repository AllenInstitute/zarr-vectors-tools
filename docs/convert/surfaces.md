# Cortical surfaces

Convert FreeSurfer subjects and GIFTI files to surface stores, attach CIFTI
maps to them, and export to GIFTI. Chunk shape, `--overwrite` and the export
options are in [Converting files](index.md).

| Format | In / out | Extra | INPUT |
|---|---|---|---|
| `freesurfer` | in | `surfaces` | a subject directory (holding `surf/` and `label/`), or its `surf/` |
| `gifti` | in, out | `surfaces` | one `.gii` file, or a directory of them |
| CIFTI | attach only | `surfaces` | `.dscalar.nii`, `.dlabel.nii`, `.dtseries.nii` |

```bash
zvtools convert subjects/bert bert.zv --chunk-shape 20,20,20
```

```python
from zarr_vectors_tools.convert.ingest.gifti import ingest_gifti

ingest_gifti("sub-01/anat", "sub-01.zv", (20.0, 20.0, 20.0), geometry="pial")
```

`--chunk-shape` is in mm. Each GIFTI file's kind, hemisphere and name come from
its metadata, else its file name.

| Option | Format | Default | Effect |
| --- | --- | --- | --- |
| `--geometry NAME` | both | `midthickness` (GIFTI: the first present of midthickness, pial, white, smoothwm, orig) | The surface to chunk. |
| `--hemisphere NAME` | both | both present | `lh` or `rh` (`left`, `right`). For GIFTI, given once, it assigns files that name neither. |
| `--space MODE` | both | `auto` | `auto`, `scanner` or `surface`; see [Coordinates](#coordinates). |
| `--surface NAME` | freesurfer | white, pial, inflated, sphere | Other surfaces to keep (repeatable). |
| `--morph NAME` | freesurfer | thickness, curv, sulc, area | Maps from `surf/?h.<name>` (repeatable). |
| `--annot NAME` | freesurfer | aparc | Parcellations from `label/?h.<name>.annot` (repeatable). |

FreeSurfer defaults take the files that exist; a name you pass must exist.
A GIFTI map whose vertex count differs from its surface is refused.

## What the store holds

- One mesh object per hemisphere (left 0, right 1), also as groups `left` and
  `right`.
- The geometry surface as positions. Every other surface is a 3-column vertex
  attribute, `coords_<name>`.
- Maps as float32 vertex attributes (`thickness`). Parcellations as int32 codes,
  with names and colours in the `surface` header.
- `zv_join_key`: each vertex's hemisphere and source vertex number. CIFTI attach
  and GIFTI export use it.

Every name: [Headers and attributes](../reference/headers_attributes.md).

## Coordinates

FreeSurfer surfaces are in *surface RAS*. The same subject's T1 and
tractography are in *scanner RAS*, one translation `c_ras` away. With
`--space auto`, the ingest applies a known transform to anatomical surfaces:
FreeSurfer's `c_ras`, or a GIFTI file's matrix to a named space. `scanner`
requires a transform to scanner RAS; `surface` keeps positions as stored. Inflated
and spherical surfaces are never moved. The summary prints the `space` used.

To view surfaces over the same subject's TRK, convert the TRK with
`--apply-affine` ([Streamlines](streamlines.md#coordinates)).

## CIFTI with zvtools attach

```bash
zvtools attach bert.zv sub-bert.MyelinMap.dscalar.nii
```

Each map becomes a vertex attribute: float32 for `.dscalar.nii`, int32 codes
for `.dlabel.nii`, one column per time point for `.dtseries.nii`. `--name` sets
the name. Vertices the file leaves out, such as the medial wall, get NaN (-1
for labels). Subcortical voxels are skipped. The file must be on the store's
mesh. Attributes go to level 0, so attach before building a pyramid.

## Export to GIFTI

```bash
zvtools convert bert.zv bert_gifti --prefix sub-bert
```

A surface store exported to a path with no extension writes GIFTI: one
`.surf.gii` per hemisphere and surface, and maps as `.shape.gii`, `.func.gii`
or `.label.gii`. Vertices keep the source numbering, so the directory ingests
back to the same store.

Build the pyramid with `--method mesh_decimate`
([Meshes](meshes.md#pyramids)). Reading a hemisphere back and per-parcel
summaries: [Cortical surfaces](../algorithms/streamlines_skeletons.md#cortical-surfaces).
