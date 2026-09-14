# Cortical surfaces

Three sources, one store layout:

| Source | What it holds | Entry point | CLI |
| --- | --- | --- | --- |
| GIFTI (`.surf.gii`, `.shape.gii`, `.func.gii`, `.label.gii`) | surfaces, per-vertex maps, parcellations | `ingest.gifti.ingest_gifti` | `zvtools convert DIR out.zv` |
| FreeSurfer subject (`surf/`, `label/`) | surfaces, morphometry, `.annot` parcellations | `ingest.freesurfer.ingest_freesurfer` | `zvtools convert SUBJECT out.zv` |
| CIFTI-2 (`.dscalar.nii`, `.dlabel.nii`, `.dtseries.nii`) | per-vertex maps with no geometry | `ingest.cifti.attach_cifti` | `zvtools attach out.zv FILE` |

All three need `nibabel`:

```bash
pip install "zarr-vectors-tools[surfaces]"
```

## The store

A subject's cortex is always several files that share one vertex numbering
per hemisphere. The store keeps that structure:

- **One mesh object per hemisphere.** Left is object 0 and right is object 1
  when both are present. They are also named groups (`left`, `right`), so
  `split --by groups` cuts a store into one per hemisphere.
- **One surface is the geometry.** It is the surface that gets chunked and
  answers bounding-box queries. It must be anatomical: `midthickness`
  (preferred), `pial`, `white`, `smoothwm` or `orig`.
- **Every other surface is an attribute.** Pial, white, inflated, sphere and
  flat become three-column `coords_<name>` attributes on the same vertices,
  so a viewer can switch between them without a second store.
- **Maps are attributes.** Continuous maps are float attributes named after
  what they measure (`thickness`, `curv`, `myelin`). Parcellations are int32
  codes, and their names and colours go in the header's label table.
- **Every vertex has a join key.** `zv_join_key` is
  `hemisphere << 32 | source_vertex`, with left fixed at 0 and right at 1.
  Chunking reorders vertices, and the key is how data indexed by the
  original vertex number still lands on the right one.

A map present for only one hemisphere is kept, not dropped: the other
hemisphere is filled with `NaN`, or `-1` for labels. The summary lists those
maps under `filled`.

## GIFTI: `ingest_gifti`

```python
from zarr_vectors_tools.ingest.gifti import ingest_gifti

ingest_gifti(
    "derivatives/fmriprep/sub-01/anat",   # a directory, a list of files, or one file
    "sub-01_surfaces.zv",
    (20.0, 20.0, 20.0),                   # chunk_shape in mm
    geometry=None,                        # default: midthickness > pial > white > ...
    hemisphere=None,                      # only for files that name no hemisphere
)
```

Each file is classified from its own metadata first and its name second:

- **Kind.** A `POINTSET` plus `TRIANGLE` array is a surface. A `LABEL` array
  is a parcellation. Anything else is a scalar map.
- **Hemisphere.** From `AnatomicalStructurePrimary`, else from the filename
  (`hemi-L`, `.L.`, `lh.`, `left`).
- **Surface name.** From `GeometricType` for the shapes that are not
  anatomy, then the filename, then `AnatomicalStructureSecondary`. The order
  matters: HCP writes its inflated surface with
  `AnatomicalStructureSecondary=MidThickness`, because it was inflated from
  the midthickness.
- **Attribute name.** From the filename, with the subject, hemisphere, mesh
  and file-type parts removed. A BIDS `desc-` entity wins.

| Filename | Attribute |
| --- | --- |
| `sub-01_hemi-L_thickness.shape.gii` | `thickness` |
| `100307.L.MyelinMap_BC.32k_fs_LR.func.gii` | `MyelinMap_BC` |
| `sub-01_hemi-R_desc-aparc.label.gii` | `aparc` |
| `lh.aparc.a2009s.label.gii` | `aparc_a2009s` |

A file with several distinctly named arrays gives one attribute per array.
Unnamed arrays, and any time series, give one multi-column attribute.

Positions are stored as the file has them. The declared dataspace, such as
`talairach` or `scanner`, is recorded in the header's `space` field but never
applied, because GIFTI positions are already in it.

:::{warning}
A map with a different vertex count from its surface is refused, and the
error names both counts. The usual cause is mixing mesh resolutions: a 32k
fs_LR metric next to a native or 164k surface. Resample one to the other
first, for example with `wb_command -metric-resample`.
:::

## FreeSurfer: `ingest_freesurfer`

```python
from zarr_vectors_tools.ingest.freesurfer import ingest_freesurfer

ingest_freesurfer(
    "subjects/bert",                 # the subject directory, or its surf/
    "bert_surfaces.zv",
    (20.0, 20.0, 20.0),
    geometry="midthickness",         # computed from white and pial if there is no file
    alternates=None,                 # default: white, pial, inflated, sphere if present
    morphometry=None,                # default: thickness, curv, sulc, area if present
    annotations=None,                # default: aparc if present; e.g. ["aparc.a2009s"]
    hemispheres=None,                # default: both that exist
    space="auto",                    # "auto" | "scanner" | "surface"
)
```

Defaults are opportunistic, because not every subject has every file. A
name you pass yourself must exist, and a missing one is an error.

### Coordinates: surface RAS and scanner RAS

FreeSurfer writes surfaces in *surface RAS*, centred on the conformed
volume. The T1, the diffusion data, TRK and TRX tractography and fMRIPrep's
GIFTI surfaces are all in *scanner RAS*. The two differ by one translation,
`c_ras`, which each surface file records in its footer.

| `space` | Result |
| --- | --- |
| `"auto"` (default) | Scanner RAS when the files record `c_ras`, surface RAS otherwise |
| `"scanner"` | Scanner RAS, or an error when `c_ras` is missing |
| `"surface"` | FreeSurfer's own coordinates, untouched |

Only anatomical surfaces are shifted. Inflated and spherical surfaces are in
no anatomical space, so they stay as written. `c_ras` is recorded in the
header either way, with `c_ras_applied` saying whether it was added.

Use scanner RAS whenever the surfaces will be viewed or queried next to a
tractogram of the same subject. Otherwise the cortex and the tracts sit
apart by `c_ras`, often by a few centimetres.

### Midthickness

`recon-all` writes no midthickness surface, but it is the best default
geometry. It lies halfway through cortex, which makes it the least biased
surface for sampling volume data. When there is no `?h.midthickness` file,
it is computed as the vertex-wise mean of white and pial. The header's
`midthickness_computed` field records that.

## CIFTI: `attach_cifti`

CIFTI files carry values but no geometry. They index vertices of a surface
you already have, so they are attached to a surface store rather than
ingested:

```python
from zarr_vectors_tools.ingest.cifti import attach_cifti

attach_cifti("sub-01_surfaces.zv", "sub-01.MyelinMap_BC.32k_fs_LR.dscalar.nii")
attach_cifti("sub-01_surfaces.zv", "Glasser_MMP1.32k_fs_LR.dlabel.nii")
attach_cifti("sub-01_surfaces.zv", "rfMRI_REST1_LR.dtseries.nii", name="rest1_lr")
```

| File | Written as |
| --- | --- |
| `.dscalar.nii`, one map | one float attribute, named from the file |
| `.dscalar.nii`, several named maps | one float attribute per map; `name=` keeps them together as one multi-column attribute instead |
| `.dlabel.nii` | one int32 attribute per map, with its label table in the header |
| `.dtseries.nii` | one `(V, T)` float attribute; start, step and unit go in the header's `series` entry |

Each value's vertex comes from the file's brain-model axis. Vertices the file
leaves out, which is normally the medial wall, get `NaN` or `-1`. Pass
`missing="error"` to refuse instead. Subcortical voxels have no vertex to
land on and are skipped, and the summary counts them.

:::{warning}
A file defined on a different mesh from the store is refused. Joining a 32k
fs_LR map onto a 164k surface would otherwise match the first 32k vertex
numbers and put every value on the wrong vertex.
:::

Attributes are written to level 0. Attach before building a pyramid, or
rebuild the pyramid afterwards so the coarse levels carry the new maps.

## From the command line

A directory is recognised by its contents, so neither form needs `--format`:

```bash
# fMRIPrep / HCP GIFTI directory, with one coarser level.
zvtools convert sub-01/anat sub-01_surfaces.zv --chunk-shape 20,20,20 \
    --coarsen 4 --sparsity 1

# FreeSurfer subject in scanner RAS, with the Destrieux atlas as well.
zvtools convert subjects/bert bert.zv --chunk-shape 20,20,20 \
    --annot aparc --annot aparc.a2009s

# Add CIFTI maps.
zvtools attach sub-01_surfaces.zv sub-01.MyelinMap_BC.32k_fs_LR.dscalar.nii
zvtools attach sub-01_surfaces.zv rest.dtseries.nii --name rest
```

| Flag | Applies to | Meaning |
| --- | --- | --- |
| `--geometry NAME` | gifti, freesurfer | surface to chunk |
| `--hemisphere left｜right｜lh｜rh` | gifti, freesurfer | freesurfer: which hemispheres to read, repeatable. gifti: the hemisphere of files that name none |
| `--space auto｜scanner｜surface` | freesurfer | coordinate space, see above |
| `--surface NAME` | freesurfer | alternate surfaces to keep, repeatable |
| `--morph NAME` | freesurfer | morphometry maps, repeatable |
| `--annot NAME` | freesurfer | parcellations from `label/`, repeatable |
| `--name NAME` | `attach` with CIFTI | attribute name |

Every one of these is refused by name for a format it does not apply to.

## Pyramids

Surface stores are mesh stores, so `zvtools pyramid` and `build_pyramid`
work unchanged. Continuous maps and the `coords_<name>` surfaces are
averaged within each vertex cluster. Parcellation codes take a value that
exists in the source, never an average of two parcels.

## The header

`HeaderRegistry(store).get("surface")` returns a `SurfaceHeader`:

| Field | Holds |
| --- | --- |
| `source` | `gifti` or `freesurfer` |
| `space` | the geometry's coordinate space |
| `geometry` | which surface was chunked |
| `hemispheres` | per hemisphere: `object_id`, `hemisphere`, CIFTI `structure`, `n_vertices`, `n_faces` |
| `alternates` | `coords_<name>` attribute to surface name |
| `scalars` | continuous attribute to the file it came from |
| `label_tables` | parcellation attribute to `{code: {"name", "rgba"}}` |
| `key_attribute` | the join-key attribute, `zv_join_key` |
| `c_ras` | FreeSurfer's scanner offset, when known |
| `extra` | FreeSurfer subject details and CIFTI `series` timing |

`header.object_for("left")` returns the object id holding that hemisphere.

## Not yet supported

- Writing surfaces back out as GIFTI.
- Per-parcel summaries, such as mean thickness per `aparc` region.
- Parcellated and connectivity CIFTI files (`.pscalar`, `.ptseries`,
  `.dconn`), which have no per-vertex axis.

## See also

- [Meshes](meshes.md) for OBJ and STL
- [Tractography](tractography.md) for the streamlines to view beside cortex
- [Headers](../headers.md)
- [The `zvtools` CLI](../getting_started/cli.md)
