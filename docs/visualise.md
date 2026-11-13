# Visualise in Neuroglancer

Zarr Vectors stores open in ngplatform, a Neuroglancer build with a
`zarr-vectors` data source. The parent package's
{zvpy}`Neuroglancer guide <how_to/view_in_neuroglancer.html>` covers the
viewer: the live site, a local build, opening stores, serving and levels of
detail. This page covers what zvtools adds.

## Source URL

Point the viewer at a store, or at a file served by `zvtools virtual serve`,
with the `zarr-vectors` data source:

```text
<store URL>/|zarr-vectors:                   e.g. http://127.0.0.1:8000/tracts.zv/|zarr-vectors:
<store URL>/|zarr-vectors:#attributes=a,b    load only vertex attributes a and b
```

Percent-encode attribute names containing `,` or `&`.

## Serve files without converting

`zvtools virtual serve` serves each file as a store, made from the file on
request:

```bash
zvtools virtual serve tracts.trk surface.obj --port 9000
```

It prints one source URL per file, such as
`http://127.0.0.1:9000/tracts.trk/|zarr-vectors:`. It reads TRK, TCK and TRX
tractograms (TRK in RAS mm), OBJ, STL, PLY, GIFTI and FreeSurfer surfaces, SWC
neurons (`folder/@all.swc` reads every SWC file in a folder as one store) and
CSV, TSV and XYZ point tables. `--host` defaults to `127.0.0.1` and `--port` to
`8000`.

A served file has one level, so the viewer draws all of it at every zoom.
Convert a large file, with a pyramid, instead.

## Serving requirements

`zvtools virtual serve` answers with CORS headers, byte ranges and HTML folder
listings. A converted store needs a server that does the same; see the
{zvpy}`Neuroglancer guide <how_to/view_in_neuroglancer.html>`. Without
folder listings, the viewer cannot find a store's object attributes, and a
mesh's faces that cross a chunk face are missing.

## Preparing a store

| Check | Why |
|---|---|
| Two or more levels ([Pyramids](pyramids.md)) | With one level the viewer draws level 0 at every zoom |
| Mesh levels with growing chunks (`--chunk-scale 2,…`) | The viewer draws multi-resolution meshes only from levels whose chunks double; otherwise meshes stay at level 0 |
| Mesh faces indexed with `zvtools index-faces` | The viewer reads one object's faces alone, by byte range, instead of whole chunks |
| Uncompressed per-chunk arrays (the default, [Compressor](store_layout.md#compressor)) | The viewer reads an object's rows by byte range; a compressed store works, but whole chunks are read |
| TRK converted with `--apply-affine` | Without it the store keeps TrackVis voxel-mm coordinates, and tracts sit apart from RAS data |
| Units | The viewer reads them from the store: TRK, TCK, TRX, GIFTI and FreeSurfer stores record millimetres; inputs without units (SWC, OBJ, CSV) open unitless |

## Sizing the pyramid for the GPU budget

The viewer draws one level per view: the finest whose vertices fit its target
density on screen. It takes each level's size from the level's `vertex_count`,
which zvtools writes at every level. Add levels until the coarsest is small
enough for a whole view ([recipes](pyramids.md#recommended-recipes)).

## Stock Neuroglancer

Stock Neuroglancer cannot read a store. For it, export a skeleton, graph or mesh
store as a precomputed layer ([Skeletons](convert/skeletons.md),
[Meshes](convert/meshes.md)); streamlines and point clouds have no such export:

```bash
zvtools convert neurons.zv neurons_precomputed --format precomputed
```

To select streamlines by region and export them as a file, see
[Streamlines, skeletons and surfaces](algorithms/streamlines_skeletons.md).
