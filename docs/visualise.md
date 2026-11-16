# Visualise in Neuroglancer

Stores open in ngplatform, a Neuroglancer build with a `zarr-vectors` data
source. This page covers what zvtools adds; the core package's
{zvpy}`Neuroglancer guide <how_to/view_in_neuroglancer.html>` covers the viewer.

## Source URL

```text
<store URL>/|zarr-vectors:                   e.g. http://127.0.0.1:8000/tracts.zv/|zarr-vectors:
<store URL>/|zarr-vectors:#attributes=a,b    load only vertex attributes a and b
```

Percent-encode attribute names containing `,` or `&`.

## Serve files without converting

`zvtools virtual serve` serves each file as a store, built from the file on
request, and prints its source URL:

```bash
zvtools virtual serve tracts.trk surface.obj --port 9000
```

It reads TRK, TCK and TRX tractograms, OBJ, STL, PLY, GIFTI and FreeSurfer
surfaces, SWC neurons (`folder/@all.swc` reads a folder as one store) and CSV,
TSV and XYZ point tables. In Python, `zarr_vectors_tools.virtual.open_virtual(path)`
opens a file as a store the same way. A served file has one level, so convert a large file
with a pyramid instead.

## Serving requirements

The server must send CORS headers, answer byte ranges and list folders, as
`zvtools virtual serve` does. Without folder listings the viewer cannot find
object attributes, and mesh faces that cross a chunk face are missing.

## Preparing a store

| Check | Why |
|---|---|
| Two or more levels ([Pyramids](pyramids.md)) | With one level the viewer draws level 0 at every zoom |
| Mesh levels with growing chunks (`--chunk-scale 2,…`) | The viewer draws multi-resolution meshes only from levels whose chunks double |
| Mesh faces indexed with `zvtools index-faces` | The viewer reads one object's faces by byte range, not whole chunks |
| Uncompressed arrays (the default) | The viewer reads an object's rows by byte range; a compressed store reads whole chunks |
| TRK converted with `--apply-affine` | Otherwise tracts stay in TrackVis voxel-mm and sit apart from RAS data |

## Sizing the pyramid for the GPU budget

The viewer draws the finest level whose vertices fit its target density on
screen, using each level's `vertex_count`. Add levels until the coarsest is
small enough for a whole view ([recipes](pyramids.md#recommended-recipes)).

## Stock Neuroglancer

Stock Neuroglancer cannot read a store. Export a skeleton, graph or mesh store
as a precomputed layer instead; streamlines and point clouds have no such
export:

```bash
zvtools convert neurons.zv neurons_precomputed --format precomputed
```
