# Visualise in Neuroglancer

Zarr Vectors stores open in the
[BRIDGE Neuroscience fork of Neuroglancer](https://github.com/BRIDGE-Neuroscience/neuroglancer/tree/zarr_vectors_roi_store),
which adds a `zarr-vectors` data source, region filtering and tract export. Run it
locally; for stores in a public bucket the hosted build at
<https://zarr-vectors-viewer.web.app> also works.

Stock Neuroglancer cannot read a store. For it, export a skeleton, graph or mesh
store as a precomputed layer ([Skeletons](convert/skeletons.md),
[Meshes](convert/meshes.md)); streamlines and point clouds have no such export:

```bash
zvtools convert neurons.zv neurons_precomputed --format precomputed
```

## Quick start

Write a store with a pyramid ([Streamlines](convert/streamlines.md), [Pyramids](pyramids.md)),
then get the fork and start the viewer (Node.js 22.18 or later):

```bash
zvtools convert tracts.trk tracts.zv --apply-affine --compute-length \
    --coarsen 1,1,1 --sparsity 2,2,2 --rdp-tolerance 0.5,1,2 --chunk-scale 2,2,2
git clone --depth 1 -b zarr_vectors_roi_store https://github.com/BRIDGE-Neuroscience/neuroglancer.git
cd neuroglancer
npm i
npm run dev-server        # viewer at http://localhost:8080; leave it running
# in a second terminal, in the same checkout, serve the folder holding the store:
npx http-server /path/to/data -p 9000 --cors
```

Open <http://localhost:8080>, click **+** in the layer bar, paste the source URL and
press Enter twice (the second Enter creates a segmentation layer):

```text
http://127.0.0.1:9000/tracts.zv/|zarr-vectors:
```

## Serving requirements

| The server must provide | Because | Without it |
|---|---|---|
| CORS: `Access-Control-Allow-Origin: *` or the viewer's origin | The viewer is a web page reading another origin | "blocked by CORS policy" |
| Directory listing | Vertex attributes, object attributes, groups and mesh faces that cross a chunk face are found by listing | Attributes disappear ("could not list vertex_attributes/"); meshes crack along every chunk face |
| HTTP Range requests | A [sharded](store_layout.md#sharding) store is read in byte ranges | "Raw-format chunk is … bytes" on every chunk |

| Local server, run in the fork checkout | CORS | Listing | Range |
|---|---|---|---|
| `npx http-server DIR -p 9000 --cors` | yes | yes | yes |
| `python cors_webserver.py -d DIR -p 9000` (no Node.js) | yes | yes | no: first run `zvtools shard STORE --unshard` |

On Google Cloud Storage use the `gs://` form, grant `allUsers` the Storage Object
Viewer role (read and list), and set a bucket CORS policy allowing `GET` with the
`Range` header from the viewer's origin (or `*`). S3 (`s3://`) needs the same:
public read and list, plus CORS.

## Source URL

```text
<store URL>/|zarr-vectors:                   e.g. gs://bucket/brain/tracts.zv/|zarr-vectors:
<store URL>/|zarr-vectors:#attributes=a,b    load only vertex attributes a and b
<store URL>/|zarr-vectors:#attributes=       load no vertex attributes
zarr-vectors://<store URL>                   older form, still accepted
```

The viewer never detects a store by itself, so always add `|zarr-vectors:`. Query
parameters (`?…`) are refused. Percent-encode attribute names containing `,` or `&`.

## How each geometry renders

| Geometry | Drawn as | Default colour | Export tab |
|---|---|---|---|
| `point_cloud` | a dot per vertex; no objects, so no segment properties | per point | no |
| `streamline`, `polyline`, `line` | lines | direction (x red, y green, z blue) | yes |
| `skeleton` | lines, including branches | per object | yes |
| `graph` | lines along the stored edges | direction | yes |
| `mesh` | triangles | per object | no |

## Preparing a store

| Check | Why |
|---|---|
| Per-chunk arrays uncompressed (the default, [Compressor](store_layout.md#compressor)) | The viewer reads chunks raw; a `zstd` or `blosc` store draws nothing |
| Two or more levels ([Pyramids](pyramids.md)) | With one level the viewer draws that level at every zoom, so a whole view needs all of it in GPU memory |
| Growing chunks (`--chunk-scale 2,…`) in a pyramid that keeps every object (meshes, graphs, `--sparsity 1`) | Levels with the same objects and the same chunk size tie, and the viewer then draws the coarsest at every zoom |
| Meshes coarsened with `--method mesh_decimate` ([Meshes](convert/meshes.md#pyramids)) | Clustering leaves stray triangles at coarse levels |
| Single-column vertex attributes | A multi-column one (LAS `color`, OBJ `normal`) stops every chunk loading. Open such a store with `#attributes=` naming only single-column attributes, or with `#attributes=` alone |
| 3-D positions | A 2-D store (two h5ad embedding columns, say) fails: "a rank-2 store of this geometry has nothing to render" |
| TRK ingested with `--apply-affine` | Without it the store keeps TrackVis voxel-mm coordinates (2 to 181 mm on a test file, against −89 to 90 mm RAS) and the viewer ignores the stored affine, so tracts sit apart from other RAS data |
| Units | Read from the store: TRK, TCK and TRX stores record millimetres; inputs without units (SWC, OBJ, CSV) open unitless |
| At most 65,536 objects | Needed for object attributes in the viewer ([below](#objects-attributes-and-colour-by)) |

## Sizing the pyramid for the GPU budget

The viewer keeps drawn geometry within its GPU memory limit (Settings, gear icon →
**GPU memory limit**; 1 GB by default). It costs each level from its `vertex_count`,
which zvtools writes. Per vertex it counts 12 bytes of position, 12 of direction
(streamlines, lines, skeletons, graphs), 16 of object id and edge, and 4 per loaded
vertex attribute: 40 bytes for a streamline store without vertex attributes, 28 for a
point cloud or mesh. So 1 GB holds 25 M streamline vertices: a tractogram of 100 M level-0
vertices (about 4 GB) needs a coarser level of at most 25 M vertices for a
whole-brain view, fewer if other layers share the GPU. Add levels until the coarsest
fits with room to spare ([recipes](pyramids.md#recommended-recipes)).

If the volume draws only in part, raise the limit, or tick **Ignore memory ceiling**
on the Render tab (a wide view can then exhaust GPU memory). **Detail focus** there
spends leftover memory near the camera (`local`) or on whole objects (`object`).
Prefer `object`; it needs `object_attributes/vertex_count` at every level, which
zvtools writes for every store with objects. `object` draws each object from the
coarsest level that kept it: full detail where sparsity drops objects (streamlines
and skeletons, for example), the coarsest level of a mesh or graph pyramid.
`local` has two faults on a pyramid whose chunks grow: it draws a coarse chunk and the
finer chunks inside it at once (doubled lines, flickering mesh faces) or leaves holes,
and it overruns a small memory limit, leaving most of the view blank. "Ignore memory
ceiling" has no effect in `local`.

## Objects, attributes and colour-by

**Colour by (background)** on the Render tab offers **Direction (tangent)** where the
geometry has one, **Vertex:** per loaded vertex attribute and **Object:** per scalar
object attribute; **Filter by (background)** hides objects outside an object
attribute's range. The **Seg.** tab lists object attributes (e.g. `length`) as numerical properties.

- Groups (TRX groups, for example) become tags: type `#cst` in the **Seg.** tab's
  search box to list group `cst` and show or hide it.
- Object columns are read from one 65,536-row chunk, the size zvtools writes; with more
  objects, object attributes are lost ("spans multiple chunks … skipping").
- Vertex attributes: float32, (u)int8/16/32, or 64-bit (converted to float32, so
  integers above 2^24 lose precision); h5ad categories show as integer codes. The
  first 32 load (declared order, then alphabetical) unless `#attributes=` names them;
  one named `tangent` is ignored. Coarse point-cloud levels carry them as bin
  means ([Pyramids](pyramids.md)).

## Filtering and exporting dissections

The **Filter** tab works on every geometry. **+ New group**, then **+ Sphere**, **+ Box**
or **+ Plane…**, adds a region at the crosshair with a **Role** (Include, Exclude; later
regions also Or) and a **Test**: **Crosses** (a segment passes through) or **Point
inside** (a vertex lies inside); the tab counts what passes. **By segmentation label**
(pick a segmentation layer as **Parcellation**) and **By attribute** build groups from
labels or attribute values. Each group has its own **Opacity**, **Colour by** and
**Filter by attribute…**; **⇗** moves it to its own layer. **Save to store** and
**Browse saved…** share groups via a public GCS bucket (saving needs Google sign-in).

The **Export** tab writes the passing objects, or the whole store, of a line-type
store. Set **Format** to **New zarr-vectors store** and click **Download job spec**
(it saves `dissection.json`). Then, in the fork checkout, with zarr-vectors-tools installed,
`pip install neuroglancer` for its dependencies, and the store's server still running:

```bash
PYTHONPATH=python python -m neuroglancer.tract_export dissection.json --dry-run   # counts only
PYTHONPATH=python python -m neuroglancer.tract_export dissection.json -o dissection.zv
zvtools convert dissection.zv dissection.trk
```

Or paste the URL that `PYTHONPATH=python python -m neuroglancer.tract_export --serve`
prints into the tab's **Exporter URL**: **Download** then writes the store where it runs.

:::{warning}
The fork's TRK format does not work yet: in the browser **Download** reports "No
whole tracts are loaded at high detail", and the job runner asks for the `[trk]`
extra even when it is installed. Export a store and convert it as above.
:::

## Troubleshooting

| Symptom (browser console or layer panel) | Cause | Fix |
|---|---|---|
| "No zarr.json found at … is this a zarr v3 store?" | Wrong URL | Point the URL at the store directory |
| "Permission was denied … address space" | Hosted build reading a local server | Run the viewer locally |
| Nothing draws; "vlen-bytes chunk truncated" | Compressed store | Rewrite it uncompressed |
| Nothing draws; "Raw-format chunk is … bytes" | Sharded store, server without Range | `http-server`, or `zvtools shard STORE --unshard` |
| Nothing draws; "dtype=float32 expected N bytes …, got 3N" | Multi-column vertex attribute | `#attributes=` with single-column names |
| No **Vertex:**/**Object:** options; "could not list …" (harmless on a store without attributes) | No directory listing | Serve with listing; on GCS grant list access |
| Tracts offset from MNI or other RAS data | TRK ingested without `--apply-affine` | Re-ingest with `--apply-affine` |
| "store zv_version … predates the 0.9.0 layout" | Store older than Zarr Vectors format 0.9 | Rewrite it with current zvtools |
| Parts of the volume missing | GPU memory limit reached; `local` detail focus misjudges a small limit | Raise the limit, switch detail focus to `object`, or add coarser levels |
| Doubled lines, flickering faces or holes where levels meet | `local` detail focus on a pyramid with growing chunks | Detail focus `object` |
| A mesh or graph never gains detail when zooming in | Detail focus `object`, or levels that keep every object and one chunk size | Detail focus `local`, on a pyramid built with `--chunk-scale 2,…` |
| Line segments show only near their two ends | A segment is drawn only inside the chunks holding its end points | Chunks several times longer than most segments |
| Long straight chords across SWC trees at full detail | This viewer version adds edges on level 0 of an SWC store | Zoom out to level 1, or view a precomputed export ([Skeletons](convert/skeletons.md)) |
| Meshes crack along chunk faces | No directory listing | Serve with listing |
