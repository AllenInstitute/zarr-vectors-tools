# Getting started with Zarr Vectors

If you have arrived here from a neuroscience or geospatial pipeline and
have not met the Zarr Vector Format before, start on this page. It
explains what the format is for, the handful of ideas you need, and how
the two Python packages divide the work — before any code.

## The problem

Scientific vector geometry has outgrown its file formats. A whole-brain
tractogram is tens of millions of streamlines. An EM connectomics volume
yields hundreds of thousands of neuron skeletons. A MERFISH experiment
produces billions of transcript points.

Formats like TRK, SWC, PLY and OBJ share one assumption: **you read the
whole file.** There is no way to ask for "the streamlines passing through
this cubic millimetre" without parsing everything before it. That
assumption is fine at 10 MB and fatal at 100 GB — and it rules out
serving the data from cloud object storage, where you pay per request and
per byte.

Raster imaging solved this years ago with chunked, pyramidal formats
(OME-Zarr, Neuroglancer precomputed): cut the volume into chunks, index
them spatially, precompute downsampled levels, and let the viewer fetch
only what it is displaying. **Zarr Vectors applies the same idea to
vector geometry.**

## The idea

A Zarr Vectors store is a Zarr v3 group holding geometry that has been:

**Spatially chunked.** Space is divided by a regular grid. Each object's
vertices are assigned to the chunk they fall in, so a bounding-box query
touches only the chunks that intersect it. An object crossing a chunk
boundary is split into *fragments*, one per chunk, and reassembled on
read.

**Multiresolution.** Coarser resolution levels are precomputed, so a
viewer zoomed out fetches a decimated version instead of everything.

**Cloud-native.** It is Zarr, so a store on S3 or GCS is read with
ranged HTTP requests. No server, no database, no download-first step.

**Self-describing.** Coordinate axes, units, geometry types, bounds and
capabilities live in the store's metadata, so a reader knows what it has
without out-of-band configuration.

## The geometry types

The format defines a small set of geometry types. Every ingest path maps
onto exactly one:

| Geometry type | What it is | Typical source |
| --- | --- | --- |
| point cloud | unconnected vertices | MERFISH transcripts, LiDAR, cell centroids |
| line | independent segments, two endpoints each | vasculature segments, synaptic links |
| polyline | ordered vertex sequences | tractography streamlines |
| graph | vertices plus arbitrary edges | connectomes |
| skeleton | a graph constrained to a tree | neuron morphology, SWC, EM skeletons |
| mesh | vertices plus triangular faces | segmented surfaces |

The distinction between a graph and a skeleton is a real constraint, not
a label — skeletons get a rooted-tree representation and topology-aware
coarsening that a general graph cannot use.

## Four ideas worth knowing

**Chunk shape** is the physical extent of one chunk in store coordinates.
It sets the granularity of both I/O and spatial queries. Too large and
every query over-fetches; too small and you drown in requests and
metadata. See [Choosing chunk and bin shape](../how_to/choose_chunk_and_bin.md).

**Bin shape** is an optional finer grid *inside* each chunk, used as a
spatial-hash bucket by some accelerators. Most ingests can leave it unset.

**Fragments.** Because objects cross chunk boundaries, an object is stored
as an ordered list of fragments, and an *object index* records which
fragments belong to which object. This is what lets you read one neuron
without scanning the grid.

**Links** are connectivity — edges between vertices, or mesh faces. Under
format {{ zvf_version }} all connectivity lives in a single family at
`links/<delta>/<offsets>/`, where `delta` distinguishes intra-chunk from
cross-chunk relationships.

:::{warning}
If you have read older documentation or pre-0.9 code, you may have seen a
separate `cross_chunk_links/` group. **It no longer exists.** Links were
merged into one family. The practical consequence for algorithm authors
is the double-count trap: `read_links` now returns the *whole* family, so
the old idiom of reading per-chunk links and then adding cross-chunk
links counts every intra-chunk edge twice. See
[Algorithms](../algorithms/index.md).
:::

## The two packages

The work is split across two Python packages, and knowing which owns what
saves a lot of searching:

| | [`zarr-vectors-py`](https://zarr-vectors-py.readthedocs.io/en/latest) | `zarr-vectors-tools` (this package) |
| --- | --- | --- |
| Owns | the format itself | everything built on it |
| Provides | chunk encoding, spatial index, links, readers/writers, lazy access, validation, Neuroglancer integration | file-format conversion, streaming algorithms, the rich multiresolution layer, the `zvtools` CLI |
| Dependencies | deliberately light | heavy optional readers, gated behind extras |

The split is deliberate. The core stays installable anywhere; the
third-party readers (`nibabel`, `laspy`, `cloud-volume`, `networkx`) and
the heavier algorithms live out here.

Multiresolution is split the same way and it is worth being explicit: core
ships a basic, dependency-free `per_object` binning pyramid with `random`
object selection, plus a **plug-in registry**. The rich strategies —
skeleton and polyline coarsening, spatial-coverage and length-ranked
selection — live in this package and register themselves into that
registry at import time. So `zarr_vectors.multiresolution.coarsen_level(
method="skeleton")` works once `zarr_vectors_tools` has been imported,
without core taking a dependency on tools.

## Where to go next

| If you want to… | Go to |
| --- | --- |
| install and convert your first file | [Installation](installation.md), then [Quickstart](quickstart.md) |
| understand how the pieces compose | [Concepts](concepts.md) |
| work from the terminal | [The `zvtools` CLI](cli.md) |
| find the right function | [Modules](../modules/index.md) or the [API reference](../api/index.rst) |
| convert a specific file format | [Ingest workflows](../ingest/index.md) |
| build resolution pyramids | [Coarsening versus sparsity](../multiresolution/concepts.md) |
| read the normative format definition | the [specification site](https://alleninstitute.github.io/zarr_vectors/) |

## See also

- [Main library documentation](https://zarr-vectors-py.readthedocs.io/en/latest) — `zarr-vectors-py`.
- [Zarr Vector Format specification](https://alleninstitute.github.io/zarr_vectors/) — the normative reference.
- [Metadata schema](https://alleninstitute.github.io/zarr_vectors/08-metadata.html) — LinkML source and generated JSON Schema.
