# How this package relates to `zarr-vectors-py`

`zarr-vectors-tools` is an **extension of**
[`zarr-vectors-py`](https://zarr-vectors-py.readthedocs.io/en/latest). It
adds workflows on top of that package; it does not reimplement it, and
this documentation does not re-describe it.

If you have not met Zarr Vectors before, read the parent
package's introduction first — it is the one place the format is
explained:

| Read this first | Covers |
| --- | --- |
| {zvpy}`Core concepts <getting_started/concepts.html>` | What the format is, why chunked vector geometry, and the ideas it is built from |
| {zvpy}`Quickstart <getting_started/quickstart.html>` | Creating a store, writing, querying, building a pyramid |
| {zvpy}`the specification <spec/index.html>` | The normative specification: store structure, metadata, spatial indexing, links, conformance |
| {zvpy}`Geometry types <spec/geometry_types/index.html>` | The geometry types — point cloud, line, polyline, graph, skeleton, mesh — and what distinguishes them |

Everything on this page is about the **division of labour** between the
two packages. Nothing on it describes the format.

## Who owns what

| | [`zarr-vectors-py`](https://zarr-vectors-py.readthedocs.io/en/latest) | `zarr-vectors-tools` (this package) |
| --- | --- | --- |
| Owns | the format, and the Python API over it | everything built on top of it |
| Provides | store layout, chunk and bin geometry, fragments, links, the object model, resolution-level metadata, readers and writers, lazy access, validation | file-format conversion, streaming algorithms, the rich multiresolution layer, the `zvtools` CLI |
| Supported surfaces | `zarr_vectors.api`, `zarr_vectors.building` | `zarr_vectors_tools.*` |
| Dependencies | deliberately light | heavy optional readers, gated behind extras |

The split is deliberate. The core stays installable anywhere; the
third-party readers (`nibabel`, `laspy`, `cloud-volume`, `networkx`) and
the heavier algorithms live out here.

## Which core surface this package builds on

`zarr-vectors-py` promises two module surfaces and declares the rest
internal. This package uses only the promised ones, and so should any code
you write against it:

- **`zarr_vectors.api`** — the data-oriented surface: open a store, select
  a region or a set of objects, read it back, edit it. Re-exported from the
  top-level package, so `zarr_vectors.open(...)` works.
- **`zarr_vectors.building`** — the builder surface, for code whose job
  *is* the physical layout: ingest converters, pyramid builders, exporters.
  This package routes every core call through it.

Anything else in `zarr_vectors` — `core`, `encoding`, `lazy`, `ops`,
`spatial`, `multiresolution`, `rechunk` — is **internal to the parent
package and changes without notice**. Do not import from it, and do not
take a name you find in this documentation to be a promise about it.
Core answers the question itself:

```python
import zarr_vectors as zv

zv.stability("zarr_vectors.building")        # 'supported'
zv.stability("zarr_vectors.multiresolution") # 'internal'
```

The tiers are explained at {zvpy}`the API reference <api/index.html>`.

## How the multiresolution split works

Multiresolution is split across the two packages, and it is the one place
where core dispatches *into* this one.

Core ships a basic, dependency-free `per_object` binning pyramid with
`random` object selection, plus a **plug-in registry**. The rich
strategies — skeleton and polyline coarsening, spatial-coverage and
length-ranked selection — live here and register themselves into that
registry at import time, through core's supported
`zarr_vectors.building.register_coarsen_strategy` and
`register_selection_strategy` entry points.

So importing this package widens what core will accept, without core
taking a dependency on it:

```python
import zarr_vectors as zv

zv.coarsen_methods()          # ('per_object',)

import zarr_vectors_tools     # noqa: F401  — registers on import

zv.coarsen_methods()          # ('per_object', 'polyline', 'skeleton')
```

The registration is one-directional: tools depends on core, core never
depends on tools, and the registration degrades quietly against a core too
old to have the registry.

:::{note}
Pyramid *refresh* is the one piece not injected this way. Core's editing
path uses core's own basic refresher. To get the rich re-coarsening after
an edit, call [`rebuild_pyramid_from_level`](../multiresolution/refresh.md)
yourself.
:::

## Where to go next

| If you want to… | Go to |
| --- | --- |
| understand the format itself | {zvpy}`Core concepts <getting_started/concepts.html>` — **the parent package** |
| read the normative format definition | {zvpy}`the specification <spec/index.html>` |
| use the core read/write API directly | {zvpy}`the data API <api/api.html>` |
| install and convert your first file | [Installation](installation.md), then [Quickstart](quickstart.md) |
| understand how *this* package's pieces compose | [Concepts](concepts.md) |
| work from the terminal | [The `zvtools` CLI](cli.md) |
| find a function in this package | [Modules](../modules/index.md) or the [API reference](../api/index.rst) |
| convert a specific file format | [Ingest workflows](../ingest/index.md) |
| build resolution pyramids | [Coarsening versus sparsity](../multiresolution/concepts.md) |
