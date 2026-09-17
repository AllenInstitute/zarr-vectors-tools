.. zarr-vectors-tools documentation master file

.. image:: zarr-vectors.png
   :width: 55%
   :align: center
   :alt: zarr-vectors-tools

----

**zarr-vectors-tools is an extension of** `zarr-vectors-py
<https://zarr-vectors-py.readthedocs.io/en/latest>`_. It is not a
standalone library and does not restate that package's documentation.

``zarr-vectors-py`` owns **Zarr Vectors** — the specification and the
Python API over it
— the store layout, chunk and bin geometry, fragments, links, the object
model, resolution-level metadata, validation, and the two supported
surfaces ``zarr_vectors.api`` and ``zarr_vectors.building``. **Those are
documented there and only there.**

This package adds the layers built on top: **conversion workflows** that
wrap third-party readers and writers (``laspy``, ``plyfile``, ``nibabel``,
``trx-python``, ``networkx``, ``cloud-volume``), **streaming graph and mesh
algorithms** that never materialise a whole store, the **rich
multiresolution layer** — skeleton and polyline coarsening,
spatial-coverage and length-ranked object selection — and the ``zvtools``
CLI.

.. admonition:: Where to look things up
   :class: important

   Anything about the **format** or the **core Python API** belongs to
   ``zarr-vectors-py``: read it at :zvpy:`the specification <spec/index.html>` and
   :zvpy:`the API reference <api/index.html>`. Pages here link out to it rather than
   paraphrasing it, deliberately — a second description of the same format
   is a second description to keep in sync, and the one that drifts is
   always the copy.

This release is built against on-disk format version |zv_version|. What
that version *is* — including the merged ``links/<delta>/<offsets>/``
layout this package assumes — is specified at
:zvpy:`Links <spec/object_model/links.html>`.

The format was originally specified by Forrest Collman at the Allen
Institute for Brain Sciences.

----

Related sites
-------------

.. list-table::
   :widths: 30 70

   * - `zarr-vectors-py docs <https://zarr-vectors-py.readthedocs.io/en/latest>`__
     - **The parent package.** The format specification, the ``api`` and
       ``building`` surfaces, chunk and bin geometry, links, validation.
       Start here for anything this package does not itself own.
   * - :zvpy:`Specification <spec/index.html>`
     - The Zarr Vectors specification as this implementation targets
       it: store structure, metadata documents, spatial indexing, links,
       conformance levels.
   * - :zvpy:`Core API reference <api/index.html>`
     - Which core modules are supported, which are internal, and how to ask
       at runtime with ``zarr_vectors.stability()``.
   * - `Upstream specification <https://alleninstitute.github.io/zarr_vectors/>`__
     - The original Allen Institute format definition this implementation
       derives from.
   * - `GitHub repository <https://github.com/AllenInstitute/zarr-vectors-tools>`__
     - Source, issues, and the notebooks under ``examples/``.

Where to start
--------------

.. list-table::
   :widths: 35 65

   * - :doc:`getting_started/zarr_vectors`
     - New to Zarr Vectors? Start here — what the format is, why chunked
       vector geometry, and how the two packages divide the work.
   * - :doc:`getting_started/quickstart`
     - Convert a file, build a pyramid, run an algorithm, export — from
       the CLI and from Python.
   * - :doc:`getting_started/cli`
     - The ``zvtools`` command line: ``convert``, ``pyramid``,
       ``validate``, ``info``.
   * - :doc:`modules/index`
     - Module-by-module summary of the package: what each subpackage owns
       and where its entry points are.
   * - :doc:`multiresolution/concepts`
     - Coarsening versus sparsity — the two orthogonal axes of a pyramid,
       and the one that most people get wrong first.
   * - :doc:`api/index`
     - Auto-generated reference for every public function.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started/zarr_vectors
   getting_started/installation
   getting_started/quickstart
   getting_started/concepts
   getting_started/cli

.. toctree::
   :maxdepth: 1
   :caption: Modules
   :hidden:

   modules/index

.. toctree::
   :maxdepth: 1
   :caption: Ingest Workflows
   :hidden:

   ingest/index
   ingest/point_clouds
   ingest/single_cell
   ingest/lines
   ingest/tractography
   ingest/tractography_at_scale
   ingest/skeletons
   ingest/em_skeletons
   ingest/graphs
   ingest/meshes
   ingest/surfaces

.. toctree::
   :maxdepth: 1
   :caption: Compose
   :hidden:

   compose/index

.. toctree::
   :maxdepth: 1
   :caption: Multiresolution
   :hidden:

   multiresolution/index
   multiresolution/concepts
   multiresolution/building_pyramids
   multiresolution/strategies
   multiresolution/object_selection
   multiresolution/cross_level_links
   multiresolution/refresh

.. toctree::
   :maxdepth: 1
   :caption: Algorithms
   :hidden:

   algorithms/index
   algorithms/graph_search
   algorithms/graph_components
   algorithms/graph_clustering
   algorithms/mesh_summary
   algorithms/mesh_attributes
   algorithms/mesh_query

.. toctree::
   :maxdepth: 1
   :caption: Export Workflows
   :hidden:

   export/index
   export/point_clouds
   export/single_cell
   export/streamlines
   export/skeletons
   export/meshes
   export/surfaces

.. toctree::
   :maxdepth: 1
   :caption: How-To Guides
   :hidden:

   how_to/parallelism
   how_to/compressors
   how_to/choose_chunk_and_bin
   how_to/large_scale_pipelines

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:

   enrichments
   headers
   examples
   upstream/links-merge-findings

.. toctree::
   :maxdepth: 1
   :caption: Benchmarks
   :hidden:

   benchmarks/index

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   api/index
