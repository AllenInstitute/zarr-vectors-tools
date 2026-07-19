.. zarr-vectors-tools documentation master file

.. image:: zarr-vectors.png
   :width: 55%
   :align: center
   :alt: zarr-vectors-tools

----

**zarr-vectors-tools** is the file-format, algorithm, and multiresolution
companion to `zarr-vectors-py
<https://github.com/BRIDGE-Neuroscience/zarr-vectors-py>`_. The core
read/write APIs — chunk encoding, the spatial index, links, lazy access —
live in the :mod:`zarr_vectors` package. This package adds the three layers
that sit on top of it: **conversion workflows** that wrap third-party
readers and writers (``laspy``, ``plyfile``, ``nibabel``, ``trx-python``,
``networkx``, ``cloud-volume``), **streaming graph and mesh algorithms**
that never materialise a whole store, and the **rich multiresolution
layer** — skeleton and polyline coarsening, spatial-coverage and
length-ranked object selection, cross-level links.

It targets the merged ``links/<delta>/<offsets>/`` layout, on-disk format
version |zvf_version|. Connectivity is a single family at this version;
there is no ``cross_chunk_links/`` group to fall back to.

The library implements the `Zarr Vector Format
<https://github.com/AllenInstitute/zarr_vectors>`_ originally specified
by Forrest Collman at the Allen Institute for Brain Sciences.

----

Related sites
-------------

.. list-table::
   :widths: 30 70

   * - `Main library docs <https://zarr-vectors-py.readthedocs.io/en/latest>`__
     - ``zarr-vectors-py`` — the core format, readers, writers, lazy access,
       and Neuroglancer integration.
   * - `Specification <https://alleninstitute.github.io/zarr_vectors/>`__
     - The normative Zarr Vector Format spec: store structure, spatial
       indexing, links, conformance levels.
   * - `Schema <https://alleninstitute.github.io/zarr_vectors/08-metadata.html>`__
     - The LinkML source and generated JSON Schema for ZVF metadata.
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
   ingest/lines
   ingest/tractography
   ingest/tractography_at_scale
   ingest/skeletons
   ingest/em_skeletons
   ingest/graphs
   ingest/meshes

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
   export/streamlines
   export/skeletons
   export/meshes

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
