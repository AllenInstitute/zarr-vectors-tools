.. zarr-vectors-tools documentation master file

.. image:: zarr-vectors.png
   :width: 55%
   :align: center
   :alt: zarr-vectors-tools

----

**zarr-vectors-tools** is the file-format and algorithm companion to
`zarr-vectors-py <https://github.com/Andrew-Keenlyside/zarr-vectors-py>`_.
The core read/write APIs (chunking, sharding, multiresolution, spatial
binning) live in the :mod:`zarr_vectors` package; this package adds the
format-conversion workflows that wrap third-party readers and writers
(``laspy``, ``plyfile``, ``nibabel``, ``trx-python``, ``networkx``) and
the streaming graph / mesh algorithms that operate on chunked Zarr
Vectors stores.

The library implements the `Zarr Vector Format
<https://github.com/AllenInstitute/zarr_vectors>`_ originally specified
by Forrest Collman at the Allen Institute for Brain Sciences.

----

| `Link to the GitHub repository <https://github.com/Andrew-Keenlyside/zarr-vectors-tools>`__

Where to start
--------------

.. list-table::
   :widths: 35 65

   * - :doc:`getting_started/quickstart`
     - Ingest a CSV point cloud, run a graph algorithm, export to PLY in
       a few lines of Python.
   * - :doc:`getting_started/concepts`
     - How ingest, algorithm, and export workflows compose over a Zarr
       Vectors store.
   * - :doc:`ingest/index`
     - File formats this package ingests, what each maps to in ZVF, and
       which third-party deps each needs.
   * - :doc:`algorithms/index`
     - Graph search, components, clustering; mesh summary, attributes,
       spatial queries — all streaming over chunked stores.
   * - :doc:`api/index`
     - Auto-generated reference for every public function.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started/installation
   getting_started/quickstart
   getting_started/concepts

.. toctree::
   :maxdepth: 1
   :caption: Ingest Workflows
   :hidden:

   ingest/index
   ingest/point_clouds
   ingest/lines
   ingest/polylines_streamlines
   ingest/graphs
   ingest/skeletons
   ingest/meshes

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
   :caption: Reference
   :hidden:

   enrichments
   headers
   examples

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
