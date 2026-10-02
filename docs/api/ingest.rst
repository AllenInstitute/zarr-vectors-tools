Ingest
======

``zarr_vectors_tools.convert.ingest``: one module per input format. Each reads
a file and writes a store. ``zvtools convert FILE STORE`` calls these.

Points and tables
-----------------

CSV / XYZ points
~~~~~~~~~~~~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.csv_points
   :members:
   :undoc-members:
   :show-inheritance:

Keyed tables
~~~~~~~~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.cell_table
   :members:
   :undoc-members:
   :show-inheritance:

AnnData (h5ad)
~~~~~~~~~~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.h5ad
   :members:
   :undoc-members:
   :show-inheritance:

Attach by key
~~~~~~~~~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.attach
   :members:
   :undoc-members:
   :show-inheritance:

LAS / LAZ
~~~~~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.las
   :members:
   :undoc-members:
   :show-inheritance:

PLY
~~~

.. automodule:: zarr_vectors_tools.convert.ingest.ply
   :members:
   :undoc-members:
   :show-inheritance:

Streamlines and lines
---------------------

TrackVis TRK
~~~~~~~~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.trk
   :members:
   :undoc-members:
   :show-inheritance:

TRK, parallel
~~~~~~~~~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.trk_parallel
   :members:
   :undoc-members:
   :show-inheritance:

LINC TRK
~~~~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.linc_trk
   :members:
   :undoc-members:
   :show-inheritance:

MRtrix TCK
~~~~~~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.tck
   :members:
   :undoc-members:
   :show-inheritance:

TRX
~~~

.. automodule:: zarr_vectors_tools.convert.ingest.trx
   :members:
   :undoc-members:
   :show-inheritance:

Line segments
~~~~~~~~~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.lines
   :members:
   :undoc-members:
   :show-inheritance:

Skeletons
---------

SWC
~~~

.. automodule:: zarr_vectors_tools.convert.ingest.swc
   :members:
   :undoc-members:
   :show-inheritance:

Precomputed layers
~~~~~~~~~~~~~~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.precomputed
   :members:
   :undoc-members:
   :show-inheritance:

Precomputed, spatial index
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.precomputed_skeletons
   :members:
   :undoc-members:
   :show-inheritance:

Precomputed, plain
~~~~~~~~~~~~~~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.precomputed_plain_skeletons
   :members:
   :undoc-members:
   :show-inheritance:

Synapse tables
~~~~~~~~~~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.synapses
   :members:
   :undoc-members:
   :show-inheritance:

Meshes
------

Wavefront OBJ
~~~~~~~~~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.obj
   :members:
   :undoc-members:
   :show-inheritance:

STL
~~~

.. automodule:: zarr_vectors_tools.convert.ingest.stl
   :members:
   :undoc-members:
   :show-inheritance:

Precomputed meshes
~~~~~~~~~~~~~~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.precomputed_meshes
   :members:
   :undoc-members:
   :show-inheritance:

Cortical surfaces
-----------------

GIFTI
~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.gifti
   :members:
   :undoc-members:
   :show-inheritance:

FreeSurfer
~~~~~~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.freesurfer
   :members:
   :undoc-members:
   :show-inheritance:

CIFTI
~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.cifti
   :members:
   :undoc-members:
   :show-inheritance:

Graphs
------

GraphML
~~~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.graphml
   :members:
   :undoc-members:
   :show-inheritance:

Edge list
~~~~~~~~~

.. automodule:: zarr_vectors_tools.convert.ingest.edgelist
   :members:
   :undoc-members:
   :show-inheritance:
