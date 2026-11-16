Headers
=======

``zarr_vectors_tools.headers``: the source format's metadata that the store's
arrays cannot hold, kept under ``headers/<format>/`` so that an export can
write it back. :class:`~zarr_vectors_tools.headers.registry.HeaderRegistry`
reads and writes it as one dataclass per format.

Registry
--------

.. automodule:: zarr_vectors_tools.headers.registry
   :members:
   :undoc-members:
   :show-inheritance:

Format-specific headers
-----------------------

.. automodule:: zarr_vectors_tools.headers.formats
   :members:
   :undoc-members:
   :show-inheritance:
