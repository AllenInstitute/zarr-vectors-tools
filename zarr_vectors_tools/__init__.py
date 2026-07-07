"""zarr-vectors-tools: ingest and export workflows for the zarr vectors format.

The core read/write APIs live in the ``zarr_vectors`` package. This package
provides format-conversion workflows that wrap third-party libraries
(nibabel, laspy, plyfile, trimesh, networkx, trx-python), the rich multi-scale
coordination + compute layer (``zarr_vectors_tools.multiresolution``), and the
example notebooks that demonstrate end-to-end pipelines.

Multi-scale ownership: core ships only a *basic*, dependency-free coarsening
layer (the ``per_object`` binning pyramid + ``random`` object selection) plus a
plug-in strategy registry (``zarr_vectors.multiresolution.registry``). The rich
strategies (skeleton / polyline coarsening; spatial-coverage / length /
attribute / point-thinning selection, with their heavier dependencies) live
here.

Importing this package does two things:

- populates the tools-local coarsener registry (via ``...multiresolution.coarsen``)
  so ``zarr_vectors_tools.multiresolution.coarsen.coarsen_level`` / ``build_pyramid``
  dispatch works; and
- registers those rich strategies into core's plug-in registry, so
  ``zarr_vectors.multiresolution.coarsen_level`` can dispatch to them by name
  (e.g. ``method="skeleton"``, ``sparsity_strategy="spatial_coverage"``) without
  core taking a hard dependency on this package.

Note: pyramid refresh is not injected into core. ``EditSession(refresh_pyramid=...)``
uses core's own basic refresher; to re-coarsen with the rich strategies after an
edit, call ``zarr_vectors_tools.multiresolution.refresh.rebuild_pyramid_from_level``
directly once pending edits are committed.
"""


def _register_multiscale_strategies() -> None:
    """Wire the rich multi-scale strategies into both registries (idempotent).

    Importing ``coarsen`` runs its bottom-of-module ``register_coarsener(...)``
    calls (tools-local registry). Then, when core exposes the plug-in registry
    (PR #27), register the same coarsen methods and the object-selection
    strategies into core so ``zarr_vectors.multiresolution.coarsen_level`` /
    ``apply_sparsity`` dispatch to them by name.
    """
    # Tools-local registry (used by zarr_vectors_tools.multiresolution.coarsen).
    from zarr_vectors_tools.multiresolution import coarsen as _coarsen  # noqa: F401

    # Core plug-in registry — absent on pre-PR#27 cores; degrade quietly.
    try:
        from zarr_vectors.multiresolution.registry import (
            register_coarsen_strategy,
            register_selection_strategy,
        )
    except ImportError:
        return

    from zarr_vectors_tools.multiresolution import object_selection as _sel
    from zarr_vectors_tools.multiresolution.coarsen import (
        _polyline_coarsener,
        _skeleton_coarsener,
    )

    # Coarsen methods (core reserves the "per_object" builtin; register the rest).
    # Their signatures already match core's coarsen_level kwargs.
    register_coarsen_strategy("skeleton", _skeleton_coarsener)
    register_coarsen_strategy("polyline", _polyline_coarsener)

    # Object-selection strategies. Core calls
    # ``fn(n_objects, target_count, *, seed, lengths, attribute_values,
    #     attribute_mode, representative_points, bin_shape)``; adapt each to the
    # corresponding tools selector.
    def _spatial_coverage(n_objects, target_count, *,
                          representative_points=None, bin_shape=None, **_):
        return _sel.select_by_spatial_coverage(
            representative_points, bin_shape, target_count,
        )

    def _length(n_objects, target_count, *, lengths=None, **_):
        return _sel.select_by_length(lengths, target_count)

    def _attribute(n_objects, target_count, *,
                   attribute_values=None, attribute_mode="max", **_):
        return _sel.select_by_attribute(
            attribute_values, target_count, mode=attribute_mode,
        )

    def _point_thinning(n_objects, target_count, *,
                        representative_points=None, bin_shape=None, seed=None, **_):
        return _sel.select_point_thinning(
            representative_points, bin_shape, seed=seed,
        )

    register_selection_strategy("spatial_coverage", _spatial_coverage)
    register_selection_strategy("length", _length)
    register_selection_strategy("attribute", _attribute)
    register_selection_strategy("point_thinning", _point_thinning)


_register_multiscale_strategies()
