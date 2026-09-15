"""AnnData (.h5ad) ingest/export round-trip tests.

Skipped in full when ``anndata`` is not installed, except for the
missing-dependency check, which is the one thing that must hold either way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.exceptions import ExportError, IngestError
from zarr_vectors.types.points import read_points

anndata = pytest.importorskip("anndata", reason="anndata not installed")

import pandas as pd  # noqa: E402  (after the importorskip guard)

from zarr_vectors_tools.convert.export.h5ad import export_h5ad  # noqa: E402
from zarr_vectors_tools.convert.ingest.h5ad import ingest_h5ad  # noqa: E402


def make_adata(
    n: int = 200,
    n_genes: int = 12,
    ndim: int = 3,
    spatial_key: str = "spatial",
    seed: int = 0,
):
    """A small spatial-omics-shaped AnnData with mixed obs dtypes."""
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                rng.choice(["Tcell", "Bcell", "Neuron/glia"], n)
            ),
            "total_counts": rng.uniform(100, 5000, n),
            "n_genes": rng.integers(50, 500, n).astype(np.int32),
            "is_doublet": rng.random(n) > 0.9,
            "sample id": rng.choice(["s1", "s2"], n),  # space -> sanitised name
        },
        index=[f"CELL-{i:05d}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"GENE{i}" for i in range(n_genes)])
    adata = anndata.AnnData(
        X=rng.random((n, n_genes)).astype(np.float32), obs=obs, var=var
    )
    adata.obsm[spatial_key] = rng.uniform(0, 1000, (n, ndim)).astype(np.float32)
    return adata


@pytest.fixture
def h5ad_file(tmp_path: Path):
    """Write the default fixture to disk and hand back (path, adata)."""
    adata = make_adata()
    path = tmp_path / "src.h5ad"
    adata.write_h5ad(path)
    return path, adata


class TestH5ADRoundTrip:

    def test_full_round_trip_is_lossless(self, tmp_path: Path, h5ad_file) -> None:
        """Positions, obs values, dtypes, barcodes and order all survive."""
        src, adata = h5ad_file
        store = tmp_path / "store.zarr"

        summary = ingest_h5ad(src, store, (250.0, 250.0, 250.0), genes=["GENE0", "GENE3"])
        assert summary["vertex_count"] == adata.n_obs
        assert summary["spatial_key"] == "spatial"
        assert summary["n_vars"] == adata.n_vars
        assert summary["obs_columns_stored"] == len(adata.obs.columns)
        assert summary["genes_stored"] == 2
        assert summary["obs_index_stored"] is True

        out = tmp_path / "out.h5ad"
        result = export_h5ad(store, out)
        assert result["vertex_count"] == adata.n_obs
        assert result["order_restored"] is True

        back = anndata.read_h5ad(out)

        # Identity and order.
        assert list(back.obs_names) == list(adata.obs_names)
        np.testing.assert_allclose(back.obsm["spatial"], adata.obsm["spatial"])

        # Values and dtypes, per column.
        assert (back.obs["cell_type"].astype(str) == adata.obs["cell_type"].astype(str)).all()
        assert str(back.obs["cell_type"].dtype) == "category"
        assert (back.obs["sample id"].astype(str) == adata.obs["sample id"]).all()
        np.testing.assert_allclose(back.obs["total_counts"], adata.obs["total_counts"])
        assert back.obs["n_genes"].dtype == np.int32
        assert (back.obs["n_genes"] == adata.obs["n_genes"]).all()
        assert back.obs["is_doublet"].dtype == bool
        assert (back.obs["is_doublet"] == adata.obs["is_doublet"]).all()

        # Expression rebuilt into X under the right var_names.
        assert list(back.var_names) == ["GENE0", "GENE3"]
        np.testing.assert_allclose(
            np.asarray(back.X), adata[:, ["GENE0", "GENE3"]].X, atol=1e-6
        )

    def test_2d_spatial_coordinates(self, tmp_path: Path) -> None:
        """Visium/Xenium-style 2D coordinates stay 2D end to end."""
        adata = make_adata(n=120, ndim=2)
        src = tmp_path / "flat.h5ad"
        adata.write_h5ad(src)

        store = tmp_path / "flat.zarr"
        summary = ingest_h5ad(src, store, (300.0, 300.0))
        assert summary["vertex_count"] == 120

        assert read_points(str(store))["positions"].shape == (120, 2)

        out = tmp_path / "flat_out.h5ad"
        export_h5ad(store, out)
        back = anndata.read_h5ad(out)
        assert back.obsm["spatial"].shape == (120, 2)
        np.testing.assert_allclose(back.obsm["spatial"], adata.obsm["spatial"])

    def test_sparse_x_is_densified_per_gene(self, tmp_path: Path) -> None:
        """A sparse X still yields correct per-gene expression attributes."""
        sparse = pytest.importorskip("scipy.sparse")
        adata = make_adata(n=80, n_genes=20)
        dense = adata.X.copy()
        adata.X = sparse.csr_matrix(dense)
        src = tmp_path / "sparse.h5ad"
        adata.write_h5ad(src)

        store = tmp_path / "sparse.zarr"
        ingest_h5ad(src, store, (400.0, 400.0, 400.0), genes=["GENE5"])

        out = tmp_path / "sparse_out.h5ad"
        export_h5ad(store, out)
        back = anndata.read_h5ad(out)
        np.testing.assert_allclose(
            np.asarray(back.X).ravel(), dense[:, 5], atol=1e-6
        )

    def test_object_ids_group_cells(self, tmp_path: Path, h5ad_file) -> None:
        """An obs column can drive Zarr Vectors object grouping."""
        src, adata = h5ad_file
        store = tmp_path / "grouped.zarr"

        summary = ingest_h5ad(
            src, store, (250.0, 250.0, 250.0),
            object_id_column="cell_type",
            per_object_vertex_count=True,
        )
        assert summary["object_count"] == adata.obs["cell_type"].nunique()

        # The column is stored as an attribute too, so it survives export
        # even though core does not return object_ids on an unfiltered read.
        out = tmp_path / "g.h5ad"
        export_h5ad(store, out)
        back = anndata.read_h5ad(out)
        assert (back.obs["cell_type"].astype(str) == adata.obs["cell_type"].astype(str)).all()

    def test_backed_mode_matches_in_memory(self, tmp_path: Path, h5ad_file) -> None:
        """backed=True leaves X on disk but produces the same store."""
        src, adata = h5ad_file
        store = tmp_path / "backed.zarr"

        summary = ingest_h5ad(
            src, store, (250.0, 250.0, 250.0), genes=["GENE1"], backed=True
        )
        assert summary["vertex_count"] == adata.n_obs

        out = tmp_path / "backed.h5ad"
        export_h5ad(store, out)
        back = anndata.read_h5ad(out)
        np.testing.assert_allclose(
            np.asarray(back.X).ravel(), adata[:, "GENE1"].X.ravel(), atol=1e-6
        )


class TestH5ADSelection:

    def test_obs_column_subset(self, tmp_path: Path, h5ad_file) -> None:
        src, _ = h5ad_file
        store = tmp_path / "subset.zarr"
        summary = ingest_h5ad(
            src, store, (250.0, 250.0, 250.0), obs_columns=["cell_type"]
        )
        assert summary["obs_columns_stored"] == 1

        out = tmp_path / "subset.h5ad"
        export_h5ad(store, out)
        assert list(anndata.read_h5ad(out).obs.columns) == ["cell_type"]

    def test_no_obs_columns(self, tmp_path: Path, h5ad_file) -> None:
        """``obs_columns=[]`` stores geometry only."""
        src, _ = h5ad_file
        store = tmp_path / "bare.zarr"
        summary = ingest_h5ad(src, store, (250.0, 250.0, 250.0), obs_columns=[])
        assert summary["obs_columns_stored"] == 0

        out = tmp_path / "bare.h5ad"
        export_h5ad(store, out)
        back = anndata.read_h5ad(out)
        assert len(back.obs.columns) == 0
        assert back.n_obs == 200

    def test_auto_detects_embedding_when_no_spatial(self, tmp_path: Path) -> None:
        """Falls back to X_umap for dissociated (non-spatial) data."""
        adata = make_adata(n=100, ndim=2, spatial_key="X_umap")
        src = tmp_path / "umap.h5ad"
        adata.write_h5ad(src)

        store = tmp_path / "umap.zarr"
        summary = ingest_h5ad(src, store, (2.0, 2.0))
        assert summary["spatial_key"] == "X_umap"

        out = tmp_path / "umap_out.h5ad"
        assert export_h5ad(store, out)["spatial_key"] == "X_umap"
        assert "X_umap" in anndata.read_h5ad(out).obsm

    def test_spatial_columns_selects_axes(self, tmp_path: Path) -> None:
        """A wide embedding can be narrowed to chosen components."""
        adata = make_adata(n=60, ndim=5)
        src = tmp_path / "wide.h5ad"
        adata.write_h5ad(src)

        store = tmp_path / "wide.zarr"
        ingest_h5ad(src, store, (300.0, 300.0), spatial_columns=[0, 4])
        positions = read_points(str(store))["positions"]
        assert positions.shape == (60, 2)
        np.testing.assert_allclose(
            np.sort(positions[:, 1]), np.sort(adata.obsm["spatial"][:, 4]), rtol=1e-5
        )

    def test_bbox_filter_keeps_attributes(self, tmp_path: Path, h5ad_file) -> None:
        """A spatial subset exports fewer cells but keeps their obs."""
        src, _ = h5ad_file
        store = tmp_path / "bbox.zarr"
        ingest_h5ad(src, store, (250.0, 250.0, 250.0))

        out = tmp_path / "bbox.h5ad"
        result = export_h5ad(store, out, bbox=([0, 0, 0], [500, 500, 500]))
        back = anndata.read_h5ad(out)
        assert 0 < result["vertex_count"] < 200
        assert back.n_obs == result["vertex_count"]
        assert "cell_type" in back.obs.columns
        assert (back.obsm["spatial"] <= 500).all()


class TestH5ADErrors:

    def test_missing_file_raises_ingest_error(self, tmp_path: Path) -> None:
        with pytest.raises(IngestError, match="Input file not found"):
            ingest_h5ad(tmp_path / "nope.h5ad", tmp_path / "o.zarr", (1.0, 1.0, 1.0))

    def test_unknown_spatial_key(self, tmp_path: Path, h5ad_file) -> None:
        src, _ = h5ad_file
        with pytest.raises(IngestError, match="not found"):
            ingest_h5ad(
                src, tmp_path / "o.zarr", (1.0, 1.0, 1.0), spatial_key="missing"
            )

    def test_unknown_obs_column(self, tmp_path: Path, h5ad_file) -> None:
        src, _ = h5ad_file
        with pytest.raises(IngestError, match="obs columns not found"):
            ingest_h5ad(
                src, tmp_path / "o.zarr", (1.0, 1.0, 1.0), obs_columns=["nope"]
            )

    def test_unknown_gene(self, tmp_path: Path, h5ad_file) -> None:
        src, _ = h5ad_file
        with pytest.raises(IngestError, match="genes not found"):
            ingest_h5ad(src, tmp_path / "o.zarr", (1.0, 1.0, 1.0), genes=["NOSUCH"])

    def test_chunk_shape_dimension_mismatch(self, tmp_path: Path) -> None:
        """A 2D embedding with a 3D chunk shape is caught before writing."""
        adata = make_adata(n=30, ndim=2)
        src = tmp_path / "d2.h5ad"
        adata.write_h5ad(src)
        with pytest.raises(IngestError, match="2-dimensional"):
            ingest_h5ad(src, tmp_path / "o.zarr", (10.0, 10.0, 10.0))

    def test_object_filter_with_attributes_is_refused(
        self, tmp_path: Path, h5ad_file
    ) -> None:
        """Core drops attributes on object-filtered reads — say so, don't
        silently write an .h5ad with an empty obs."""
        src, _ = h5ad_file
        store = tmp_path / "obj.zarr"
        ingest_h5ad(
            src, store, (250.0, 250.0, 250.0), object_id_column="cell_type"
        )

        with pytest.raises(ExportError, match="object_ids filtering"):
            export_h5ad(store, tmp_path / "bad.h5ad", object_ids=[0])

        # Positions-only export through the same filter is fine.
        result = export_h5ad(
            store, tmp_path / "ok.h5ad", object_ids=[0], attribute_names=[]
        )
        assert result["vertex_count"] > 0
        assert anndata.read_h5ad(tmp_path / "ok.h5ad").n_obs == result["vertex_count"]

    def test_drop_na_removes_bad_coordinates(self, tmp_path: Path) -> None:
        adata = make_adata(n=50)
        coords = np.asarray(adata.obsm["spatial"])
        coords[:5] = np.nan
        adata.obsm["spatial"] = coords
        src = tmp_path / "na.h5ad"
        adata.write_h5ad(src)

        store = tmp_path / "na.zarr"
        summary = ingest_h5ad(src, store, (300.0, 300.0, 300.0), drop_na=True)
        assert summary["dropped_na"] == 5
        assert summary["vertex_count"] == 45


class TestH5ADHeaderPreservation:

    def test_header_records_names_and_categories(
        self, tmp_path: Path, h5ad_file
    ) -> None:
        from zarr_vectors_tools.headers.formats import H5ADHeader
        from zarr_vectors_tools.headers.registry import HeaderRegistry

        src, adata = h5ad_file
        store = tmp_path / "hdr.zarr"
        ingest_h5ad(src, store, (250.0, 250.0, 250.0), genes=["GENE2"])

        header = HeaderRegistry(str(store)).get("h5ad")
        assert isinstance(header, H5ADHeader)
        assert header.spatial_key == "spatial"
        assert header.spatial_ndim == 3
        assert header.n_obs == adata.n_obs
        assert header.n_vars == adata.n_vars
        assert header.gene_names == ["GENE2"]
        assert set(header.categories) == {"cell_type", "sample id"}
        assert sorted(header.categories["cell_type"]) == ["Bcell", "Neuron/glia", "Tcell"]
        # The space in "sample id" is not a legal Zarr key, so the stored
        # attribute is renamed and the original label kept in the header.
        assert "sample id" in header.obs_names
        assert "sample id" not in header.obs_attrs
        stored_as = header.obs_attrs[header.obs_names.index("sample id")]
        assert header.attr_to_obs()[stored_as] == "sample id"

    def test_obs_index_skipped_above_cap(self, tmp_path: Path, h5ad_file) -> None:
        """Above the cap the barcodes are dropped, but order still restores."""
        src, _ = h5ad_file
        store = tmp_path / "capped.zarr"
        summary = ingest_h5ad(
            src, store, (250.0, 250.0, 250.0), max_obs_index=10
        )
        assert summary["obs_index_stored"] is False

        out = tmp_path / "capped.h5ad"
        assert export_h5ad(store, out)["order_restored"] is True
        assert list(anndata.read_h5ad(out).obs_names) == [str(i) for i in range(200)]

    def test_export_without_header_still_works(self, tmp_path: Path) -> None:
        """A points store written by another ingester exports as plain obs."""
        from zarr_vectors.types.points import write_points

        store = tmp_path / "plain.zarr"
        rng = np.random.default_rng(1)
        write_points(
            str(store),
            rng.uniform(0, 100, (40, 3)).astype(np.float32),
            chunk_shape=(50.0, 50.0, 50.0),
            vertex_attributes={"intensity": rng.random(40).astype(np.float32)},
        )

        out = tmp_path / "plain.h5ad"
        result = export_h5ad(store, out)
        assert result["vertex_count"] == 40
        assert result["order_restored"] is False

        back = anndata.read_h5ad(out)
        assert back.n_obs == 40
        assert back.n_vars == 0
        assert "intensity" in back.obs.columns
        assert "spatial" in back.obsm
