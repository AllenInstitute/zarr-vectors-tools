"""Staged attach: keyed tables, join semantics, and h5ad expression staging.

Covers the workflow where a dataset is split across files that share cells
but not rows — build the store from the coordinate table, then stage the
rest in — which is how the Allen Brain Cell Atlas MERFISH releases ship.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from zarr_vectors.exceptions import IngestError
from zarr_vectors.types.points import read_points

from zarr_vectors_tools.ingest.attach import (
    DEFAULT_KEY_ATTRIBUTE,
    attach_attributes,
    hash_keys,
)
from zarr_vectors_tools.ingest.cell_table import attach_table, ingest_table


def B_open(store):
    """The level-0 group of ``store``, for poking at raw Zarr nodes."""
    import zarr_vectors.building as B

    return B.get_resolution_level(B.open_store(str(store), "r"), 0)

# Identifiers as wide as the atlas's own: 39 decimal digits, i.e. 128-bit,
# which is exactly why they cannot be carried in a numeric column.
_BASE_LABEL = 182941331246012878296807398333956011710


def labels_for(count: int, start: int = 0) -> np.ndarray:
    return np.array([str(_BASE_LABEL + i) for i in range(start, start + count)])


@pytest.fixture
def coord_table(tmp_path: Path):
    """A coordinate table for 60 cells, plus the labels/positions used."""
    rng = np.random.default_rng(0)
    labels = labels_for(60)
    positions = rng.uniform(0, 100, (60, 3))
    path = tmp_path / "coords.csv"
    pd.DataFrame(
        {
            "cell_label": labels,
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "parcellation_index": rng.integers(0, 20, 60),
        }
    ).to_csv(path, index=False)
    return path, labels, positions


@pytest.fixture
def keyed_store(tmp_path: Path, coord_table):
    """A store ingested from the coordinate table, ready to attach onto."""
    path, labels, positions = coord_table
    store = tmp_path / "keyed.zarr"
    ingest_table(
        path, store, (50.0, 50.0, 50.0),
        position_columns=["x", "y", "z"], key_column="cell_label",
    )
    return store, labels, positions


class TestHashKeys:

    def test_is_deterministic_and_distinct(self) -> None:
        labels = labels_for(1000)
        first, second = hash_keys(labels), hash_keys(labels)
        np.testing.assert_array_equal(first, second)
        assert first.dtype == np.int64
        assert len(np.unique(first)) == 1000

    def test_ignores_fixed_width_padding(self) -> None:
        """The same label must hash alike from S39 and S64 columns.

        numpy byte-string arrays zero-pad to the widest element, so a file
        with one long label would otherwise re-pad every other label and
        break the join against a store built from a narrower column.
        """
        labels = labels_for(50)
        narrow = hash_keys(labels.astype("S39"))
        padded = hash_keys(labels.astype("S64"))
        np.testing.assert_array_equal(narrow, padded)
        np.testing.assert_array_equal(narrow, hash_keys(labels.astype("U")))

    def test_integers_pass_through(self) -> None:
        values = np.array([5, 9, 21], dtype=np.int64)
        np.testing.assert_array_equal(hash_keys(values), values)

    def test_order_independent(self) -> None:
        labels = labels_for(100)
        shuffled = np.random.default_rng(3).permutation(labels)
        lookup = dict(zip(labels, hash_keys(labels)))
        for label, digest in zip(shuffled, hash_keys(shuffled)):
            assert lookup[label] == digest

    def test_empty_and_rejects_floats(self) -> None:
        assert len(hash_keys(np.array([], dtype="U1"))) == 0
        with pytest.raises(IngestError, match="join keys must be"):
            hash_keys(np.array([1.5, 2.5]))


class TestIngestTable:

    def test_stores_join_key_and_columns(self, keyed_store) -> None:
        store, labels, positions = keyed_store
        result = read_points(
            str(store),
            attribute_names=[DEFAULT_KEY_ATTRIBUTE, "parcellation_index"],
        )
        assert result["vertex_count"] == 60
        stored = result["vertex_attributes"][DEFAULT_KEY_ATTRIBUTE]
        assert set(stored.tolist()) == set(hash_keys(labels).tolist())

    def test_drops_na_coordinates_by_default(self, tmp_path: Path) -> None:
        path = tmp_path / "gappy.csv"
        frame = pd.DataFrame(
            {
                "cell_label": labels_for(10),
                "x": [1.0] * 10, "y": [2.0] * 10, "z": [3.0] * 10,
            }
        )
        frame.loc[:2, "x"] = np.nan
        frame.to_csv(path, index=False)

        summary = ingest_table(
            path, tmp_path / "g.zarr", (10.0, 10.0, 10.0),
            position_columns=["x", "y", "z"], key_column="cell_label",
        )
        assert summary["dropped_na"] == 3
        assert summary["vertex_count"] == 7

    def test_rejects_unknown_columns(self, tmp_path: Path, coord_table) -> None:
        path, _, _ = coord_table
        with pytest.raises(IngestError, match="position columns not found"):
            ingest_table(
                path, tmp_path / "bad.zarr", (10.0, 10.0),
                position_columns=["lat", "lon"],
            )

    def test_chunk_shape_must_match_position_count(
        self, tmp_path: Path, coord_table
    ) -> None:
        path, _, _ = coord_table
        with pytest.raises(IngestError, match="chunk_shape has"):
            ingest_table(
                path, tmp_path / "bad.zarr", (10.0, 10.0),
                position_columns=["x", "y", "z"],
            )


class TestAttachTable:

    def test_joins_on_key_regardless_of_order(
        self, tmp_path: Path, keyed_store
    ) -> None:
        """A shuffled table lands each value on the right point."""
        store, labels, _ = keyed_store
        shuffled = np.random.default_rng(5).permutation(labels)
        # value == the label's original position, so misalignment is visible
        expected = {label: i for i, label in enumerate(labels)}

        extra = tmp_path / "extra.csv"
        pd.DataFrame(
            {
                "cell_label": shuffled,
                "score": [float(expected[label]) for label in shuffled],
            }
        ).to_csv(extra, index=False)

        summary = attach_table(store, extra, key_column="cell_label")
        assert summary["vertices_matched"] == 60
        assert summary["vertices_unmatched"] == 0

        result = read_points(
            str(store), attribute_names=[DEFAULT_KEY_ATTRIBUTE, "score"]
        )
        keys = result["vertex_attributes"][DEFAULT_KEY_ATTRIBUTE]
        scores = result["vertex_attributes"]["score"]
        by_hash = {hash_keys(np.array([label]))[0]: i for label, i in expected.items()}
        for key, score in zip(keys, scores):
            assert score == pytest.approx(by_hash[key])

    def test_partial_overlap_fills_missing(
        self, tmp_path: Path, keyed_store
    ) -> None:
        """A table covering only some cells fills the rest, not fails."""
        store, labels, _ = keyed_store
        subset = labels[:25]
        extra = tmp_path / "partial.csv"
        pd.DataFrame(
            {"cell_label": subset, "score": np.arange(25, dtype=float)}
        ).to_csv(extra, index=False)

        summary = attach_table(store, extra, key_column="cell_label")
        assert summary["vertices_matched"] == 25
        assert summary["vertices_unmatched"] == 35

        scores = read_points(str(store), attribute_names=["score"])[
            "vertex_attributes"
        ]["score"]
        assert np.isnan(scores).sum() == 35

    def test_extra_rows_are_ignored(self, tmp_path: Path, keyed_store) -> None:
        """Rows for cells the store does not hold are simply unused."""
        store, labels, _ = keyed_store
        extended = np.concatenate([labels, labels_for(40, start=1000)])
        extra = tmp_path / "super.csv"
        pd.DataFrame(
            {"cell_label": extended, "score": np.arange(len(extended), dtype=float)}
        ).to_csv(extra, index=False)

        summary = attach_table(store, extra, key_column="cell_label")
        assert summary["rows_available"] == 100
        assert summary["vertices_matched"] == 60
        assert summary["vertices_unmatched"] == 0

    def test_missing_error_mode(self, tmp_path: Path, keyed_store) -> None:
        store, labels, _ = keyed_store
        extra = tmp_path / "short.csv"
        pd.DataFrame(
            {"cell_label": labels[:10], "score": np.arange(10, dtype=float)}
        ).to_csv(extra, index=False)

        with pytest.raises(IngestError, match="no matching row"):
            attach_table(store, extra, key_column="cell_label", missing="error")

    def test_categorical_column_round_trips(
        self, tmp_path: Path, keyed_store
    ) -> None:
        anndata = pytest.importorskip("anndata")
        from zarr_vectors_tools.export.h5ad import export_h5ad

        store, labels, _ = keyed_store
        sexes = np.where(np.arange(60) % 2 == 0, "M", "F")
        extra = tmp_path / "sex.csv"
        pd.DataFrame({"cell_label": labels, "donor_sex": sexes}).to_csv(
            extra, index=False
        )
        attach_table(store, extra, key_column="cell_label")

        out = tmp_path / "merged.h5ad"
        export_h5ad(store, out)
        back = anndata.read_h5ad(out)
        assert str(back.obs["donor_sex"].dtype) == "category"
        by_label = dict(zip(labels, sexes))
        for label, value in zip(back.obs_names, back.obs["donor_sex"].astype(str)):
            assert value == by_label[str(label)]
        # Bookkeeping attributes stay internal.
        assert DEFAULT_KEY_ATTRIBUTE not in back.obs.columns

    def test_overwrite_guard(self, tmp_path: Path, keyed_store) -> None:
        store, labels, _ = keyed_store
        extra = tmp_path / "s.csv"
        pd.DataFrame(
            {"cell_label": labels, "score": np.zeros(60)}
        ).to_csv(extra, index=False)
        attach_table(store, extra, key_column="cell_label")

        with pytest.raises(IngestError, match="already exist"):
            attach_table(store, extra, key_column="cell_label")

        pd.DataFrame(
            {"cell_label": labels, "score": np.ones(60)}
        ).to_csv(extra, index=False)
        attach_table(store, extra, key_column="cell_label", overwrite=True)
        scores = read_points(str(store), attribute_names=["score"])[
            "vertex_attributes"
        ]["score"]
        assert (scores == 1).all()


class TestAttachAttributes:

    def test_requires_a_join_key_attribute(self, tmp_path: Path) -> None:
        """A store built without a key column cannot be attached onto."""
        from zarr_vectors.types.points import write_points

        store = tmp_path / "nokey.zarr"
        write_points(
            str(store),
            np.random.default_rng(0).uniform(0, 10, (20, 3)).astype(np.float32),
            chunk_shape=(10.0, 10.0, 10.0),
        )
        with pytest.raises(IngestError, match="no join-key attribute"):
            attach_attributes(
                store, {"a": np.zeros(20)}, keys=labels_for(20)
            )

    def test_positional_mode_uses_row_attribute(self, keyed_store) -> None:
        """With keys=None the store attribute indexes the incoming rows."""
        store, labels, _ = keyed_store
        values = np.arange(60, dtype=np.float64) * 2.0
        summary = attach_attributes(
            store, {"doubled": values}, keys=None, key_attribute="table_row"
        )
        assert summary["vertices_matched"] == 60

        result = read_points(
            str(store), attribute_names=["table_row", "doubled"]
        )
        rows = result["vertex_attributes"]["table_row"]
        doubled = result["vertex_attributes"]["doubled"]
        np.testing.assert_allclose(doubled, rows * 2.0)

    def test_rejects_mismatched_lengths(self, keyed_store) -> None:
        store, labels, _ = keyed_store
        with pytest.raises(IngestError, match="differing lengths"):
            attach_attributes(
                store, {"a": np.zeros(5), "b": np.zeros(6)}, keys=labels[:5]
            )
        with pytest.raises(IngestError, match="keys has"):
            attach_attributes(store, {"a": np.zeros(5)}, keys=labels[:4])

    def test_shard_shape_creates_native_sharded_arrays(self, keyed_store) -> None:
        """Sharding must be chosen at create time, not repacked later.

        Unsharded, a store costs one file per chunk per attribute, so a
        wide panel lands in the millions of files and repacking it means
        rereading every one. Attaching with ``shard_shape`` avoids that.
        """
        import zarr
        from zarr_vectors.sharding.io import _is_native_sharded

        store, labels, _ = keyed_store
        values = np.arange(60, dtype=np.float64)
        summary = attach_attributes(
            store, {"packed": values}, keys=labels, shard_shape=4
        )
        assert summary["sharded"] is True
        assert summary["vertices_matched"] == 60

        root = B_open(store)
        node = root.zarr_group["vertex_attributes/packed"]
        assert isinstance(node, zarr.Array)
        assert _is_native_sharded(node)
        assert node.shards == (4, 4, 4)

        # And the values survive the sharded round-trip.
        result = read_points(
            str(store), attribute_names=[DEFAULT_KEY_ATTRIBUTE, "packed"]
        )
        by_hash = dict(zip(hash_keys(labels), values))
        for key, got in zip(
            result["vertex_attributes"][DEFAULT_KEY_ATTRIBUTE],
            result["vertex_attributes"]["packed"],
        ):
            assert got == pytest.approx(by_hash[key])

    def test_unsharded_by_default(self, keyed_store) -> None:
        store, labels, _ = keyed_store
        summary = attach_attributes(
            store, {"plain": np.zeros(60)}, keys=labels
        )
        assert summary["sharded"] is False

    def test_reports_duplicate_keys(self, tmp_path: Path, keyed_store) -> None:
        store, labels, _ = keyed_store
        doubled = np.concatenate([labels[:10], labels[:10]])
        summary = attach_attributes(
            store, {"score": np.arange(20, dtype=float)}, keys=doubled
        )
        assert summary["duplicate_keys"] == 10


class TestAttachH5AD:

    def test_stages_genes_and_obs(self, tmp_path: Path, keyed_store) -> None:
        anndata = pytest.importorskip("anndata")
        from zarr_vectors_tools.export.h5ad import export_h5ad
        from zarr_vectors_tools.ingest.h5ad import attach_h5ad

        store, labels, _ = keyed_store
        rng = np.random.default_rng(11)
        # Superset of the store's cells, in a different order -- the shape
        # the atlas expression matrices actually have.
        extended = np.concatenate([labels, labels_for(40, start=500)])
        order = rng.permutation(len(extended))
        shuffled = extended[order]
        expression = rng.random((len(extended), 6)).astype(np.float32)[order]

        adata = anndata.AnnData(
            X=expression,
            obs=pd.DataFrame(
                {"region": pd.Categorical(rng.choice(["ctx", "hpf"], len(extended)))},
                index=shuffled,
            ),
            var=pd.DataFrame(
                {"gene_symbol": [f"Sym{i}" for i in range(6)]},
                index=[f"ENSG{i:05d}" for i in range(6)],
            ),
        )
        source = tmp_path / "expr.h5ad"
        adata.write_h5ad(source)

        summary = attach_h5ad(
            store, source, genes=["ENSG00002", "Sym4"], obs_columns=["region"]
        )
        assert summary["genes_attached"] == 2
        assert summary["obs_columns_attached"] == 1
        assert summary["vertices_matched"] == 60
        assert summary["rows_available"] == 100

        out = tmp_path / "merged.h5ad"
        export_h5ad(store, out)
        back = anndata.read_h5ad(out)

        assert sorted(back.var_names) == ["ENSG00002", "Sym4"]
        assert "region" in back.obs.columns
        assert str(back.obs["region"].dtype) == "category"

        truth = {label: expression[i] for i, label in enumerate(shuffled)}
        regions = dict(zip(shuffled, adata.obs["region"].astype(str)))
        matrix = np.asarray(back.X)
        var_names = list(back.var_names)
        col_ensg = var_names.index("ENSG00002")
        col_sym = var_names.index("Sym4")
        exported_regions = back.obs["region"].astype(str)

        for position, label in enumerate(back.obs_names):
            expected = truth[str(label)]
            assert matrix[position, col_ensg] == pytest.approx(expected[2], abs=1e-6)
            assert matrix[position, col_sym] == pytest.approx(expected[4], abs=1e-6)
            assert exported_regions.iloc[position] == regions[str(label)]

    def test_sparse_expression(self, tmp_path: Path, keyed_store) -> None:
        anndata = pytest.importorskip("anndata")
        sparse = pytest.importorskip("scipy.sparse")
        from zarr_vectors_tools.ingest.h5ad import attach_h5ad

        store, labels, _ = keyed_store
        rng = np.random.default_rng(2)
        dense = (rng.random((60, 5)) * (rng.random((60, 5)) > 0.5)).astype(np.float32)
        adata = anndata.AnnData(
            X=sparse.csr_matrix(dense),
            obs=pd.DataFrame(index=labels),
            var=pd.DataFrame(index=[f"G{i}" for i in range(5)]),
        )
        source = tmp_path / "sparse.h5ad"
        adata.write_h5ad(source)

        attach_h5ad(store, source, genes=["G3"])
        result = read_points(
            str(store), attribute_names=[DEFAULT_KEY_ATTRIBUTE, "gene_G3"]
        )
        by_hash = dict(zip(hash_keys(labels), dense[:, 3]))
        for key, value in zip(
            result["vertex_attributes"][DEFAULT_KEY_ATTRIBUTE],
            result["vertex_attributes"]["gene_G3"],
        ):
            assert value == pytest.approx(by_hash[key], abs=1e-6)

    def test_requires_something_to_attach(self, tmp_path: Path, keyed_store) -> None:
        pytest.importorskip("anndata")
        from zarr_vectors_tools.ingest.h5ad import attach_h5ad

        store, _, _ = keyed_store
        source = tmp_path / "missing.h5ad"
        with pytest.raises(IngestError, match="Input file not found"):
            attach_h5ad(store, source, genes=["X"])

    def test_h5ad_ingest_stores_a_join_key(self, tmp_path: Path) -> None:
        """A store built from an .h5ad can itself be attached onto."""
        anndata = pytest.importorskip("anndata")
        from zarr_vectors_tools.ingest.h5ad import ingest_h5ad

        labels = labels_for(30)
        adata = anndata.AnnData(
            X=np.zeros((30, 2), dtype=np.float32),
            obs=pd.DataFrame(index=labels),
            var=pd.DataFrame(index=["a", "b"]),
        )
        adata.obsm["spatial"] = np.random.default_rng(0).uniform(0, 10, (30, 3))
        source = tmp_path / "src.h5ad"
        adata.write_h5ad(source)

        store = tmp_path / "fromh5ad.zarr"
        ingest_h5ad(source, store, (5.0, 5.0, 5.0))

        extra = tmp_path / "extra.csv"
        pd.DataFrame(
            {"cell_label": labels, "score": np.arange(30, dtype=float)}
        ).to_csv(extra, index=False)
        summary = attach_table(store, extra, key_column="cell_label")
        assert summary["vertices_matched"] == 30
