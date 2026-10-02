"""Regression tests for ``zvtools merge`` / ``split`` / ``attach`` fixes.

One class per fixed behaviour, each written against what a user sees:
the command's exit status, the store it leaves, the message it prints.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
import zarr_vectors as zv
from zarr_vectors.building import read_level_metadata, read_object_attributes
from zarr_vectors.exceptions import IngestError, StoreError

from tests.test_compose import line, make_store, write_trk
from zarr_vectors_tools.cli import main
from zarr_vectors_tools.compose import (
    Geometry,
    GeometrySource,
    StoreSource,
    merge_stores,
    plan_merge,
    split_parts,
    split_store,
)
from zarr_vectors_tools.convert.ingest._polyline_enrichments import (
    compute_endpoints,
    compute_lengths,
)
from zarr_vectors_tools.multiresolution.coarsen import read_coarsening_record


def objects(path: Path, level: int = 0) -> int:
    return len(zv.open(str(path), mode="r").level(level).objects.ids(present=True))


def polylines_by_id(path: Path) -> dict[int, np.ndarray]:
    result = zv.open(str(path), mode="r").level(0).read()
    return {int(o): result.polylines[i] for i, o in enumerate(result.part_objects)}


def negative_store(path: Path) -> Path:
    """A streamline store whose data runs from -40 to 40 on every axis."""
    rng = np.random.default_rng(0)
    parts = [
        np.cumsum(rng.normal(0, 3, (12, 3)), axis=0).astype(np.float32)
        + rng.uniform(-25, 25, 3).astype(np.float32)
        for _ in range(8)
    ]
    parts.append(line([-40, -40, -40], [-39, -39, -39]))
    parts.append(line([39, 39, 39], [40, 40, 40]))
    merge_stores(
        str(path),
        [GeometrySource(Geometry(kind="streamline", parts=parts), label="neg")],
        create=True, cell_size=(10.0, 10.0, 10.0), pyramid="drop",
    )
    return path


def trk_lines(n: int, seed: int, lo: float = 5.0, hi: float = 80.0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        start = rng.uniform(lo, hi, 3)
        steps = rng.normal(0, 1.5, (int(rng.integers(6, 20)), 3))
        out.append((start + np.cumsum(steps, axis=0)).clip(1, 99).astype(np.float32))
    return out


# =====================================================================
# 1. Geometry below coordinate 0
# =====================================================================


class TestNegativeCoordinates:

    def test_store_merges_into_a_copy_of_itself(self, tmp_path: Path) -> None:
        source = negative_store(tmp_path / "a.zv")
        target = tmp_path / "b.zv"
        shutil.copytree(source, target)

        summary = merge_stores(str(target), [str(source)], pyramid="drop")

        assert summary["objects_added"] == 10
        before, after = polylines_by_id(source), polylines_by_id(target)
        assert len(after) == 20
        for oid, part in before.items():
            np.testing.assert_allclose(after[oid + 10], part)

    def test_points_table_merges_from_the_cli(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(1)
        rows = ["x,y,z,cell,sample"]
        for i in range(60):
            x, y, z = rng.uniform(-50, 50, 3)
            rows.append(f"{x:.3f},{y:.3f},{z:.3f},c{i},{i % 5}")
        table = tmp_path / "neg.csv"
        table.write_text("\n".join(rows) + "\n")
        store = tmp_path / "neg.zv"
        assert main([
            "convert", str(table), str(store), "--format", "table",
            "--position-columns", "x,y,z", "--key-column", "cell",
            "--object-id-column", "sample", "--chunk-shape", "25,25,25",
        ]) == 0
        copy = tmp_path / "copy.zv"
        shutil.copytree(store, copy)

        assert main(["merge", str(copy), str(store), "--pyramid", "drop"]) == 0
        assert objects(copy) == 10

    def test_plan_reports_the_fit(self, tmp_path: Path) -> None:
        source = negative_store(tmp_path / "a.zv")
        target = tmp_path / "b.zv"
        shutil.copytree(source, target)
        plan = plan_merge(str(target), [str(source)])
        assert plan["sources"][0]["fits_target_bounds"] is True

    def test_expand_grows_upward_from_a_negative_first_cell(self, tmp_path: Path) -> None:
        target = negative_store(tmp_path / "a.zv")
        before = zv.open(str(target), mode="r").level(0).store.chunk_grid_bounds("vertices")
        # Diagonal across new cells, so link segments the target never had
        # are created after the grid has grown.
        far = Geometry(kind="streamline", parts=[line([30, 30, 30], [55, 55, 55], [75, 75, 75])])

        summary = merge_stores(
            str(target), [GeometrySource(far, label="far")],
            on_out_of_bounds="expand", pyramid="drop",
        )

        assert summary["objects_added"] == 1
        origin, shape = zv.open(str(target), mode="r").level(0).store.chunk_grid_bounds("vertices")
        assert origin == before[0]                     # first cell unmoved
        assert shape[0] > before[1][0]
        np.testing.assert_allclose(
            polylines_by_id(target)[10], [[30, 30, 30], [55, 55, 55], [75, 75, 75]],
        )
        lo, hi = zv.open(str(target), mode="r").bounds
        assert hi[0] >= 75 and lo[0] <= -40

    def test_expand_refuses_below_the_first_cell_before_writing(self, tmp_path: Path) -> None:
        target = negative_store(tmp_path / "a.zv")
        below = Geometry(kind="streamline", parts=[line([-90, 0, 0], [-80, 0, 0])])
        with pytest.raises(StoreError) as caught:
            merge_stores(
                str(target), [GeometrySource(below, label="below")],
                on_out_of_bounds="expand", pyramid="drop",
            )
        message = str(caught.value)
        assert "downward" in message and "renumber" in message
        assert "--on-out-of-bounds skip" in message and "--transform" in message
        assert objects(target) == 10

    def test_skip_drops_what_lies_below(self, tmp_path: Path) -> None:
        target = negative_store(tmp_path / "a.zv")
        mixed = Geometry(kind="streamline", parts=[
            line([-90, 0, 0], [-80, 0, 0]), line([-30, -30, -30], [-20, -20, -20]),
        ])
        summary = merge_stores(
            str(target), [GeometrySource(mixed, label="mixed")],
            on_out_of_bounds="skip", pyramid="drop",
        )
        assert summary["objects_added"] == 1
        assert summary["skipped_out_of_bounds"] == 1


# =====================================================================
# 2. split --bounds fit
# =====================================================================


class TestSplitFit:

    def test_part_smaller_than_the_parent_is_sized_to_itself(self, tmp_path: Path) -> None:
        parent = make_store(
            tmp_path / "t.zv",
            [line([1, 1, 1], [2, 2, 2]), line([3, 3, 3], [4, 4, 4]),
             line([16, 16, 16], [19, 19, 19])],
            bounds=((0.0, 0.0, 0.0), (20.0, 20.0, 20.0)),
        )
        del parent
        summary = split_store(
            tmp_path / "t.zv", tmp_path / "parts", by="objects",
            parts={"low": [0, 1], "high": [2]}, bounds="fit",
        )
        by_name = {e["name"]: e for e in summary["written"]}
        assert by_name["low"]["objects"] == 2 and by_name["high"]["objects"] == 1

        low = zv.open(by_name["low"]["path"], mode="r")
        np.testing.assert_allclose(low.bounds[0], [1, 1, 1])
        np.testing.assert_allclose(low.bounds[1], [4, 4, 4])
        assert low.level(0).store.chunk_grid_bounds("vertices")[1] == (1, 1, 1)

        high = zv.open(by_name["high"]["path"], mode="r")
        origin, shape = high.level(0).store.chunk_grid_bounds("vertices")
        # Still the parent's cells: 16..19 is cell 3 of 5-unit cells.
        assert tuple(origin) == (3, 3, 3) and shape == (1, 1, 1)
        np.testing.assert_allclose(
            polylines_by_id(Path(by_name["high"]["path"]))[0], [[16, 16, 16], [19, 19, 19]],
        )

    def test_a_subset_source_does_not_claim_the_parents_extent(self, tmp_path: Path) -> None:
        make_store(
            tmp_path / "t.zv",
            [line([1, 1, 1], [2, 2, 2]), line([16, 16, 16], [19, 19, 19])],
            bounds=((0.0, 0.0, 0.0), (20.0, 20.0, 20.0)),
        )
        small = make_store(tmp_path / "small.zv", [line([1, 1, 1], [2, 2, 2])])
        del small
        subset = StoreSource(str(tmp_path / "t.zv"), objects=[0])
        assert subset.info.bounds_exact is False
        # The parent's box overhangs the small target; the object does not.
        summary = merge_stores(str(tmp_path / "small.zv"), [subset], pyramid="drop")
        assert summary["objects_added"] == 1


# =====================================================================
# 3. split --pyramid
# =====================================================================


def _pyramid_parent(tmp_path: Path) -> Path:
    trk = write_trk(tmp_path / "t.trk", trk_lines(80, seed=3))
    store = tmp_path / "parent.zv"
    assert main([
        "convert", str(trk), str(store), "--num-chunks", "4", "--compute-length",
        "--coarsen", "2,2", "--sparsity", "2,4", "--chunk-scale", "2,1",
        "--rdp-tolerance", "1,3",
    ]) == 0
    return store


def _level_params(store: Path) -> dict[int, tuple]:
    dataset = zv.open(str(store), mode="r")
    out = {}
    for index in dataset.levels:
        if index == 0:
            continue
        meta = read_level_metadata(dataset.store, index)
        out[int(index)] = (
            tuple(meta.bin_shape), meta.chunk_shape and tuple(meta.chunk_shape),
            meta.object_sparsity, meta.coarsening_method, meta.parent_level,
            read_coarsening_record(dataset.store, index),
        )
    return out


class TestSplitPyramid:

    def test_rebuild_gives_each_part_the_parents_levels(self, tmp_path: Path) -> None:
        parent = _pyramid_parent(tmp_path)
        expected = _level_params(parent)
        assert set(expected) == {1, 2}

        summary = split_store(
            parent, tmp_path / "ids", by="objects",
            parts={"a": list(range(0, 48)), "b": list(range(48, 80))},
            pyramid="rebuild",
        )
        for entry in summary["written"]:
            path = Path(entry["path"])
            assert _level_params(path) == expected
            n0, n1, n2 = (objects(path, lv) for lv in (0, 1, 2))
            # Sparsity is per level: half of level 0, then a quarter of that.
            assert n1 == pytest.approx(n0 / 2, abs=1)
            assert n2 == pytest.approx(n1 / 4, abs=1)
            assert entry["levels"] == 3

    def test_cli_rebuild_and_explicit_factors(self, tmp_path: Path) -> None:
        parent = _pyramid_parent(tmp_path)
        from zarr_vectors.building import create_groupings_array, write_groupings

        dataset = zv.open(str(parent), mode="r+")
        create_groupings_array(dataset.level(0).store)
        write_groupings(dataset.level(0).store, {0: list(range(40)), 1: list(range(40, 80))})
        dataset.level(0).groups.name_rows(["left", "right"])

        assert main(["split", str(parent), str(tmp_path / "r"), "--pyramid", "rebuild"]) == 0
        for name in ("left", "right"):
            assert zv.open(str(tmp_path / "r" / f"parent_{name}.zv"), mode="r").levels == (0, 1, 2)

        # Factors alone ask for a pyramid: no --pyramid needed.
        assert main([
            "split", str(parent), str(tmp_path / "f"),
            "--pyramid-coarsen", "2", "--pyramid-sparsity", "2",
        ]) == 0
        assert zv.open(str(tmp_path / "f" / "parent_left.zv"), mode="r").levels == (0, 1)

        with pytest.raises(SystemExit, match="nothing with --pyramid drop"):
            main([
                "split", str(parent), str(tmp_path / "x"), "--pyramid", "drop",
                "--pyramid-coarsen", "2",
            ])
        with pytest.raises(SystemExit):
            main(["split", str(parent), str(tmp_path / "k"), "--pyramid", "keep"])

    def test_keep_is_refused_with_a_reason(self, tmp_path: Path) -> None:
        make_store(tmp_path / "t.zv", [line([1, 1, 1], [2, 2, 2])])
        with pytest.raises(ValueError, match="no coarse levels to keep"):
            split_store(tmp_path / "t.zv", tmp_path / "p", by="objects",
                        parts={"a": [0]}, pyramid="keep")

    def test_rebuild_from_a_coarse_level_needs_factors(self, tmp_path: Path) -> None:
        parent = _pyramid_parent(tmp_path)
        with pytest.raises(IngestError, match="--pyramid-coarsen"):
            split_store(parent, tmp_path / "p", by="objects",
                        parts={"a": [0, 1]}, level=1, pyramid="rebuild")
        assert not list((tmp_path / "p").glob("*.zv"))

    def test_parts_are_stored_like_the_parent(self, tmp_path: Path) -> None:
        """Uncompressed (the viewer reads no other) with the parent's bins."""
        parent = _pyramid_parent(tmp_path)
        summary = split_store(parent, tmp_path / "p", by="objects", parts={"a": [0, 1]})
        part = zv.open(summary["written"][0]["path"], mode="r")
        assert part.level(0).store.zarr_group["vertices"].compressors == ()
        from zarr_vectors.building import read_root_metadata

        assert (
            tuple(read_root_metadata(part.store).effective_bin_shape)
            == tuple(read_root_metadata(zv.open(str(parent)).store).effective_bin_shape)
        )


# =====================================================================
# 4. Integer vertex attributes the source lacks
# =====================================================================


class TestVertexAttributeFill:

    @pytest.mark.parametrize(
        "dtype, sentinel",
        [(np.int32, np.iinfo(np.int32).min), (np.uint16, np.iinfo(np.uint16).max)],
    )
    def test_filled_with_a_value_the_dtype_holds(self, tmp_path: Path, dtype, sentinel) -> None:
        first = Geometry(
            kind="streamline",
            parts=[line([1, 1, 1], [2, 2, 2], [3, 3, 3])],
            vertex_attributes={"label": np.array([4, 5, 6], dtype=dtype)},
        )
        merge_stores(
            str(tmp_path / "t.zv"), [GeometrySource(first, label="first")],
            create=True, cell_size=(5.0, 5.0, 5.0),
            bounds=((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)), pyramid="drop",
        )
        second = Geometry(kind="streamline", parts=[line([6, 6, 6], [7, 7, 7])])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            merge_stores(
                str(tmp_path / "t.zv"), [GeometrySource(second, label="second")],
                pyramid="drop",
            )

        batch = next(StoreSource(str(tmp_path / "t.zv")).iter_batches())
        column = batch.vertex_attributes["label"]
        assert column.dtype == dtype
        np.testing.assert_array_equal(column, [4, 5, 6, sentinel, sentinel])


# =====================================================================
# 5. Derived per-object columns for incoming streamlines
# =====================================================================


class TestDerivedColumns:

    def test_length_and_endpoints_are_computed_and_kept_in_the_pyramid(
        self, tmp_path: Path,
    ) -> None:
        target = tmp_path / "t.zv"
        assert main([
            "convert", str(write_trk(tmp_path / "a.trk", trk_lines(100, seed=1))),
            str(target), "--num-chunks", "4", "--compute-length",
            "--compute-endpoints", "--coarsen", "2", "--sparsity", "2",
        ]) == 0
        incoming = trk_lines(150, seed=2)
        trk = write_trk(tmp_path / "b.trk", incoming)

        assert main([
            "merge", str(target), str(trk), "--sparsity-strategy", "length",
            "--on-out-of-bounds", "expand",
        ]) == 0

        level = zv.open(str(target), mode="r").level(0)
        length = np.asarray(read_object_attributes(level.store, "length"))
        start = np.asarray(read_object_attributes(level.store, "start"))
        end = np.asarray(read_object_attributes(level.store, "end"))
        assert not np.isnan(length).any()
        parts = [polylines_by_id(target)[oid] for oid in range(100, 250)]
        np.testing.assert_allclose(length[100:], compute_lengths(parts), rtol=1e-5)
        first, last = compute_endpoints(parts)
        np.testing.assert_allclose(start[100:], first)
        np.testing.assert_allclose(end[100:], last)
        # A NaN length reads as a dead object: 100 of 250 survived before.
        assert objects(target, 1) == 125

    def test_line_length_is_computed(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.convert.ingest.lines import ingest_lines_csv

        (tmp_path / "a.csv").write_text("x0,y0,z0,x1,y1,z1\n0,0,0,3,4,0\n")
        (tmp_path / "b.csv").write_text("x0,y0,z0,x1,y1,z1\n5,5,5,5,5,7\n")
        ingest_lines_csv(tmp_path / "a.csv", tmp_path / "a.zv", (10.0, 10.0, 10.0),
                         compute_length=True)
        ingest_lines_csv(tmp_path / "b.csv", tmp_path / "b.zv", (10.0, 10.0, 10.0))

        merge_stores(str(tmp_path / "a.zv"), [str(tmp_path / "b.zv")], pyramid="drop")

        length = read_object_attributes(
            zv.open(str(tmp_path / "a.zv"), mode="r").level(0).store, "length",
        )
        np.testing.assert_allclose(np.asarray(length), [5.0, 2.0])

    def test_a_column_that_cannot_be_derived_is_still_filled(self, tmp_path: Path) -> None:
        make_store(
            tmp_path / "t.zv", [line([1, 1, 1], [2, 2, 2])],
            object_attributes={"rgb": np.array([[1.0, 0.5, 0.0]], dtype=np.float32)},
        )
        merge_stores(
            str(tmp_path / "t.zv"),
            [GeometrySource(Geometry(kind="streamline", parts=[line([6, 6, 6], [7, 7, 7])]),
                            label="b")],
            pyramid="drop",
        )
        rgb = np.asarray(read_object_attributes(
            zv.open(str(tmp_path / "t.zv"), mode="r").level(0).store, "rgb",
        ))
        assert rgb.shape == (2, 3) and np.isnan(rgb[1]).all()


# =====================================================================
# 6. Error messages name the CLI flags
# =====================================================================


class TestMessages:

    def test_out_of_bounds_refusal_names_the_flags(self, tmp_path: Path) -> None:
        make_store(tmp_path / "t.zv", [line([1, 1, 1], [2, 2, 2])])
        far = Geometry(kind="streamline", parts=[line([50, 50, 50], [51, 51, 51])])
        with pytest.raises(StoreError) as caught:
            merge_stores(str(tmp_path / "t.zv"), [GeometrySource(far, label="far")])
        message = str(caught.value)
        assert "--on-out-of-bounds skip" in message
        assert "--transform" in message
        assert "on_out_of_bounds='skip'" in message

    def test_missing_target_names_create(self, tmp_path: Path) -> None:
        far = Geometry(kind="streamline", parts=[line([1, 1, 1], [2, 2, 2])])
        with pytest.raises(StoreError, match="--create"):
            merge_stores(str(tmp_path / "absent.zv"), [GeometrySource(far, label="x")])


# =====================================================================
# 7. Header keys and 8. part names
# =====================================================================


class TestNames:

    def test_header_copies_use_the_source_name(self, tmp_path: Path) -> None:
        target = tmp_path / "tracts.zv"
        assert main([
            "convert", str(write_trk(tmp_path / "a.trk", trk_lines(10, seed=1))),
            str(target), "--num-chunks", "2",
        ]) == 0
        other = tmp_path / "subject_c.zv"
        assert main([
            "convert", str(write_trk(tmp_path / "c.trk", trk_lines(5, seed=4))),
            str(other), "--num-chunks", "2",
        ]) == 0
        trk = write_trk(tmp_path / "subject_b.trk", trk_lines(5, seed=2))

        assert main([
            "merge", str(target), str(trk), str(other), "--pyramid", "drop",
            "--on-out-of-bounds", "expand",
        ]) == 0

        formats = set(zv.open(str(target), mode="r").headers.available_formats)
        assert {"trk@subject_b", "trk@subject_c"} <= formats

    def test_split_by_provenance_names_parts_after_their_sources(self, tmp_path: Path) -> None:
        target = tmp_path / "tracts.zv"
        assert main([
            "convert", str(write_trk(tmp_path / "a.trk", trk_lines(10, seed=1))),
            str(target), "--num-chunks", "2",
        ]) == 0
        trk = write_trk(tmp_path / "subject_b.trk", trk_lines(5, seed=2))
        assert main(["merge", str(target), str(trk), "--pyramid", "drop"]) == 0
        assert main(["merge", str(target), str(trk), "--pyramid", "drop"]) == 0

        parts = split_parts(zv.open(str(target), mode="r"), by="provenance")
        assert list(parts) == ["original", "subject_b", "subject_b_2"]
        assert [len(v) for v in parts.values()] == [10, 5, 5]

        assert main(["split", str(target), str(tmp_path / "parts"), "--by", "provenance"]) == 0
        assert sorted(p.name for p in (tmp_path / "parts").iterdir()) == [
            "tracts_original.zv", "tracts_subject_b.zv", "tracts_subject_b_2.zv",
        ]

    def test_part_names_are_readable_and_unique(self, tmp_path: Path) -> None:
        make_store(
            tmp_path / "atlas.zv",
            [line([1, 1, 1], [2, 2, 2]), line([3, 3, 3], [4, 4, 4])],
        )
        summary = split_store(
            tmp_path / "atlas.zv", tmp_path / "parts", by="objects",
            parts={"Left Arcuate": [0], "Left_Arcuate": [1]},
        )
        assert [Path(e["path"]).name for e in summary["written"]] == [
            "atlas_Left_Arcuate.zv", "atlas_Left_Arcuate_2.zv",
        ]


# =====================================================================
# 9. attach --missing error leaves the store as it was
# =====================================================================


def _cells(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    rows = ["x,y,z,cell_id"]
    for i in range(80):
        x, y, z = rng.uniform(0, 100, 3)
        rows.append(f"{x:.3f},{y:.3f},{z:.3f},c{i}")
    table = tmp_path / "cells.csv"
    table.write_text("\n".join(rows) + "\n")
    store = tmp_path / "cells.zv"
    assert main([
        "convert", str(table), str(store), "--format", "table",
        "--position-columns", "x,y,z", "--key-column", "cell_id",
        "--chunk-shape", "25,25,25",
    ]) == 0
    partial = tmp_path / "meta.csv"
    partial.write_text(
        "cell_id,score,label\n"
        + "".join(f"c{i},{i * 0.5},{i % 3}\n" for i in range(0, 80, 2))
    )
    return store, partial


def _vertex_attributes(store: Path) -> set[str]:
    return set(zv.open(str(store), mode="r").level(0).attribute_names("vertex"))


class TestAttachMissingError:

    def test_table_refusal_writes_nothing_and_a_retry_needs_no_overwrite(
        self, tmp_path: Path, capsys,
    ) -> None:
        store, partial = _cells(tmp_path)
        before = _vertex_attributes(store)
        root_meta = (store / "zarr.json").read_text()

        assert main([
            "attach", str(store), str(partial), "--key-column", "cell_id",
            "--missing", "error",
        ]) == 1
        err = capsys.readouterr().err
        assert "40 of 80 vertices" in err and "--missing fill" in err
        assert _vertex_attributes(store) == before
        assert not (store / "0" / "vertex_attributes" / "score").exists()
        assert (store / "zarr.json").read_text() == root_meta   # header untouched

        assert main(["attach", str(store), str(partial), "--key-column", "cell_id"]) == 0
        assert {"score", "label"} <= _vertex_attributes(store)

    def test_cifti_refusal_writes_nothing(self, tmp_path: Path) -> None:
        pytest.importorskip("nibabel")
        from nibabel import cifti2

        from tests._surface_fixtures import brain_models, sheet, write_cifti, write_gifti_subject
        from zarr_vectors_tools.convert.ingest.cifti import attach_cifti
        from zarr_vectors_tools.convert.ingest.gifti import ingest_gifti
        from zarr_vectors_tools.headers.registry import HeaderRegistry

        store = tmp_path / "surf.zv"
        ingest_gifti(write_gifti_subject(tmp_path / "gii"), store, (20.0, 20.0, 20.0))
        n = len(sheet("left")[0])
        cortex = np.arange(n)[5:]                        # a medial wall of 5
        axis = brain_models({"left": n, "right": n}, {"left": cortex, "right": cortex})
        path = write_cifti(
            tmp_path / "thick.dscalar.nii", cifti2.ScalarAxis(["thick"]), axis,
            np.concatenate([cortex * 1.0, cortex * 2.0])[None, :],
        )
        before = _vertex_attributes(store)
        header = HeaderRegistry(str(store)).get("surface")
        scalars = dict(header.scalars)

        with pytest.raises(IngestError, match="nothing was written"):
            attach_cifti(store, path, missing="error")
        assert _vertex_attributes(store) == before
        assert dict(HeaderRegistry(str(store)).get("surface").scalars) == scalars

        summary = attach_cifti(store, path)                # no overwrite needed
        assert "thick" in summary["attributes"]

    def test_a_failed_write_takes_its_new_arrays_with_it(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        import zarr_vectors_tools.convert.ingest.attach as attach

        store, _ = _cells(tmp_path)
        before = _vertex_attributes(store)
        real = attach._write_all_chunks

        def fail_partway(B, level_group, chunk_keys, *args, **kwargs):
            real(B, level_group, chunk_keys[:2], *args, **kwargs)
            raise OSError("disk full")

        monkeypatch.setattr(attach, "_write_all_chunks", fail_partway)
        with pytest.raises(OSError):
            attach.attach_attributes(
                store, {"score": np.arange(80, dtype=np.float32)},
                keys=[f"c{i}" for i in range(80)],
            )
        assert _vertex_attributes(store) == before
