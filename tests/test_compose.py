"""Tests for :mod:`zarr_vectors_tools.compose` — readers, merge, split.

The cases worth having here are the ones that fail *quietly* if the code
is wrong, because those are the ones a smoke test would pass:

* ``n_properties`` at byte 238.  Read from 236 (as the parallel ingester
  does) every record's stride is four bytes short and the parse walks off
  the data — but only for a file that has properties, which most do not.
* Per-vertex attribute alignment across an append.  A fragment appended
  without extending that chunk's attribute blob leaves the two lists
  different lengths; nothing raises, and every later fragment's values
  are paired with the wrong vertices.
* Object-attribute backfill.  A dense column left short reads back as the
  array's fill value, which is indistinguishable from real absence.
* ``segment_id``.  Absent, the merge succeeds and the *pyramid rebuild*
  fails, a long way from the cause.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import zarr_vectors as zv
from zarr_vectors.api.schema import Layout, Schema
from zarr_vectors.building import read_object_attributes

from zarr_vectors_tools.compose import (
    Geometry,
    GeometrySource,
    StoreSource,
    derive_groups,
    merge_stores,
    plan_merge,
    plan_split,
    read_trk,
    read_trk_header,
    split_parts,
    split_store,
)
from zarr_vectors_tools.compose.readers import load_lut

# =====================================================================
# Helpers
# =====================================================================


def write_trk(
    path: Path,
    streamlines: list[np.ndarray],
    *,
    scalars: list[np.ndarray] | None = None,
    properties: np.ndarray | None = None,
    scalar_name: bytes = b"label_id",
    property_name: bytes = b"label_id",
) -> Path:
    """Write a minimal but spec-correct TrackVis file.

    Hand-built rather than produced with nibabel so the byte offsets under
    test are asserted against the specification, not against whatever
    another reader happens to agree with us about.
    """
    n_scalars = 0 if scalars is None else int(np.atleast_2d(scalars[0]).shape[-1])
    n_props = 0 if properties is None else int(np.atleast_2d(properties).shape[-1])

    header = bytearray(1000)
    header[0:6] = b"TRACK\x00"
    struct.pack_into("<3h", header, 6, 10, 10, 10)
    struct.pack_into("<3f", header, 12, 1.0, 1.0, 1.0)
    struct.pack_into("<h", header, 36, n_scalars)
    if n_scalars:
        header[38:38 + len(scalar_name)] = scalar_name
    struct.pack_into("<h", header, 238, n_props)          # 238, per the spec
    if n_props:
        header[240:240 + len(property_name)] = property_name
    struct.pack_into("<16f", header, 440, *np.eye(4, dtype=np.float32).ravel())
    header[948:951] = b"RAS"
    struct.pack_into("<i", header, 988, len(streamlines))
    struct.pack_into("<i", header, 992, 2)
    struct.pack_into("<i", header, 996, 1000)

    with open(path, "wb") as handle:
        handle.write(bytes(header))
        for i, line in enumerate(streamlines):
            pts = np.asarray(line, dtype="<f4")
            handle.write(struct.pack("<i", len(pts)))
            if n_scalars:
                block = np.hstack(
                    [pts, np.asarray(scalars[i], dtype="<f4").reshape(len(pts), -1)]
                )
            else:
                block = pts
            handle.write(np.ascontiguousarray(block, dtype="<f4").tobytes())
            if n_props:
                handle.write(
                    np.asarray(properties[i], dtype="<f4").ravel().tobytes()
                )
    return path


def make_store(
    path: Path,
    polylines: list[np.ndarray],
    *,
    bounds=((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
    cell=(5.0, 5.0, 5.0),
    object_attributes=None,
    vertex_attributes=None,
):
    dataset = zv.create(
        str(path),
        schema=Schema(
            ndim=3, bounds=bounds, kind="streamline",
            layout=Layout(cell_size=cell),
        ),
    )
    dataset.add_polylines(
        polylines, streamlines=True,
        attributes=vertex_attributes,
        object_attributes=object_attributes,
    )
    # ``add_polylines`` allocates the chunk arrays from the extent of the
    # data it was given, not from the declared bounds, so a store whose
    # polylines stop at 4 gets a one-cell grid even though it claims to
    # span 0..10.  Grow it to what it declares, so these tests exercise
    # merging into a normal store rather than the allocation shortfall --
    # which has tests of its own in TestBounds.
    from zarr_vectors_tools.compose._carry import expand_grid

    expand_grid(
        dataset, dataset.level(0),
        tuple(int(np.floor(h / c)) + 1 for h, c in zip(bounds[1], cell)),
    )
    return dataset


def line(*points) -> np.ndarray:
    return np.asarray(points, dtype=np.float32)


# =====================================================================
# Readers
# =====================================================================


class TestTrkReader:

    def test_n_properties_is_read_from_byte_238(self, tmp_path: Path) -> None:
        """A file with properties parses, and the count is not zero.

        Reading the count from 236 lands in the last two bytes of the
        ``scalar_name`` table, yields 0, and shortens every record's
        stride by four bytes.  The parse then either raises or returns
        nonsense — so the assertion that matters is that the streamlines
        come back *exactly* as written.
        """
        lines = [line([1, 1, 1], [2, 2, 2]), line([3, 3, 3], [4, 4, 4], [5, 5, 5])]
        props = np.array([[21.0], [53.0]], dtype=np.float32)
        path = write_trk(tmp_path / "p.trk", lines, properties=props)

        header = read_trk_header(path)
        assert header["n_properties"] == 1
        assert header["property_names"] == ["label_id"]

        geometry = read_trk(path)
        assert len(geometry.parts) == 2
        np.testing.assert_allclose(geometry.parts[0], lines[0])
        np.testing.assert_allclose(geometry.parts[1], lines[1])
        np.testing.assert_allclose(
            geometry.object_attributes["label_id"], [21.0, 53.0]
        )

    def test_scalars_and_properties_together(self, tmp_path: Path) -> None:
        lines = [line([1, 1, 1], [2, 2, 2]), line([3, 3, 3], [4, 4, 4], [5, 5, 5])]
        scalars = [np.array([[7.0], [7.0]]), np.array([[9.0], [9.0], [9.0]])]
        props = np.array([[7.0], [9.0]], dtype=np.float32)
        path = write_trk(tmp_path / "sp.trk", lines, scalars=scalars, properties=props)

        geometry = read_trk(path)
        np.testing.assert_allclose(
            geometry.vertex_attributes["label_id"], [7, 7, 9, 9, 9]
        )
        np.testing.assert_allclose(geometry.object_attributes["label_id"], [7, 9])

    def test_positions_only(self, tmp_path: Path) -> None:
        lines = [line([1, 1, 1], [2, 2, 2])]
        geometry = read_trk(write_trk(tmp_path / "b.trk", lines))
        assert geometry.object_attributes == {}
        assert geometry.vertex_attributes == {}

    def test_truncated_file_is_reported(self, tmp_path: Path) -> None:
        path = write_trk(tmp_path / "t.trk", [line([1, 1, 1], [2, 2, 2])])
        data = path.read_bytes()
        path.write_bytes(data[:-8])
        with pytest.raises(Exception, match="truncated"):
            read_trk(path)


class TestDeriveGroups:

    def test_label_column_becomes_named_groups(self, tmp_path: Path) -> None:
        geometry = Geometry(
            kind="streamline",
            parts=[line([1, 1, 1], [2, 2, 2]) for _ in range(4)],
            object_attributes={"label_id": np.array([21.0, 53.0, 21.0, 99.0])},
        )
        lut = {21: "ac", 53: "putvmpfc"}
        derive_groups(geometry, "label_id", names=lut)

        assert set(geometry.groups) == {"ac", "putvmpfc", "label_id_99"}
        np.testing.assert_array_equal(geometry.groups["ac"], [0, 2])
        np.testing.assert_array_equal(geometry.groups["putvmpfc"], [1])

    def test_lut_keys_join_across_types(self, tmp_path: Path) -> None:
        """A LUT read from JSON has int keys; the column is float32.

        Without canonicalising, ``53.0`` misses ``53`` and every bundle
        comes back named ``label_id_53`` — a join that fails silently and
        looks like a missing lookup table.
        """
        path = tmp_path / "lut.json"
        path.write_text(json.dumps({"ac.trk": 21, "putvmpfc.trk": 53}))
        names = load_lut(path)
        assert names == {21: "ac", 53: "putvmpfc"}

        geometry = Geometry(
            kind="streamline",
            parts=[line([1, 1, 1], [2, 2, 2]) for _ in range(2)],
            object_attributes={"label_id": np.array([21.0, 53.0], dtype=np.float32)},
        )
        derive_groups(geometry, "label_id", names=names)
        assert set(geometry.groups) == {"ac", "putvmpfc"}


# =====================================================================
# Merge
# =====================================================================


class TestMerge:

    def test_appends_rather_than_replaces(self, tmp_path: Path) -> None:
        """The property ``add_polylines`` does not have.

        A second ``add_polylines`` leaves the first call's objects gone;
        this is the regression guard that the merge path does not.
        """
        first = [line([1, 1, 1], [2, 2, 2]), line([3, 3, 3], [4, 4, 4])]
        second = [line([6, 6, 6], [7, 7, 7])]
        make_store(tmp_path / "t.zarrvectors", first)

        summary = merge_stores(
            str(tmp_path / "t.zarrvectors"),
            [GeometrySource(Geometry(kind="streamline", parts=second), label="b")],
            pyramid="drop",
        )
        assert summary["objects_added"] == 1

        level = zv.open(str(tmp_path / "t.zarrvectors"), mode="r").level(0)
        assert len(level.objects) == 3
        result = level.read()
        by_id = {int(o): result.polylines[i] for i, o in enumerate(result.part_objects)}
        np.testing.assert_allclose(by_id[0], first[0])
        np.testing.assert_allclose(by_id[1], first[1])
        np.testing.assert_allclose(by_id[2], second[0])

    def test_object_attributes_are_backfilled_both_ways(self, tmp_path: Path) -> None:
        """A column on one side only still ends up as long as the index."""
        make_store(
            tmp_path / "t.zarrvectors",
            [line([1, 1, 1], [2, 2, 2]), line([3, 3, 3], [4, 4, 4])],
            object_attributes={"tag": np.array([10.0, 11.0], dtype=np.float32)},
        )
        incoming = Geometry(
            kind="streamline",
            parts=[line([6, 6, 6], [7, 7, 7])],
            object_attributes={"other": np.array([99.0], dtype=np.float32)},
        )
        merge_stores(
            str(tmp_path / "t.zarrvectors"),
            [GeometrySource(incoming, label="b")], pyramid="drop",
        )

        level = zv.open(str(tmp_path / "t.zarrvectors"), mode="r").level(0)
        tag = np.asarray(read_object_attributes(level.store, "tag"))
        other = np.asarray(read_object_attributes(level.store, "other"))
        assert len(tag) == 3 and len(other) == 3
        np.testing.assert_allclose(tag[:2], [10.0, 11.0])
        assert np.isnan(tag[2])          # target-only column, filled
        assert np.isnan(other[:2]).all()  # source-only column, backfilled
        np.testing.assert_allclose(other[2:], [99.0])

    def test_vertex_attributes_stay_aligned_per_object(self, tmp_path: Path) -> None:
        """The silent-corruption case: values paired with the wrong vertices.

        Read back per object through the manifests, which is the same
        walk the readers use, so a fragment-index drift shows up as
        mismatched values rather than as a length error.
        """
        target = [line([1, 1, 1], [2, 2, 2]), line([3, 3, 3], [4, 4, 4], [8, 8, 8])]
        target_fa = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]
        make_store(
            tmp_path / "t.zarrvectors", target,
            vertex_attributes={"fa": target_fa},
        )
        incoming_parts = [line([6, 6, 6], [7, 7, 7]), line([1, 9, 1], [2, 9, 2])]
        incoming = Geometry(
            kind="streamline",
            parts=incoming_parts,
            vertex_attributes={"fa": np.array([60.0, 70.0, 80.0, 90.0])},
        )
        merge_stores(
            str(tmp_path / "t.zarrvectors"),
            [GeometrySource(incoming, label="b")], pyramid="drop",
        )

        source = StoreSource(zv.open(str(tmp_path / "t.zarrvectors"), mode="r"))
        seen: dict[int, np.ndarray] = {}
        for batch in source.iter_batches():
            offset = 0
            for i, oid in enumerate(batch.object_ids):
                n = len(batch.parts[i])
                seen[int(oid)] = batch.vertex_attributes["fa"][offset:offset + n]
                offset += n
        np.testing.assert_allclose(seen[0], [1.0, 2.0])
        np.testing.assert_allclose(seen[1], [3.0, 4.0, 5.0])
        np.testing.assert_allclose(seen[2], [60.0, 70.0])
        np.testing.assert_allclose(seen[3], [80.0, 90.0])

    def test_groups_are_remapped_not_copied(self, tmp_path: Path) -> None:
        make_store(
            tmp_path / "t.zarrvectors",
            [line([1, 1, 1], [2, 2, 2]), line([3, 3, 3], [4, 4, 4])],
        )
        incoming = Geometry(
            kind="streamline",
            parts=[line([6, 6, 6], [7, 7, 7]), line([8, 8, 8], [9, 9, 9])],
            groups={"cst": np.array([0]), "af": np.array([1])},
        )
        merge_stores(
            str(tmp_path / "t.zarrvectors"),
            [GeometrySource(incoming, label="b")], pyramid="drop",
        )

        level = zv.open(str(tmp_path / "t.zarrvectors"), mode="r").level(0)
        assert set(level.groups.names()) == {"cst", "af"}
        # Source ids 0 and 1 must have become 2 and 3, not stayed put.
        np.testing.assert_array_equal(level.groups["cst"].members, [2])
        np.testing.assert_array_equal(level.groups["af"].members, [3])

    def test_same_group_name_is_extended(self, tmp_path: Path) -> None:
        dataset = make_store(
            tmp_path / "t.zarrvectors",
            [line([1, 1, 1], [2, 2, 2]), line([3, 3, 3], [4, 4, 4])],
        )
        from zarr_vectors.building import create_groupings_array, write_groupings

        create_groupings_array(dataset.level(0).store)
        write_groupings(dataset.level(0).store, {0: [0, 1]})
        dataset.level(0).groups.name_rows(["cst"])

        incoming = Geometry(
            kind="streamline",
            parts=[line([6, 6, 6], [7, 7, 7])],
            groups={"cst": np.array([0])},
        )
        merge_stores(
            str(tmp_path / "t.zarrvectors"),
            [GeometrySource(incoming, label="b")], pyramid="drop",
        )
        level = zv.open(str(tmp_path / "t.zarrvectors"), mode="r").level(0)
        np.testing.assert_array_equal(level.groups["cst"].members, [0, 1, 2])

    def test_group_prefix_keeps_them_distinct(self, tmp_path: Path) -> None:
        dataset = make_store(
            tmp_path / "t.zarrvectors", [line([1, 1, 1], [2, 2, 2])],
        )
        from zarr_vectors.building import create_groupings_array, write_groupings

        create_groupings_array(dataset.level(0).store)
        write_groupings(dataset.level(0).store, {0: [0]})
        dataset.level(0).groups.name_rows(["cst"])

        incoming = Geometry(
            kind="streamline",
            parts=[line([6, 6, 6], [7, 7, 7])],
            groups={"cst": np.array([0])},
        )
        merge_stores(
            str(tmp_path / "t.zarrvectors"),
            [GeometrySource(incoming, label="b")],
            group_prefix="atlas_", pyramid="drop",
        )
        level = zv.open(str(tmp_path / "t.zarrvectors"), mode="r").level(0)
        assert set(level.groups.names()) == {"cst", "atlas_cst"}

    def test_vertex_count_is_restamped(self, tmp_path: Path) -> None:
        """Level metadata must not keep saying the level is empty."""
        make_store(tmp_path / "t.zarrvectors", [line([1, 1, 1], [2, 2, 2])])
        merge_stores(
            str(tmp_path / "t.zarrvectors"),
            [GeometrySource(
                Geometry(kind="streamline", parts=[line([6, 6, 6], [7, 7, 7])]),
                label="b",
            )],
            pyramid="drop",
        )
        level = zv.open(str(tmp_path / "t.zarrvectors"), mode="r").level(0)
        assert level.vertex_count == 4

    def test_segment_id_is_written_for_new_fragments(self, tmp_path: Path) -> None:
        """Without it the merge looks fine and the pyramid rebuild fails."""
        from zarr_vectors.building import (
            read_chunk_fragment_attributes,
            read_object_manifests,
        )

        make_store(tmp_path / "t.zarrvectors", [line([1, 1, 1], [2, 2, 2])])
        merge_stores(
            str(tmp_path / "t.zarrvectors"),
            [GeometrySource(
                Geometry(kind="streamline", parts=[line([6, 6, 6], [7, 7, 7])]),
                label="b",
            )],
            pyramid="drop",
        )
        level = zv.open(str(tmp_path / "t.zarrvectors"), mode="r").level(0)
        manifest = read_object_manifests(level.store, ids=[1])[1]
        for cc, fi in manifest:
            column = read_chunk_fragment_attributes(
                level.store, "segment_id", tuple(cc), dtype=np.uint64,
            )
            assert int(np.asarray(column).ravel()[fi]) == 1

    def test_transform_is_applied_before_binning(self, tmp_path: Path) -> None:
        make_store(tmp_path / "t.zarrvectors", [line([1, 1, 1], [2, 2, 2])])
        far = Geometry(kind="streamline", parts=[line([50, 50, 50], [51, 51, 51])])
        affine = np.eye(4)
        affine[:3, 3] = [-45.0, -45.0, -45.0]

        merge_stores(
            str(tmp_path / "t.zarrvectors"),
            [GeometrySource(far, transform=affine, label="shifted")],
            pyramid="drop",
        )
        level = zv.open(str(tmp_path / "t.zarrvectors"), mode="r").level(0)
        result = level.read()
        moved = [p for i, p in enumerate(result.polylines) if result.part_objects[i] == 1]
        np.testing.assert_allclose(moved[0], [[5, 5, 5], [6, 6, 6]])

    def test_transform_shapes(self) -> None:
        """A square matrix is homogeneous, so 3-D needs 4x4 — and a 3x3
        applied to 3-D points fails on the matmul rather than quietly
        projecting them to two dimensions."""
        from zarr_vectors_tools.compose.sources import as_transform

        points = np.array([[1.0, 2.0, 3.0]])
        affine = np.eye(4)
        affine[:3, 3] = [10.0, 20.0, 30.0]
        np.testing.assert_allclose(
            as_transform(affine)(points), [[11.0, 22.0, 33.0]]
        )
        np.testing.assert_allclose(
            as_transform(affine[:3, :])(points), [[11.0, 22.0, 33.0]]
        )
        with pytest.raises(ValueError):
            as_transform(np.eye(3))(points)
        with pytest.raises(ValueError, match="expected"):
            as_transform(np.zeros((3, 7)))


class TestBounds:

    def _target_and_far(self, tmp_path: Path):
        make_store(tmp_path / "t.zarrvectors", [line([1, 1, 1], [2, 2, 2])])
        return GeometrySource(
            Geometry(kind="streamline", parts=[line([50, 50, 50], [51, 51, 51])]),
            label="far",
        )

    def test_raises_by_default(self, tmp_path: Path) -> None:
        source = self._target_and_far(tmp_path)
        with pytest.raises(Exception, match="does not fit"):
            merge_stores(
                str(tmp_path / "t.zarrvectors"), [source], pyramid="keep",
            )

    def test_skip_drops_and_counts(self, tmp_path: Path) -> None:
        source = self._target_and_far(tmp_path)
        summary = merge_stores(
            str(tmp_path / "t.zarrvectors"), [source],
            on_out_of_bounds="skip", pyramid="drop",
        )
        assert summary["objects_added"] == 0
        assert summary["skipped_out_of_bounds"] == 1

    def test_expand_grows_the_grid_upwards(self, tmp_path: Path) -> None:
        source = self._target_and_far(tmp_path)
        summary = merge_stores(
            str(tmp_path / "t.zarrvectors"), [source],
            on_out_of_bounds="expand", pyramid="drop",
        )
        assert summary["objects_added"] == 1
        assert summary["sources"][0]["grid_expansion"]["expanded"] is True

        level = zv.open(str(tmp_path / "t.zarrvectors"), mode="r").level(0)
        result = level.read()
        by_id = {int(o): result.polylines[i] for i, o in enumerate(result.part_objects)}
        np.testing.assert_allclose(by_id[0], [[1, 1, 1], [2, 2, 2]])
        np.testing.assert_allclose(by_id[1], [[50, 50, 50], [51, 51, 51]])

    def test_expand_refuses_negative_cells(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.compose._carry import expand_grid

        dataset = make_store(tmp_path / "t.zarrvectors", [line([1, 1, 1], [2, 2, 2])])
        with pytest.raises(Exception, match="negative chunk coordinates"):
            expand_grid(dataset, dataset.level(0), (-1, 2, 2))


class TestPlan:

    def test_plan_reads_no_geometry(self, tmp_path: Path) -> None:
        make_store(tmp_path / "t.zarrvectors", [line([1, 1, 1], [2, 2, 2])])
        make_store(tmp_path / "s.zarrvectors", [line([6, 6, 6], [7, 7, 7])])

        plan = plan_merge(str(tmp_path / "t.zarrvectors"), [str(tmp_path / "s.zarrvectors")])
        assert plan["target_objects"] == 1
        assert plan["total_objects_after"] == 2
        assert plan["sources"][0]["id_offset"] == 1
        assert plan["sources"][0]["fits_target_bounds"] is True


# =====================================================================
# Split
# =====================================================================


class TestSplit:

    def _grouped_store(self, tmp_path: Path):
        path = tmp_path / "t.zarrvectors"
        dataset = make_store(
            path,
            [line([1, 1, 1], [2, 2, 2]), line([3, 3, 3], [4, 4, 4]),
             line([6, 6, 6], [7, 7, 7]), line([8, 8, 8], [9, 9, 9])],
            object_attributes={"label": np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)},
        )
        from zarr_vectors.building import create_groupings_array, write_groupings

        create_groupings_array(dataset.level(0).store)
        write_groupings(dataset.level(0).store, {0: [0, 1], 1: [2, 3]})
        dataset.level(0).groups.name_rows(["left", "right"])
        return path

    def test_split_by_groups(self, tmp_path: Path) -> None:
        path = self._grouped_store(tmp_path)
        summary = split_store(path, tmp_path / "parts", by="groups", pyramid="drop")
        assert summary["part_count"] == 2

        by_name = {entry["name"]: entry for entry in summary["written"]}
        assert by_name["left"]["objects"] == 2
        assert by_name["right"]["objects"] == 2

        left = zv.open(by_name["left"]["path"], mode="r").level(0)
        result = left.read()
        np.testing.assert_allclose(
            sorted(p.tolist() for p in result.polylines),
            sorted([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]),
        )

    def test_split_by_attribute(self, tmp_path: Path) -> None:
        path = self._grouped_store(tmp_path)
        parts = split_parts(
            zv.open(str(path), mode="r"), by="attribute", attribute="label",
            names={1: "one", 2: "two"},
        )
        assert set(parts) == {"one", "two"}
        np.testing.assert_array_equal(parts["one"], [0, 1])
        np.testing.assert_array_equal(parts["two"], [2, 3])

    def test_split_keeps_the_parent_grid(self, tmp_path: Path) -> None:
        """Aligned cells are what let the pieces be read together."""
        path = self._grouped_store(tmp_path)
        summary = split_store(path, tmp_path / "parts", by="groups", pyramid="drop")
        parent = zv.open(str(path), mode="r")
        child = zv.open(summary["written"][0]["path"], mode="r")
        assert tuple(child.level(0).scale) == tuple(parent.level(0).scale)
        np.testing.assert_allclose(child.bounds[0], parent.bounds[0])

    def test_split_by_provenance_undoes_a_merge(self, tmp_path: Path) -> None:
        make_store(
            tmp_path / "t.zarrvectors",
            [line([1, 1, 1], [2, 2, 2]), line([3, 3, 3], [4, 4, 4])],
        )
        incoming = Geometry(
            kind="streamline",
            parts=[line([6, 6, 6], [7, 7, 7]), line([8, 8, 8], [9, 9, 9])],
        )
        merge_stores(
            str(tmp_path / "t.zarrvectors"),
            [GeometrySource(incoming, label="atlas")], pyramid="drop",
        )
        parts = split_parts(
            zv.open(str(tmp_path / "t.zarrvectors"), mode="r"), by="provenance",
        )
        assert list(parts) == ["original", "atlas"]
        np.testing.assert_array_equal(parts["original"], [0, 1])
        np.testing.assert_array_equal(parts["atlas"], [2, 3])

    def test_plan_split_counts_without_writing(self, tmp_path: Path) -> None:
        path = self._grouped_store(tmp_path)
        plan = plan_split(zv.open(str(path), mode="r"), by="groups")
        assert plan["part_count"] == 2
        assert plan["total_objects"] == 4
        assert not (tmp_path / "parts").exists()


class TestPyramidGroups:
    """A streamline pyramid must keep its group taxonomy at every level.

    The polyline coarsener wrote no ``groups`` array at all — the
    per-object and per-fragment coarseners always did — so a labelled
    atlas came out with object indices at every level and a way to say
    what an object *is* at only level 0.  Names are a second, separate
    carry: memberships live in the ``groups`` array, labels in its
    metadata, and copying one without the other leaves coarse rows
    addressable as ``group_0``.
    """

    def _atlas(self, tmp_path: Path):
        rng = np.random.default_rng(3)
        polylines = [
            rng.uniform(0.5, 9.5, size=(6, 3)).astype(np.float32) for _ in range(24)
        ]
        dataset = make_store(tmp_path / "atlas.zarrvectors", polylines)
        from zarr_vectors.building import create_groupings_array, write_groupings

        create_groupings_array(dataset.level(0).store)
        write_groupings(
            dataset.level(0).store,
            {0: list(range(0, 12)), 1: list(range(12, 24))},
        )
        dataset.level(0).groups.name_rows(["cst", "af"])
        # add_polylines does not write the per-fragment segment_id the
        # polyline coarsener requires; the merge path stamps it, a plain
        # core write does not.
        from tests._source_helpers import stamp_segment_id_from_manifests

        stamp_segment_id_from_manifests(str(tmp_path / "atlas.zarrvectors"))
        return dataset

    def test_groups_and_names_reach_every_level(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

        dataset = self._atlas(tmp_path)
        build_pyramid(
            dataset.url, factors=[(1.0, 2.0), (1.0, 2.0)],
            sparsity_strategy="random", sparsity_seed=0,
        )

        import zarr_vectors as zv

        reopened = zv.open(str(tmp_path / "atlas.zarrvectors"), mode="r")
        assert len(reopened.levels) == 3
        for index in reopened.levels:
            catalog = reopened.level(index).groups
            assert set(catalog.names()) == {"cst", "af"}, (
                f"level {index} lost the group names"
            )
            members = np.concatenate(
                [np.asarray(catalog[n].members) for n in ("cst", "af")]
            )
            present = np.asarray(reopened.level(index).objects.ids(present=True))
            # Every member must actually hold geometry at this level; a
            # group naming a sparsified-away id reports an empty manifest
            # as a member.
            assert np.isin(members, present).all(), (
                f"level {index} has group members with no geometry"
            )


class TestGroupStratifiedSparsity:
    """``sparsity_strategy="group"`` must never empty a named group.

    Global sparsity samples one pool, so a small group is gone within a
    level or two — on the Maffei atlas, sixteen of forty-three bundles by
    level 4.  Stratifying thins each group by the same factor with a
    floor of one, so the proportions coarsen and the taxonomy does not.
    """

    def test_singleton_group_survives_every_level(self) -> None:
        from zarr_vectors_tools.multiresolution.object_selection import (
            select_stratified_by_group,
        )

        sizes = {0: 1, 1: 4, 2: 11, 3: 500, 4: 5000}
        labels = np.concatenate([np.full(n, g) for g, n in sizes.items()])
        alive = np.arange(len(labels))
        for _ in range(4):
            alive = alive[select_stratified_by_group(labels[alive], 0.25, seed=0)]
            present = {g: int((labels[alive] == g).sum()) for g in sizes}
            assert all(v >= 1 for v in present.values()), present
        # The big groups still coarsen — a floor is not "keep everything".
        assert int((labels[alive] == 4).sum()) < 100

    def test_dispatches_through_apply_sparsity(self) -> None:
        from zarr_vectors_tools.multiresolution.object_selection import apply_sparsity

        labels = np.concatenate([np.full(n, g) for g, n in {0: 1, 1: 100}.items()])
        kept = apply_sparsity(
            len(labels), 0.25, "group", seed=0, group_labels=labels,
        )
        assert 0 in kept.tolist()  # the singleton
        assert 20 <= len(kept) <= 30

    def test_missing_labels_is_an_error_not_a_silent_fallback(self) -> None:
        from zarr_vectors_tools.multiresolution.object_selection import apply_sparsity

        with pytest.raises(ValueError, match="group_labels"):
            apply_sparsity(10, 0.5, "group", seed=0)

    def test_pyramid_keeps_every_bundle(self, tmp_path: Path) -> None:
        """End to end, through build_pyramid, against a store with an
        eight-streamline bundle and a one-streamline bundle."""
        import zarr_vectors as zv
        from zarr_vectors.building import create_groupings_array, write_groupings

        from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

        rng = np.random.default_rng(11)
        polylines = [
            rng.uniform(0.5, 9.5, size=(6, 3)).astype(np.float32) for _ in range(41)
        ]
        dataset = make_store(tmp_path / "atlas.zarrvectors", polylines)
        create_groupings_array(dataset.level(0).store)
        write_groupings(
            dataset.level(0).store,
            {0: list(range(0, 32)), 1: list(range(32, 40)), 2: [40]},
        )
        dataset.level(0).groups.name_rows(["big", "small", "singleton"])
        from tests._source_helpers import stamp_segment_id_from_manifests

        stamp_segment_id_from_manifests(str(tmp_path / "atlas.zarrvectors"))

        build_pyramid(
            dataset.url, factors=[(1.0, 2.0), (1.0, 2.0), (1.0, 2.0)],
            sparsity_strategy="group", sparsity_seed=0,
        )
        reopened = zv.open(str(tmp_path / "atlas.zarrvectors"), mode="r")
        for index in reopened.levels:
            catalog = reopened.level(index).groups
            for name in ("big", "small", "singleton"):
                assert len(catalog[name].members) >= 1, (
                    f"level {index} lost group {name!r}"
                )


class TestRoundTrip:

    def test_merge_then_split_recovers_the_inputs(self, tmp_path: Path) -> None:
        first = [line([1, 1, 1], [2, 2, 2]), line([3, 3, 3], [4, 4, 4])]
        second = [line([6, 6, 6], [7, 7, 7])]
        make_store(tmp_path / "t.zarrvectors", first)
        merge_stores(
            str(tmp_path / "t.zarrvectors"),
            [GeometrySource(
                Geometry(kind="streamline", parts=second), label="second",
            )],
            pyramid="drop",
        )
        summary = split_store(
            tmp_path / "t.zarrvectors", tmp_path / "back",
            by="provenance", pyramid="drop",
        )
        recovered = {}
        for entry in summary["written"]:
            result = zv.open(entry["path"], mode="r").level(0).read()
            recovered[entry["name"]] = sorted(p.tolist() for p in result.polylines)

        assert recovered["original"] == sorted(p.tolist() for p in first)
        assert recovered["second"] == sorted(p.tolist() for p in second)


class TestMergeEdgeCases:
    """Cases that committed geometry and then failed, or wrote a sentinel
    that reads back as data."""

    def test_zero_vertex_object_does_not_desynchronise_columns(
        self, tmp_path: Path,
    ) -> None:
        """An empty part writes no geometry, so it must claim no row.

        It was counted by the attribute slice but not by ``added``, so
        ``append_object_attributes`` raised "N values for N-1 new objects"
        — after every geometry batch had already been committed, leaving
        the store with objects and no columns.
        """
        make_store(tmp_path / "t.zarrvectors", [line([1, 1, 1], [2, 2, 2])])
        incoming = Geometry(
            kind="streamline",
            parts=[
                line([6, 6, 6], [7, 7, 7]),
                np.zeros((0, 3), dtype=np.float32),
                line([8, 8, 8], [9, 9, 9]),
            ],
            object_attributes={"tag": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
        )
        summary = merge_stores(
            str(tmp_path / "t.zarrvectors"),
            [GeometrySource(incoming, label="b")], pyramid="drop",
        )
        assert summary["objects_added"] == 2

        level = zv.open(str(tmp_path / "t.zarrvectors"), mode="r").level(0)
        tag = np.asarray(read_object_attributes(level.store, "tag"))
        assert len(tag) == len(level.objects)
        # The empty part's value must not have shifted onto a real object.
        np.testing.assert_allclose(tag[1:], [1.0, 3.0])

    def test_integer_column_is_backfilled_with_a_real_sentinel(
        self, tmp_path: Path,
    ) -> None:
        """NaN is not a value an integer column can hold.

        ``np.full(n, nan, dtype=uint32)`` yields 0 — indistinguishable
        from a genuine zero — and INT64_MIN for int64.  Core's own
        sentinel for an absent row is dtype-min / dtype-max, so that is
        what the backfill must write.
        """
        make_store(
            tmp_path / "t.zarrvectors",
            [line([1, 1, 1], [2, 2, 2]), line([3, 3, 3], [4, 4, 4])],
        )
        incoming = Geometry(
            kind="streamline",
            parts=[line([6, 6, 6], [7, 7, 7])],
            object_attributes={"label": np.array([7], dtype=np.uint32)},
        )
        merge_stores(
            str(tmp_path / "t.zarrvectors"),
            [GeometrySource(incoming, label="b")], pyramid="drop",
        )

        level = zv.open(str(tmp_path / "t.zarrvectors"), mode="r").level(0)
        label = np.asarray(read_object_attributes(level.store, "label"))
        assert label.dtype.kind == "u"
        assert int(label[2]) == 7
        sentinel = np.iinfo(label.dtype).max
        assert int(label[0]) == int(label[1]) == sentinel, (
            "pre-existing objects must be marked absent, not given a 0 that "
            "reads back as a real label"
        )

    def test_expand_refuses_negative_cells_instead_of_dropping_them(
        self, tmp_path: Path,
    ) -> None:
        """``expand`` grows the grid upward; it cannot move the origin.

        Objects at negative coordinates were silently skipped under a flag
        that promised to make room for them.
        """
        make_store(tmp_path / "t.zarrvectors", [line([1, 1, 1], [2, 2, 2])])
        negative = Geometry(
            kind="streamline", parts=[line([-5, 1, 1], [-4, 1, 1])],
        )
        with pytest.raises(Exception, match="negative cell"):
            merge_stores(
                str(tmp_path / "t.zarrvectors"),
                [GeometrySource(negative, label="neg")],
                on_out_of_bounds="expand", pyramid="drop",
            )

    def test_skip_still_drops_what_does_not_fit(self, tmp_path: Path) -> None:
        """The escape hatch keeps working — expand is the strict one."""
        make_store(tmp_path / "t.zarrvectors", [line([1, 1, 1], [2, 2, 2])])
        negative = Geometry(
            kind="streamline", parts=[line([-5, 1, 1], [-4, 1, 1])],
        )
        summary = merge_stores(
            str(tmp_path / "t.zarrvectors"),
            [GeometrySource(negative, label="neg")],
            on_out_of_bounds="skip", pyramid="drop",
        )
        assert summary["objects_added"] == 0

    def test_missing_target_is_distinguished_from_a_broken_one(
        self, tmp_path: Path,
    ) -> None:
        """"Does not exist" sent the caller to create=True for nothing."""
        broken = tmp_path / "broken.zarrvectors"
        broken.mkdir()
        (broken / "zarr.json").write_text("{not json")
        with pytest.raises(Exception, match="could not be opened"):
            merge_stores(
                str(broken),
                [GeometrySource(
                    Geometry(kind="streamline", parts=[line([1, 1, 1], [2, 2, 2])]),
                    label="b",
                )],
                pyramid="drop",
            )


class TestBatchBudget:

    def test_max_vertices_counts_vertices_not_fragments(
        self, tmp_path: Path,
    ) -> None:
        """The budget bounded fragments, so batches ran many times larger.

        Each polyline here is one fragment of 8 vertices; a ceiling of 10
        vertices must therefore put one object in a batch, not ten.
        """
        store = tmp_path / "src.zarrvectors"
        make_store(
            store,
            [
                line(*[[1.0 + i * 0.1, 1.0, 1.0]] * 8)
                for i in range(6)
            ],
        )
        source = StoreSource(str(store))
        batches = list(source.iter_batches(max_objects=1000, max_vertices=10))
        assert len(batches) == 6
        assert all(len(batch.parts) == 1 for batch in batches)


class TestTrkHeaderRoundTrip:

    def test_every_field_the_reader_writes_survives(self) -> None:
        """``space`` is the load-bearing one.

        It says whether the stored coordinates still need the affine
        applied; dropped, a later merge or export cannot tell, and the
        difference is a tractogram in the wrong place.
        """
        from zarr_vectors_tools.headers.formats import TRKHeader

        stored = {
            "format_name": "trk",
            "dimensions": [10, 10, 10],
            "voxel_size": [1.0, 1.0, 1.0],
            "origin": [0.5, 0.0, 0.0],
            "n_scalars": 0,
            "scalar_names": [],
            "n_properties": 0,
            "property_names": [],
            "vox_to_ras": [1.0] * 16,
            "voxel_order": "RAS",
            "n_count": 7,
            "version": 2,
            "space": "rasmm",
            "n_count_mismatch": [7, 5],
        }
        header = TRKHeader.from_dict(stored)
        assert header.space == "rasmm"
        assert header.origin == [0.5, 0.0, 0.0]
        assert header.version == 2
        assert header.n_count_mismatch == [7, 5]
        assert header.to_dict() == stored

    def test_an_unknown_key_is_carried_rather_than_dropped(self) -> None:
        from zarr_vectors_tools.headers.formats import TRKHeader

        header = TRKHeader.from_dict({"format_name": "trk", "future": 1})
        assert header.extra == {"future": 1}
        assert header.to_dict()["future"] == 1


class TestTargetOnlyColumns:
    """Columns the target has and the source does not.

    Both failures here are in the filler, and both survive a smoke test
    because a float column filled with NaN is exactly right.
    """

    def test_multi_channel_column_does_not_break_the_merge(
        self, tmp_path: Path,
    ) -> None:
        """``--compute-endpoints`` writes ``start``/``end`` as (O, 3).

        A 1-D filler appended to those raised "append shape mismatch", so
        merging into any store built with endpoints failed outright.
        """
        make_store(
            tmp_path / "t.zarrvectors",
            [line([1, 1, 1], [2, 2, 2])],
            object_attributes={
                "start": np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
            },
        )
        merge_stores(
            str(tmp_path / "t.zarrvectors"),
            [GeometrySource(
                Geometry(kind="streamline", parts=[line([6, 6, 6], [7, 7, 7])]),
                label="b",
            )],
            pyramid="drop",
        )
        level = zv.open(str(tmp_path / "t.zarrvectors"), mode="r").level(0)
        start = np.asarray(read_object_attributes(level.store, "start"))
        assert start.shape == (2, 3)
        np.testing.assert_allclose(start[0], [1.0, 1.0, 1.0])
        assert np.isnan(start[1]).all()

    def test_integer_column_is_filled_with_a_real_sentinel(
        self, tmp_path: Path,
    ) -> None:
        """NaN cast into a uint32 column lands as 0, which reads as data."""
        from zarr_vectors.building import (
            create_object_attributes_array,
            write_object_attributes,
        )

        dataset = make_store(
            tmp_path / "t.zarrvectors", [line([1, 1, 1], [2, 2, 2])],
        )
        create_object_attributes_array(
            dataset.level(0).store, "label", dtype="uint32",
        )
        write_object_attributes(
            dataset.level(0).store, "label", np.array([7], dtype=np.uint32),
        )

        merge_stores(
            str(tmp_path / "t.zarrvectors"),
            [GeometrySource(
                Geometry(kind="streamline", parts=[line([6, 6, 6], [7, 7, 7])]),
                label="b",
            )],
            pyramid="drop",
        )
        level = zv.open(str(tmp_path / "t.zarrvectors"), mode="r").level(0)
        label = np.asarray(read_object_attributes(level.store, "label"))
        assert int(label[0]) == 7
        assert int(label[1]) == np.iinfo(label.dtype).max
