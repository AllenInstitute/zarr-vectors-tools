"""Neuroglancer Precomputed ingest and export tests.

Dependency tests (no cloud-volume needed) check the install-hint
error paths. Round-trip tests are guarded with
``pytest.importorskip("cloudvolume")``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _cloudvolume_installed() -> bool:
    try:
        import cloudvolume  # noqa: F401
        return True
    except ImportError:
        return False


skip_if_cv_installed = pytest.mark.skipif(
    _cloudvolume_installed(),
    reason="dependency-error test only meaningful when cloud-volume is missing",
)

# Precomputed file naming uses ``<seg_id>:<lod>`` which is illegal on
# Windows filesystems. Local-FS round-trips are skipped there.
skip_on_windows = pytest.mark.skipif(
    sys.platform == "win32",
    reason="precomputed format uses ':' in filenames; not supported on Windows local FS",
)


# -------------------------------------------------------------------
# Dependency-not-installed paths
# -------------------------------------------------------------------

@skip_if_cv_installed
class TestIngestDeps:

    def test_mesh_missing_dep(self, tmp_path: Path) -> None:
        from zarr_vectors.exceptions import IngestError

        from zarr_vectors_tools.ingest.precomputed import ingest_precomputed_mesh
        try:
            ingest_precomputed_mesh(
                tmp_path / "src", tmp_path / "out.zv", (100.0, 100.0, 100.0),
            )
        except IngestError as e:
            assert "cloud-volume" in str(e).lower()
        except Exception:
            pass

    def test_skeleton_missing_dep(self, tmp_path: Path) -> None:
        from zarr_vectors.exceptions import IngestError

        from zarr_vectors_tools.ingest.precomputed import ingest_precomputed_skeleton
        try:
            ingest_precomputed_skeleton(
                tmp_path / "src", tmp_path / "out.zv", (100.0, 100.0, 100.0),
            )
        except IngestError as e:
            assert "cloud-volume" in str(e).lower()
        except Exception:
            pass

    def test_annotations_missing_dep(self, tmp_path: Path) -> None:
        from zarr_vectors.exceptions import IngestError

        from zarr_vectors_tools.ingest.precomputed import ingest_precomputed_annotations
        try:
            ingest_precomputed_annotations(
                tmp_path / "src", tmp_path / "out.zv", (100.0, 100.0, 100.0),
            )
        except IngestError as e:
            assert "cloud-volume" in str(e).lower()
        except Exception:
            pass


@skip_if_cv_installed
class TestExportDeps:

    def test_mesh_export_missing_dep(self, tmp_path: Path) -> None:
        from zarr_vectors.exceptions import ExportError

        from zarr_vectors_tools.export.precomputed import export_precomputed_mesh
        try:
            export_precomputed_mesh(tmp_path / "store.zv", tmp_path / "out")
        except ExportError as e:
            assert "cloud-volume" in str(e).lower()
        except Exception:
            pass

    def test_skeleton_export_missing_dep(self, tmp_path: Path) -> None:
        from zarr_vectors.exceptions import ExportError

        from zarr_vectors_tools.export.precomputed import export_precomputed_skeleton
        try:
            export_precomputed_skeleton(tmp_path / "store.zv", tmp_path / "out")
        except ExportError as e:
            assert "cloud-volume" in str(e).lower()
        except Exception:
            pass

    def test_annotations_export_missing_dep(self, tmp_path: Path) -> None:
        from zarr_vectors.exceptions import ExportError

        from zarr_vectors_tools.export.precomputed import export_precomputed_annotations
        try:
            export_precomputed_annotations(
                tmp_path / "store.zv", tmp_path / "out", annotation_type="POINT",
            )
        except ExportError as e:
            assert "cloud-volume" in str(e).lower()
        except Exception:
            pass


# -------------------------------------------------------------------
# Helpers to build a tiny local precomputed source via cloud-volume
# -------------------------------------------------------------------

def _mesh_src_url(tmp_path: Path) -> tuple[str, list[int]]:
    """Build a 2-segment legacy unsharded precomputed mesh layer.

    Returns (file:// URL, segment IDs).
    """
    cloudvolume = pytest.importorskip("cloudvolume")
    root = tmp_path / "mesh_src"
    root.mkdir(parents=True, exist_ok=True)
    url = "file://" + str(root.resolve()).replace("\\", "/")

    info = cloudvolume.CloudVolume.create_new_info(
        num_channels=1,
        layer_type="segmentation",
        data_type="uint64",
        encoding="raw",
        resolution=[10, 10, 10],
        voxel_offset=[0, 0, 0],
        volume_size=[1, 1, 1],
        chunk_size=[1, 1, 1],
        mesh="mesh",
    )
    cv = cloudvolume.CloudVolume(url, info=info, compress=False)
    cv.commit_info()

    mesh_dir = root / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    (mesh_dir / "info").write_text(json.dumps({"@type": "neuroglancer_legacy_mesh"}))

    segments = [101, 202]
    seg_data = {
        101: (
            np.array(
                [[0, 0, 0], [10, 0, 0], [5, 10, 0], [5, 5, 10]],
                dtype=np.float32,
            ),
            np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]], dtype=np.uint32),
        ),
        202: (
            np.array(
                [[20, 20, 20], [30, 20, 20], [25, 30, 20]],
                dtype=np.float32,
            ),
            np.array([[0, 1, 2]], dtype=np.uint32),
        ),
    }

    for sid, (verts, faces) in seg_data.items():
        fkey = f"{sid}:0"
        with open(mesh_dir / fkey, "wb") as f:
            f.write(np.uint32(len(verts)).tobytes())
            f.write(np.ascontiguousarray(verts, dtype=np.float32).tobytes())
            f.write(np.ascontiguousarray(faces, dtype=np.uint32).tobytes())
        (mesh_dir / f"{sid}:0:manifest.json").write_text(
            json.dumps({"fragments": [fkey]})
        )

    return url, segments


def _skeleton_src_url(tmp_path: Path) -> tuple[str, list[int]]:
    """Build a 2-segment legacy unsharded precomputed skeleton layer."""
    cloudvolume = pytest.importorskip("cloudvolume")
    root = tmp_path / "skel_src"
    root.mkdir(parents=True, exist_ok=True)
    url = "file://" + str(root.resolve()).replace("\\", "/")

    info = cloudvolume.CloudVolume.create_new_info(
        num_channels=1,
        layer_type="segmentation",
        data_type="uint64",
        encoding="raw",
        resolution=[10, 10, 10],
        voxel_offset=[0, 0, 0],
        volume_size=[1, 1, 1],
        chunk_size=[1, 1, 1],
        skeletons="skeletons",
    )
    cv = cloudvolume.CloudVolume(url, info=info, compress=False)
    cv.commit_info()

    skel_dir = root / "skeletons"
    skel_dir.mkdir(parents=True, exist_ok=True)
    (skel_dir / "info").write_text(json.dumps({"@type": "neuroglancer_skeletons"}))

    segments = [11, 22]
    skel_data = {
        11: (
            np.array(
                [[0, 0, 0], [10, 0, 0], [10, 10, 0], [20, 10, 0]],
                dtype=np.float32,
            ),
            np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32),
        ),
        22: (
            np.array([[50, 50, 50], [60, 50, 50]], dtype=np.float32),
            np.array([[0, 1]], dtype=np.uint32),
        ),
    }
    for sid, (verts, edges) in skel_data.items():
        with open(skel_dir / str(sid), "wb") as f:
            f.write(np.uint32(len(verts)).tobytes())
            f.write(np.uint32(len(edges)).tobytes())
            f.write(np.ascontiguousarray(verts, dtype=np.float32).tobytes())
            f.write(np.ascontiguousarray(edges, dtype=np.uint32).tobytes())

    return url, segments


# -------------------------------------------------------------------
# Round-trip tests
# -------------------------------------------------------------------

@skip_on_windows
class TestMeshRoundTrip:

    def test_ingest_two_segments(self, tmp_path: Path) -> None:
        pytest.importorskip("cloudvolume")
        from zarr_vectors_tools.ingest.precomputed import ingest_precomputed_mesh

        src_url, seg_ids = _mesh_src_url(tmp_path)
        out = tmp_path / "mesh.zv"
        summary = ingest_precomputed_mesh(
            src_url, out, (100.0, 100.0, 100.0), segment_ids=seg_ids,
        )
        assert summary["segment_count"] == 2
        assert summary["vertex_count"] == 7
        assert summary["face_count"] == 5

    def test_header_round_trip(self, tmp_path: Path) -> None:
        pytest.importorskip("cloudvolume")
        from zarr_vectors_tools.headers.registry import HeaderRegistry
        from zarr_vectors_tools.ingest.precomputed import ingest_precomputed_mesh

        src_url, seg_ids = _mesh_src_url(tmp_path)
        out = tmp_path / "mesh.zv"
        ingest_precomputed_mesh(
            src_url, out, (100.0, 100.0, 100.0), segment_ids=seg_ids,
        )
        hdr = HeaderRegistry(str(out)).get("neuroglancer")
        assert hdr.data_type == "mesh"
        assert [int(s) for s in hdr.segment_ids] == seg_ids
        assert hdr.resolution == (10.0, 10.0, 10.0)

    def test_export(self, tmp_path: Path) -> None:
        pytest.importorskip("cloudvolume")
        from zarr_vectors_tools.export.precomputed import export_precomputed_mesh
        from zarr_vectors_tools.ingest.precomputed import ingest_precomputed_mesh

        src_url, seg_ids = _mesh_src_url(tmp_path)
        store = tmp_path / "mesh.zv"
        ingest_precomputed_mesh(
            src_url, store, (100.0, 100.0, 100.0), segment_ids=seg_ids,
        )
        out_dir = tmp_path / "mesh_out"
        result = export_precomputed_mesh(store, out_dir)
        assert result["segment_count"] == 2
        assert sorted(result["segment_ids"]) == sorted(seg_ids)
        # Each segment should produce a binary fragment + manifest.
        for sid in seg_ids:
            assert (out_dir / "mesh" / f"{sid}:0").exists()
            assert (out_dir / "mesh" / f"{sid}:0:manifest.json").exists()


@skip_on_windows
class TestSkeletonRoundTrip:

    def test_ingest_two_segments(self, tmp_path: Path) -> None:
        pytest.importorskip("cloudvolume")
        from zarr_vectors_tools.ingest.precomputed import ingest_precomputed_skeleton

        src_url, seg_ids = _skeleton_src_url(tmp_path)
        out = tmp_path / "skel.zv"
        summary = ingest_precomputed_skeleton(
            src_url, out, (100.0, 100.0, 100.0), segment_ids=seg_ids,
        )
        assert summary["segment_count"] == 2
        assert summary["vertex_count"] == 6
        assert summary["edge_count"] == 4

    def test_export(self, tmp_path: Path) -> None:
        pytest.importorskip("cloudvolume")
        from zarr_vectors_tools.export.precomputed import export_precomputed_skeleton
        from zarr_vectors_tools.ingest.precomputed import ingest_precomputed_skeleton

        src_url, seg_ids = _skeleton_src_url(tmp_path)
        store = tmp_path / "skel.zv"
        ingest_precomputed_skeleton(
            src_url, store, (100.0, 100.0, 100.0), segment_ids=seg_ids,
        )
        out_dir = tmp_path / "skel_out"
        result = export_precomputed_skeleton(store, out_dir)
        assert result["segment_count"] == 2
        for sid in seg_ids:
            assert (out_dir / "skeletons" / str(sid)).exists()
        assert (out_dir / "skeletons" / "info").exists()


class TestMeshExportFromZVF:
    """Exercise mesh export against a real ZVF store without writing precomputed
    bytes (replaces the binary-fragment writer with a recorder). Works on Windows.
    """

    def test_single_segment_collapse_with_warning(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        pytest.importorskip("cloudvolume")
        import warnings

        from zarr_vectors.types.meshes import write_mesh

        from zarr_vectors_tools.export import precomputed as exp
        from zarr_vectors_tools.headers.formats import NeuroglancerHeader
        from zarr_vectors_tools.headers.registry import HeaderRegistry

        verts = np.array(
            [
                [0, 0, 0], [10, 0, 0], [5, 10, 0],
                [50, 50, 50], [60, 50, 50], [55, 60, 50], [55, 55, 60],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [[0, 1, 2], [3, 4, 5], [3, 5, 6]], dtype=np.int64,
        )
        obj_ids = np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

        store = tmp_path / "m.zv"
        write_mesh(
            str(store), verts, faces, chunk_shape=(100.0, 100.0, 100.0),
            object_ids=obj_ids,
        )
        HeaderRegistry(str(store)).add(
            "neuroglancer",
            NeuroglancerHeader(
                data_type="mesh",
                resolution=(1.0, 1.0, 1.0),
                segment_ids=["111", "222"],
            ),
        )

        recorded: list[tuple[int, int, int]] = []

        def _record(_mesh_dir, sid, v, f):
            recorded.append((int(sid), int(v.shape[0]), int(f.shape[0])))

        monkeypatch.setattr(exp, "_write_precomputed_mesh_fragment", _record)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = exp.export_precomputed_mesh(store, tmp_path / "out")

        # The collapse warning is emitted because the header had 2 segments
        # but read_mesh can't split.
        assert any(
            "single precomputed segment" in str(w.message) for w in caught
        )
        assert result["segment_count"] == 1
        assert result["segment_ids"] == [111]  # first seg_id from header
        assert recorded == [(111, 7, 3)]


class TestAnnotationsExport:
    # Annotations don't use ``:`` in filenames so this works on Windows.
    """Round-trip the public annotation writers, bypassing cloud-volume's
    annotation reader (its API surface varies a lot between versions)."""

    def test_export_points(self, tmp_path: Path) -> None:
        pytest.importorskip("cloudvolume")
        from zarr_vectors.types.points import write_points

        from zarr_vectors_tools.export.precomputed import export_precomputed_annotations
        from zarr_vectors_tools.headers.formats import NeuroglancerHeader
        from zarr_vectors_tools.headers.registry import HeaderRegistry

        store = tmp_path / "pts.zv"
        positions = np.array(
            [[0, 0, 0], [10, 5, 0], [5, 10, 5], [20, 20, 20]],
            dtype=np.float32,
        )
        write_points(str(store), positions, chunk_shape=(100.0, 100.0, 100.0))
        HeaderRegistry(str(store)).add(
            "neuroglancer",
            NeuroglancerHeader(
                data_type="annotation",
                annotation_type="POINT",
                resolution=(1.0, 1.0, 1.0),
            ),
        )

        out_dir = tmp_path / "pts_out"
        result = export_precomputed_annotations(store, out_dir)
        assert result["annotation_type"] == "POINT"
        assert result["annotation_count"] == 4
        info = json.loads((out_dir / "info").read_text())
        assert info["annotation_type"] == "POINT"
        assert (out_dir / "spatial0" / "0_0_0").exists()

    def test_export_lines(self, tmp_path: Path) -> None:
        pytest.importorskip("cloudvolume")
        from zarr_vectors.types.lines import write_lines

        from zarr_vectors_tools.export.precomputed import export_precomputed_annotations
        from zarr_vectors_tools.headers.formats import NeuroglancerHeader
        from zarr_vectors_tools.headers.registry import HeaderRegistry

        store = tmp_path / "lines.zv"
        endpoints = np.array(
            [
                [[0, 0, 0], [10, 0, 0]],
                [[5, 5, 5], [15, 15, 15]],
            ],
            dtype=np.float32,
        )
        write_lines(str(store), endpoints, chunk_shape=(100.0, 100.0, 100.0))
        HeaderRegistry(str(store)).add(
            "neuroglancer",
            NeuroglancerHeader(
                data_type="annotation",
                annotation_type="LINE",
                resolution=(1.0, 1.0, 1.0),
            ),
        )

        out_dir = tmp_path / "lines_out"
        result = export_precomputed_annotations(store, out_dir)
        assert result["annotation_type"] == "LINE"
        assert result["annotation_count"] == 2
        info = json.loads((out_dir / "info").read_text())
        assert info["annotation_type"] == "LINE"
