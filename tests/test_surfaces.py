"""Cortical surfaces: GIFTI and FreeSurfer ingest, CIFTI attach.

A surface store holds one mesh object per hemisphere, chunks one anatomical
surface, keeps the other surfaces of the same topology as ``coords_<name>``
attributes, and stamps every vertex with a join key built from its hemisphere
and SOURCE vertex number.  Most of what is tested here is that last promise:
whatever reordering chunking does, every value read back sits on the vertex it
was written for, and data that arrives later indexed by source vertex (CIFTI)
lands there too.

All inputs are synthetic and written with nibabel's own writers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nibabel")

from zarr_vectors.exceptions import IngestError  # noqa: E402

from tests._surface_fixtures import (  # noqa: E402
    CRAS,
    PARCELS,
    PRIMARY,
    brain_models,
    parcels,
    read_surface_columns,
    sheet,
    thickness,
    write_cifti,
    write_freesurfer_subject,
    write_gifti_subject,
    write_label_gii,
    write_shape_gii,
    write_surf_gii,
)
from zarr_vectors_tools.headers.registry import HeaderRegistry  # noqa: E402
from zarr_vectors_tools.ingest._surface_store import map_name  # noqa: E402
from zarr_vectors_tools.ingest.cifti import attach_cifti  # noqa: E402
from zarr_vectors_tools.ingest.freesurfer import ingest_freesurfer  # noqa: E402
from zarr_vectors_tools.ingest.gifti import (  # noqa: E402
    classify_gifti,
    ingest_gifti,
)

CHUNK = (20.0, 20.0, 20.0)
N = len(sheet("left")[0])


def _split(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = keys.astype(np.int64)
    return keys >> 32, keys & 0xFFFFFFFF


@pytest.fixture
def gifti_store(tmp_path: Path) -> Path:
    source = write_gifti_subject(tmp_path / "gii")
    store = tmp_path / "surf.zarrvectors"
    ingest_gifti(source, store, CHUNK)
    return store


# ===========================================================================
# GIFTI
# ===========================================================================

class TestGiftiIngest:

    def test_every_value_lands_on_its_source_vertex(
        self, gifti_store: Path,
    ) -> None:
        positions, columns = read_surface_columns(gifti_store, {
            "thickness": 1, "aparc": 1, "coords_pial": 3, "coords_inflated": 3,
        })
        hemi, vertex = _split(columns["zv_join_key"])
        assert len(positions) == 2 * N
        assert np.array_equal(np.bincount(hemi), [N, N])
        for code, hemisphere in enumerate(("left", "right")):
            mine = hemi == code
            source, _ = sheet(hemisphere)
            idx = vertex[mine]
            np.testing.assert_allclose(positions[mine], source[idx], atol=1e-4)
            np.testing.assert_allclose(
                columns["coords_pial"][mine], source[idx] + [0, 0, 1], atol=1e-4,
            )
            np.testing.assert_allclose(
                columns["coords_inflated"][mine], source[idx] * 1.5, atol=1e-4,
            )
            np.testing.assert_allclose(
                columns["thickness"][mine], thickness(N)[idx], atol=1e-5,
            )
            np.testing.assert_array_equal(
                columns["aparc"][mine], parcels(N)[idx],
            )

    def test_the_header_describes_the_hemispheres(
        self, gifti_store: Path,
    ) -> None:
        header = HeaderRegistry(str(gifti_store)).get("surface")
        assert header.source == "gifti"
        assert header.geometry == "midthickness"
        assert header.space == "talairach"
        assert [h["hemisphere"] for h in header.hemispheres] == ["left", "right"]
        assert [h["object_id"] for h in header.hemispheres] == [0, 1]
        assert header.hemispheres[0]["structure"] == "CIFTI_STRUCTURE_CORTEX_LEFT"
        assert all(h["n_vertices"] == N for h in header.hemispheres)
        assert header.alternates == {
            "coords_inflated": "inflated", "coords_pial": "pial",
        }
        assert header.object_for("right") == 1
        table = header.label_tables["aparc"]
        assert table["1"]["name"] == "precentral"
        np.testing.assert_allclose(table["1"]["rgba"], PARCELS[1][1], atol=1e-6)

    def test_hemispheres_are_named_groups(self, gifti_store: Path) -> None:
        from zarr_vectors.building import (
            get_resolution_level,
            open_store,
            read_all_groupings,
        )
        from zarr_vectors.constants import GROUPS

        level = get_resolution_level(open_store(str(gifti_store)), 0)
        groups = read_all_groupings(level)
        assert [np.asarray(g).tolist() for g in groups] == [[0], [1]]
        assert level.read_array_meta(GROUPS)["group_names"] == ["left", "right"]

    def test_an_inflated_surface_tagged_midthickness_is_still_inflated(
        self, tmp_path: Path,
    ) -> None:
        """HCP writes AnatomicalStructureSecondary=MidThickness on the
        inflated surface.  Read first, that field would chunk a balloon."""
        vertices, faces = sheet("left")
        path = write_surf_gii(
            tmp_path / "100307.L.inflated.32k_fs_LR.surf.gii", vertices, faces,
            geometric="Inflated", secondary="MidThickness",
        )
        entry = classify_gifti(path)
        assert (entry.kind, entry.hemisphere, entry.surface) == (
            "surface", "left", "inflated",
        )

    def test_very_inflated_is_not_confused_with_inflated(
        self, tmp_path: Path,
    ) -> None:
        vertices, faces = sheet("right")
        path = write_surf_gii(
            tmp_path / "100307.R.very_inflated.32k_fs_LR.surf.gii",
            vertices, faces, geometric="Inflated",
        )
        assert classify_gifti(path).surface == "very_inflated"

    def test_only_non_anatomical_surfaces_are_refused(
        self, tmp_path: Path,
    ) -> None:
        vertices, faces = sheet("left")
        write_surf_gii(tmp_path / "lh.inflated.surf.gii", vertices, faces,
                       geometric="Inflated")
        with pytest.raises(IngestError, match="non-anatomical"):
            ingest_gifti(tmp_path, tmp_path / "out.zv", CHUNK)
        # ...unless asked for by name.
        summary = ingest_gifti(tmp_path / "lh.inflated.surf.gii",
                               tmp_path / "out2.zv", CHUNK, geometry="inflated")
        assert summary["geometry"] == "inflated"

    def test_a_map_from_another_mesh_names_both_counts(
        self, tmp_path: Path,
    ) -> None:
        vertices, faces = sheet("left")
        write_surf_gii(tmp_path / "sub-01_hemi-L_white.surf.gii", vertices, faces)
        write_shape_gii(tmp_path / "sub-01_hemi-L_curv.shape.gii",
                        [np.zeros(N + 7)])
        with pytest.raises(IngestError, match=rf"{N + 7} values.*{N} vertices"):
            ingest_gifti(tmp_path, tmp_path / "out.zv", CHUNK)

    def test_hcp_filenames_give_the_hemisphere_without_metadata(
        self, tmp_path: Path,
    ) -> None:
        vertices, faces = sheet("right")
        path = write_surf_gii(
            tmp_path / "100307.R.midthickness.32k_fs_LR.surf.gii",
            vertices, faces,
        )
        assert classify_gifti(path).hemisphere == "right"

    def test_an_unplaceable_file_is_refused_until_told(
        self, tmp_path: Path,
    ) -> None:
        vertices, faces = sheet("left")
        write_surf_gii(tmp_path / "cortex.surf.gii", vertices, faces)
        with pytest.raises(IngestError, match="which hemisphere"):
            ingest_gifti(tmp_path, tmp_path / "out.zv", CHUNK)
        summary = ingest_gifti(tmp_path / "cortex.surf.gii", tmp_path / "ok.zv",
                               CHUNK, hemisphere="right")
        assert summary["hemispheres"] == ["right"]

    def test_a_map_for_one_hemisphere_is_filled_on_the_other(
        self, tmp_path: Path,
    ) -> None:
        source = write_gifti_subject(tmp_path / "gii")
        write_shape_gii(source / "sub-01_hemi-L_myelin.shape.gii",
                        [np.full(N, 2.5)])
        store = tmp_path / "s.zv"
        summary = ingest_gifti(source, store, CHUNK)
        assert summary["filled"] == ["myelin"]
        _, columns = read_surface_columns(store, {"myelin": 1})
        hemi, _ = _split(columns["zv_join_key"])
        assert np.all(columns["myelin"][hemi == 0] == 2.5)
        assert np.all(np.isnan(columns["myelin"][hemi == 1]))

    def test_two_files_claiming_one_surface_are_refused(
        self, tmp_path: Path,
    ) -> None:
        vertices, faces = sheet("left")
        write_surf_gii(tmp_path / "a_hemi-L_pial.surf.gii", vertices, faces)
        write_surf_gii(tmp_path / "b_hemi-L_pial.surf.gii", vertices, faces)
        with pytest.raises(IngestError, match="both .* look like the pial"):
            ingest_gifti(tmp_path, tmp_path / "out.zv", CHUNK)

    def test_unnamed_arrays_in_one_file_are_one_multicolumn_attribute(
        self, tmp_path: Path,
    ) -> None:
        vertices, faces = sheet("left")
        write_surf_gii(tmp_path / "sub-01_hemi-L_white.surf.gii", vertices, faces)
        frames = [np.arange(N) * (t + 1.0) for t in range(4)]
        write_shape_gii(tmp_path / "sub-01_hemi-L_bold.func.gii", frames,
                        intent="NIFTI_INTENT_TIME_SERIES")
        store = tmp_path / "s.zv"
        summary = ingest_gifti(tmp_path, store, CHUNK)
        assert summary["scalars"] == ["bold"]
        _, columns = read_surface_columns(store, {"bold": 4})
        _, vertex = _split(columns["zv_join_key"])
        expected = np.stack(frames, axis=1)[vertex]
        np.testing.assert_allclose(columns["bold"], expected, atol=1e-4)

    def test_named_arrays_in_one_file_stay_separate(
        self, tmp_path: Path,
    ) -> None:
        vertices, faces = sheet("left")
        write_surf_gii(tmp_path / "sub-01_hemi-L_white.surf.gii", vertices, faces)
        write_shape_gii(tmp_path / "sub-01_hemi-L_morph.shape.gii",
                        [np.ones(N), np.zeros(N)], names=["sulc", "curv"])
        summary = ingest_gifti(tmp_path, tmp_path / "s.zv", CHUNK)
        assert summary["scalars"] == ["curv", "sulc"]

    @pytest.mark.parametrize(("filename", "expected"), [
        ("sub-01_hemi-L_thickness.shape.gii", "thickness"),
        ("100307.L.thickness.32k_fs_LR.shape.gii", "thickness"),
        ("100307.MyelinMap_BC.32k_fs_LR.dscalar.nii", "MyelinMap_BC"),
        ("sub-01_hemi-R_desc-aparc.label.gii", "aparc"),
        ("100307.aparc.32k_fs_LR.dlabel.nii", "aparc"),
        ("lh.sulc.gii", "sulc"),
        ("lh.aparc.a2009s.label.gii", "aparc_a2009s"),
        ("sub-01_task-rest_space-fsLR_den-91k_bold.dtseries.nii", "bold"),
    ])
    def test_attribute_names_come_from_what_the_file_measures(
        self, filename: str, expected: str,
    ) -> None:
        assert map_name(filename) == expected


# ===========================================================================
# FreeSurfer
# ===========================================================================

class TestFreeSurferIngest:

    def test_a_subject_lands_in_scanner_space(self, tmp_path: Path) -> None:
        subject = write_freesurfer_subject(tmp_path / "bert")
        store = tmp_path / "bert.zv"
        summary = ingest_freesurfer(subject, store, CHUNK)
        assert summary["space"] == "scanner"
        assert summary["geometry"] == "midthickness"
        np.testing.assert_allclose(summary["c_ras"], CRAS)

        positions, columns = read_surface_columns(store, {
            "coords_white": 3, "coords_pial": 3, "coords_inflated": 3,
            "thickness": 1, "curv": 1, "aparc": 1,
        })
        hemi, vertex = _split(columns["zv_join_key"])
        for code, hemisphere in enumerate(("left", "right")):
            mine = hemi == code
            white, _ = sheet(hemisphere)
            white = white[vertex[mine]].astype(np.float64)
            # Midthickness is computed halfway between white and pial.
            np.testing.assert_allclose(
                positions[mine], white + [0, 0, 1.0] + CRAS, atol=1e-3,
            )
            np.testing.assert_allclose(
                columns["coords_white"][mine], white + CRAS, atol=1e-3,
            )
            np.testing.assert_allclose(
                columns["coords_pial"][mine], white + [0, 0, 2.0] + CRAS, atol=1e-3,
            )
            # Inflated is not anatomy: c_ras does not apply.
            np.testing.assert_allclose(
                columns["coords_inflated"][mine], white * 1.5, atol=1e-3,
            )
            np.testing.assert_allclose(
                columns["thickness"][mine], thickness(N)[vertex[mine]], atol=1e-5,
            )
            np.testing.assert_array_equal(
                columns["aparc"][mine], parcels(N)[vertex[mine]] - 1,
            )

        header = HeaderRegistry(str(store)).get("surface")
        assert header.source == "freesurfer"
        np.testing.assert_allclose(header.c_ras, CRAS)
        assert header.extra["c_ras_applied"] is True
        assert header.extra["midthickness_computed"] is True
        assert header.extra["subject"] == "bert"
        assert header.label_tables["aparc"]["0"]["name"] == "precentral"
        np.testing.assert_allclose(
            header.label_tables["aparc"]["0"]["rgba"], [229 / 255, 25 / 255, 25 / 255, 1.0],
            atol=1e-6,
        )

    def test_surface_space_leaves_coordinates_alone(self, tmp_path: Path) -> None:
        subject = write_freesurfer_subject(tmp_path / "bert")
        store = tmp_path / "bert.zv"
        summary = ingest_freesurfer(subject, store, CHUNK, space="surface",
                                    geometry="white")
        assert summary["space"] == "surface"
        positions, columns = read_surface_columns(store, {})
        hemi, vertex = _split(columns["zv_join_key"])
        white, _ = sheet("left")
        np.testing.assert_allclose(
            positions[hemi == 0], white[vertex[hemi == 0]], atol=1e-3,
        )
        # c_ras is still recorded, so the shift can be applied later.
        np.testing.assert_allclose(
            HeaderRegistry(str(store)).get("surface").c_ras, CRAS,
        )

    def test_without_a_footer_auto_keeps_surface_space_and_scanner_refuses(
        self, tmp_path: Path,
    ) -> None:
        subject = write_freesurfer_subject(tmp_path / "bert", with_cras=False)
        summary = ingest_freesurfer(subject, tmp_path / "a.zv", CHUNK)
        assert summary["space"] == "surface"
        assert summary["c_ras"] is None
        with pytest.raises(IngestError, match="needs c_ras"):
            ingest_freesurfer(subject, tmp_path / "b.zv", CHUNK, space="scanner")

    def test_defaults_are_optional_but_named_files_are_required(
        self, tmp_path: Path,
    ) -> None:
        subject = write_freesurfer_subject(tmp_path / "bert")
        # sulc and area are default morphometry, absent here: skipped.
        summary = ingest_freesurfer(subject, tmp_path / "a.zv", CHUNK)
        assert summary["scalars"] == ["curv", "thickness"]
        with pytest.raises(IngestError, match="morphometry 'sulc'"):
            ingest_freesurfer(subject, tmp_path / "b.zv", CHUNK,
                              morphometry=["sulc"])

    def test_the_surf_directory_itself_is_accepted(self, tmp_path: Path) -> None:
        subject = write_freesurfer_subject(tmp_path / "bert")
        summary = ingest_freesurfer(subject / "surf", tmp_path / "a.zv", CHUNK)
        assert summary["labels"] == ["aparc"]  # label/ found beside surf/
        assert summary["subject"] == "bert"

    def test_one_hemisphere(self, tmp_path: Path) -> None:
        subject = write_freesurfer_subject(tmp_path / "bert", hemispheres=("right",))
        summary = ingest_freesurfer(subject, tmp_path / "a.zv", CHUNK)
        assert summary["hemispheres"] == ["right"]
        _, columns = read_surface_columns(tmp_path / "a.zv", {})
        hemi, _ = _split(columns["zv_join_key"])
        assert set(hemi.tolist()) == {1}  # right keeps its code

    def test_a_non_anatomical_geometry_is_refused(self, tmp_path: Path) -> None:
        subject = write_freesurfer_subject(tmp_path / "bert")
        with pytest.raises(IngestError, match="not an anatomical surface"):
            ingest_freesurfer(subject, tmp_path / "a.zv", CHUNK,
                              geometry="inflated")

    def test_a_directory_that_is_not_a_subject_is_refused(
        self, tmp_path: Path,
    ) -> None:
        with pytest.raises(IngestError, match="not a FreeSurfer subject"):
            ingest_freesurfer(tmp_path, tmp_path / "a.zv", CHUNK)


# ===========================================================================
# CIFTI
# ===========================================================================

#: The "medial wall": source vertices a CIFTI file has no value for.
MEDIAL = np.arange(0, N, 9)
CORTEX = np.setdiff1d(np.arange(N), MEDIAL)


def _both_cortex(voxels: int = 0):
    return brain_models(
        {"left": N, "right": N}, {"left": CORTEX, "right": CORTEX},
        voxels=voxels,
    )


class TestCiftiAttach:

    def test_dscalar_maps_land_on_their_vertices_and_the_wall_is_nan(
        self, gifti_store: Path, tmp_path: Path,
    ) -> None:
        from nibabel import cifti2

        axis = _both_cortex(voxels=5)
        k = len(CORTEX)
        myelin = np.concatenate([CORTEX * 0.01, CORTEX * 0.02 + 5, np.zeros(5)])
        curv = np.concatenate([-CORTEX * 1.0, -CORTEX * 2.0, np.zeros(5)])
        path = write_cifti(
            tmp_path / "sub.maps.dscalar.nii",
            cifti2.ScalarAxis(["myelin", "curvature"]), axis,
            np.stack([myelin, curv]),
        )
        summary = attach_cifti(gifti_store, path)
        assert summary["attributes"] == ["curvature", "myelin"]
        assert summary["voxels_skipped"] == 5
        assert summary["vertices_unmatched"] == 2 * len(MEDIAL)
        assert [h["values"] for h in summary["hemispheres"]] == [k, k]

        _, columns = read_surface_columns(gifti_store, {"myelin": 1, "curvature": 1})
        hemi, vertex = _split(columns["zv_join_key"])
        wall = np.isin(vertex, MEDIAL)
        assert np.all(np.isnan(columns["myelin"][wall]))
        left = (hemi == 0) & ~wall
        right = (hemi == 1) & ~wall
        np.testing.assert_allclose(columns["myelin"][left], vertex[left] * 0.01, atol=1e-4)
        np.testing.assert_allclose(columns["myelin"][right], vertex[right] * 0.02 + 5, atol=1e-4)
        np.testing.assert_allclose(columns["curvature"][right], -vertex[right] * 2.0, atol=1e-4)

        header = HeaderRegistry(str(gifti_store)).get("surface")
        assert header.scalars["myelin"] == "sub.maps.dscalar.nii"

    def test_dlabel_writes_codes_and_a_label_table(
        self, gifti_store: Path, tmp_path: Path,
    ) -> None:
        from nibabel import cifti2

        table = {0: ("???", (0, 0, 0, 0)), **PARCELS}
        codes = np.concatenate([CORTEX % 3 + 1, CORTEX % 2 + 1])
        path = write_cifti(
            tmp_path / "100307.glasser.32k_fs_LR.dlabel.nii",
            cifti2.LabelAxis(["glasser"], [table]), _both_cortex(), codes[None, :],
        )
        summary = attach_cifti(gifti_store, path)
        assert summary["kind"] == "label"
        _, columns = read_surface_columns(gifti_store, {"glasser": 1})
        hemi, vertex = _split(columns["zv_join_key"])
        wall = np.isin(vertex, MEDIAL)
        assert np.all(columns["glasser"][wall] == -1)
        left = (hemi == 0) & ~wall
        np.testing.assert_array_equal(columns["glasser"][left], vertex[left] % 3 + 1)

        header = HeaderRegistry(str(gifti_store)).get("surface")
        assert header.label_tables["glasser"]["2"]["name"] == "postcentral"
        assert "-1" in header.label_tables["glasser"]
        assert "glasser" not in header.scalars

    def test_dtseries_is_one_multicolumn_attribute(
        self, gifti_store: Path, tmp_path: Path,
    ) -> None:
        from nibabel import cifti2

        frames = 6
        data = np.stack([
            np.concatenate([CORTEX + t, CORTEX * 10.0 + t]) for t in range(frames)
        ])
        path = write_cifti(
            tmp_path / "rest.dtseries.nii",
            cifti2.SeriesAxis(start=0.0, step=0.72, size=frames), _both_cortex(),
            data,
        )
        summary = attach_cifti(gifti_store, path, name="rest")
        assert summary["attributes"] == ["rest"]
        _, columns = read_surface_columns(gifti_store, {"rest": frames})
        hemi, vertex = _split(columns["zv_join_key"])
        right = (hemi == 1) & ~np.isin(vertex, MEDIAL)
        expected = vertex[right][:, None] * 10.0 + np.arange(frames)[None, :]
        np.testing.assert_allclose(columns["rest"][right], expected, atol=1e-3)
        header = HeaderRegistry(str(gifti_store)).get("surface")
        assert header.extra["series"]["rest"]["step"] == pytest.approx(0.72)

    def test_a_different_mesh_resolution_is_refused(
        self, gifti_store: Path, tmp_path: Path,
    ) -> None:
        """Joining would 'work' for the first N vertex numbers and put every
        value on the wrong vertex; it must be refused instead."""
        from nibabel import cifti2

        axis = brain_models({"left": N * 4}, {"left": np.arange(N)})
        path = write_cifti(tmp_path / "x.dscalar.nii", cifti2.ScalarAxis(["x"]),
                           axis, np.zeros((1, N)))
        with pytest.raises(IngestError, match=rf"{N * 4}-vertex mesh"):
            attach_cifti(gifti_store, path)

    def test_a_store_without_a_surface_header_is_refused(
        self, tmp_path: Path,
    ) -> None:
        from nibabel import cifti2
        from zarr_vectors.types.meshes import write_mesh

        vertices, faces = sheet("left")
        store = tmp_path / "plain.zv"
        write_mesh(str(store), vertices, faces.astype(np.int64), chunk_shape=CHUNK)
        path = write_cifti(
            tmp_path / "x.dscalar.nii", cifti2.ScalarAxis(["x"]),
            brain_models({"left": N}, {"left": np.arange(N)}), np.zeros((1, N)),
        )
        with pytest.raises(IngestError, match="not a surface store"):
            attach_cifti(store, path)

    def test_name_keeps_a_multimap_dscalar_together(
        self, gifti_store: Path, tmp_path: Path,
    ) -> None:
        from nibabel import cifti2

        k = 2 * len(CORTEX)
        path = write_cifti(
            tmp_path / "m.dscalar.nii", cifti2.ScalarAxis(["a", "b", "c"]),
            _both_cortex(), np.arange(3 * k, dtype=float).reshape(3, k),
        )
        summary = attach_cifti(gifti_store, path, name="stack")
        assert summary["attributes"] == ["stack"]
        _, columns = read_surface_columns(gifti_store, {"stack": 3})
        assert columns["stack"].shape == (2 * N, 3)


# ===========================================================================
# Pyramid
# ===========================================================================

class TestSurfacePyramid:

    def test_labels_and_alternates_survive_coarsening(
        self, gifti_store: Path,
    ) -> None:
        from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

        build_pyramid(
            str(gifti_store), factors=[(2.0, 1.0)], method="mesh",
            cross_level_depth=0, cross_level_storage="none",
        )
        from zarr_vectors.building import (
            get_resolution_level,
            list_chunk_keys,
            open_store,
            read_chunk_attributes,
        )

        from zarr_vectors.building import read_chunk_vertices

        level = get_resolution_level(open_store(str(gifti_store)), 1)
        chunks = list_chunk_keys(level, "vertices")
        positions = np.concatenate([
            np.asarray(block)
            for chunk in chunks
            for block in read_chunk_vertices(level, chunk, ndim=3)
        ])
        codes = np.concatenate([
            np.asarray(block).ravel()
            for chunk in chunks
            for block in read_chunk_attributes(level, "aparc", chunk)
        ])
        pial = np.concatenate([
            np.asarray(block).reshape(-1, 3)
            for chunk in chunks
            for block in read_chunk_attributes(level, "coords_pial", chunk, ncols=3)
        ])
        assert 0 < len(positions) < 2 * N
        assert len(codes) == len(positions)
        assert set(np.unique(codes).tolist()) <= set(PARCELS)
        # The pial surface sits 1 mm above midthickness everywhere, so any
        # averaging the coarsener does must keep that offset per vertex.
        assert pial.shape == positions.shape
        np.testing.assert_allclose(pial - positions, [[0, 0, 1]] * len(pial), atol=1e-3)


# ===========================================================================
# CLI
# ===========================================================================

class TestCli:

    def test_convert_detects_a_gifti_directory(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.cli import main

        source = write_gifti_subject(tmp_path / "gii")
        store = tmp_path / "s.zarrvectors"
        assert main(["convert", str(source), str(store),
                     "--chunk-shape", "20,20,20",
                     "--coarsen", "2", "--sparsity", "1"]) == 0
        assert HeaderRegistry(str(store)).get("surface").source == "gifti"
        from zarr_vectors.building import open_store, read_level_metadata

        assert read_level_metadata(open_store(str(store)), 1).coarsening_method in (
            "mesh", "mesh_decimate",
        )

    def test_convert_detects_a_freesurfer_subject(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.cli import main

        subject = write_freesurfer_subject(tmp_path / "bert")
        store = tmp_path / "s.zarrvectors"
        assert main([
            "convert", str(subject), str(store), "--chunk-shape", "20,20,20",
            "--space", "surface", "--hemisphere", "lh", "--annot", "aparc",
        ]) == 0
        header = HeaderRegistry(str(store)).get("surface")
        assert header.space == "surface"
        assert [h["hemisphere"] for h in header.hemispheres] == ["left"]

    def test_surface_flags_are_refused_for_other_formats(
        self, tmp_path: Path,
    ) -> None:
        from zarr_vectors_tools.cli import main

        source = tmp_path / "p.csv"
        source.write_text("x,y,z\n1,2,3\n4,5,6\n")
        with pytest.raises(SystemExit, match="--geometry applies to"):
            main(["convert", str(source), str(tmp_path / "o.zv"),
                  "--chunk-shape", "5,5,5", "--geometry", "pial"])

    def test_a_directory_of_something_else_is_refused(
        self, tmp_path: Path,
    ) -> None:
        from zarr_vectors_tools.cli import main

        (tmp_path / "notes.txt").write_text("hello")
        with pytest.raises(SystemExit, match="neither a FreeSurfer subject"):
            main(["convert", str(tmp_path), str(tmp_path / "o.zv"),
                  "--chunk-shape", "5,5,5"])

    def test_attach_detects_cifti(
        self, gifti_store: Path, tmp_path: Path,
    ) -> None:
        from nibabel import cifti2

        from zarr_vectors_tools.cli import main

        k = 2 * len(CORTEX)
        path = write_cifti(tmp_path / "sub-01.thick.dscalar.nii",
                           cifti2.ScalarAxis(["thick"]), _both_cortex(),
                           np.ones((1, k)))
        assert main(["attach", str(gifti_store), str(path)]) == 0
        assert "thick" in HeaderRegistry(str(gifti_store)).get("surface").scalars

    def test_table_flags_are_refused_for_cifti(
        self, gifti_store: Path, tmp_path: Path,
    ) -> None:
        from zarr_vectors_tools.cli import main

        with pytest.raises(SystemExit, match="--key-column do not apply to cifti"):
            main(["attach", str(gifti_store), str(tmp_path / "x.dscalar.nii"),
                  "--key-column", "id"])

    def test_name_is_refused_for_a_table(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.cli import main

        with pytest.raises(SystemExit, match="--name applies to cifti"):
            main(["attach", str(tmp_path / "s.zv"), str(tmp_path / "t.csv"),
                  "--name", "x"])


def test_label_gifti_without_primary_uses_filename(tmp_path: Path) -> None:
    path = write_label_gii(tmp_path / "rh.aparc.label.gii", parcels(N))
    entry = classify_gifti(path)
    assert (entry.kind, entry.hemisphere) == ("label", "right")
    assert PRIMARY["right"] == "CortexRight"
