"""GIFTI export: a surface store back out to the files it was ingested from.

The contract under test is that nothing a GIFTI reader looks at changes on
the way through a store: ingest a subject, export it, and ``nibabel.load``
gives the same coordinates, the same triangles, the same maps and the same
label tables, in the same vertex numbering -- and ingesting the exported
directory again rebuilds the same store.

All inputs are synthetic and written with nibabel's own writers.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nibabel")

import nibabel as nib  # noqa: E402
from zarr_vectors.exceptions import ExportError  # noqa: E402

from tests._surface_fixtures import (  # noqa: E402
    CRAS,
    PARCELS,
    PRIMARY,
    brain_models,
    parcels,
    sheet,
    thickness,
    write_cifti,
    write_freesurfer_subject,
    write_gifti_subject,
    write_label_gii,
    write_shape_gii,
)
from zarr_vectors_tools.algorithms.surfaces import read_hemisphere  # noqa: E402
from zarr_vectors_tools.convert.export.gifti import (  # noqa: E402
    export_gifti,
    gifti_filename,
    metric_kind,
)
from zarr_vectors_tools.convert.ingest.freesurfer import ingest_freesurfer  # noqa: E402
from zarr_vectors_tools.convert.ingest.gifti import classify_gifti, ingest_gifti  # noqa: E402
from zarr_vectors_tools.headers.registry import HeaderRegistry  # noqa: E402

CHUNK = (20.0, 20.0, 20.0)
N = len(sheet("left")[0])
SURFACES = {"midthickness": (0.0, 1.0), "pial": ([0, 0, 1], 1.0), "inflated": (0.0, 1.5)}
FRAMES = 4


def _canonical(faces: np.ndarray) -> list[tuple[int, ...]]:
    """Triangles as a sorted list, each rotated to start at its smallest
    vertex: equal for the same triangles with the same winding, in any order."""
    rolled = []
    for tri in np.asarray(faces).tolist():
        k = tri.index(min(tri))
        rolled.append(tuple(tri[k:] + tri[:k]))
    return sorted(rolled)


def _intent(darray) -> str:
    return nib.nifti1.intent_codes.niistring[darray.intent]


def _source_surface(hemisphere: str, surface: str) -> np.ndarray:
    vertices, _ = sheet(hemisphere)
    shift, scale = SURFACES[surface]
    return (vertices * np.float32(scale) + np.asarray(shift, np.float32)).astype(np.float32)


@pytest.fixture(scope="module")
def gifti_export(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, dict]:
    """A GIFTI subject ingested and exported once, shared read-only."""
    root = tmp_path_factory.mktemp("gifti_export")
    store = root / "surf.zarrvectors"
    ingest_gifti(write_gifti_subject(root / "gii"), store, CHUNK)
    out = root / "out"
    summary = export_gifti(store, out)
    return store, out, summary


@pytest.fixture(scope="module")
def rich_export(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, dict]:
    """A subject with names BIDS cannot spell, a time series, and a map
    measured on the left hemisphere only."""
    root = tmp_path_factory.mktemp("rich_export")
    source = write_gifti_subject(root / "gii")
    write_shape_gii(source / "sub-01.L.MyelinMap_BC.func.gii",
                    [np.arange(N) * 0.5], primary=PRIMARY["left"],
                    intent="NIFTI_INTENT_NONE")
    for hemisphere, tag in (("left", "L"), ("right", "R")):
        write_shape_gii(source / f"sub-01_hemi-{tag}_bold.func.gii",
                        [np.arange(N) * (t + 1.0) for t in range(FRAMES)],
                        primary=PRIMARY[hemisphere],
                        intent="NIFTI_INTENT_TIME_SERIES")
        prefix = "lh" if hemisphere == "left" else "rh"
        write_label_gii(source / f"{prefix}.aparc.a2009s.label.gii",
                        parcels(N)[::-1].copy(), primary=PRIMARY[hemisphere])
    store = root / "rich.zarrvectors"
    ingest_gifti(source, store, CHUNK)
    out = root / "out"
    return store, out, export_gifti(store, out)


# ===========================================================================
# Fidelity: what nibabel reads back
# ===========================================================================

class TestWhatIsWritten:

    def test_one_file_per_surface_and_map_per_hemisphere(self, gifti_export) -> None:
        _, out, summary = gifti_export
        expected = {
            f"hemi-{tag}_{name}"
            for tag in ("L", "R")
            for name in ("midthickness.surf.gii", "pial.surf.gii",
                         "inflated.surf.gii", "thickness.shape.gii",
                         "aparc.label.gii")
        }
        assert {p.name for p in out.iterdir()} == expected
        assert {Path(f).name for f in summary["files"]} == expected
        assert summary["file_count"] == 10
        assert summary["hemispheres"] == ["left", "right"]
        assert summary["surfaces"][0] == "midthickness"
        assert (summary["scalars"], summary["labels"]) == (["thickness"], ["aparc"])
        assert summary["vertex_count"] == 2 * N
        assert summary["face_count"] == 2 * len(sheet("left")[1])
        assert summary["skipped"] == {} and summary["warnings"] == []

    @pytest.mark.parametrize("hemisphere", ["left", "right"])
    @pytest.mark.parametrize("surface", sorted(SURFACES))
    def test_coordinates_and_triangles_are_the_source_files(
        self, gifti_export, hemisphere: str, surface: str,
    ) -> None:
        _, out, _ = gifti_export
        tag = "L" if hemisphere == "left" else "R"
        image = nib.load(str(out / f"hemi-{tag}_{surface}.surf.gii"))
        points, triangles = image.darrays
        assert (_intent(points), _intent(triangles)) == (
            "NIFTI_INTENT_POINTSET", "NIFTI_INTENT_TRIANGLE",
        )
        assert points.data.dtype == np.float32
        assert triangles.data.dtype == np.int32
        np.testing.assert_array_equal(points.data, _source_surface(hemisphere, surface))
        assert _canonical(triangles.data) == _canonical(sheet(hemisphere)[1])

        assert image.meta["AnatomicalStructurePrimary"] == PRIMARY[hemisphere]
        geometric = {"inflated": "Inflated"}.get(surface, "Anatomical")
        assert points.meta["GeometricType"] == geometric
        secondary = {"midthickness": "MidThickness", "pial": "Pial"}.get(surface)
        assert points.meta.get("AnatomicalStructureSecondary") == secondary
        # The source files declare Talairach; an inflated surface is in no
        # anatomical space and says so.
        expected_space = "unknown" if surface == "inflated" else "talairach"
        assert nib.nifti1.xform_codes.label[points.coordsys.dataspace] == expected_space

    @pytest.mark.parametrize("hemisphere", ["left", "right"])
    def test_a_shape_map_is_the_source_values(self, gifti_export, hemisphere: str) -> None:
        _, out, _ = gifti_export
        tag = "L" if hemisphere == "left" else "R"
        image = nib.load(str(out / f"hemi-{tag}_thickness.shape.gii"))
        (darray,) = image.darrays
        assert _intent(darray) == "NIFTI_INTENT_SHAPE"
        assert darray.data.dtype == np.float32
        assert darray.meta["Name"] == "thickness"
        assert image.meta["AnatomicalStructurePrimary"] == PRIMARY[hemisphere]
        np.testing.assert_array_equal(darray.data, thickness(N))

    @pytest.mark.parametrize("hemisphere", ["left", "right"])
    def test_a_parcellation_keeps_its_codes_names_and_colours(
        self, gifti_export, hemisphere: str,
    ) -> None:
        _, out, _ = gifti_export
        tag = "L" if hemisphere == "left" else "R"
        image = nib.load(str(out / f"hemi-{tag}_aparc.label.gii"))
        (darray,) = image.darrays
        assert _intent(darray) == "NIFTI_INTENT_LABEL"
        assert darray.data.dtype == np.int32
        np.testing.assert_array_equal(darray.data, parcels(N))
        table = {
            label.key: (label.label, tuple(label.rgba)) for label in image.labeltable.labels
        }
        expected = {0: ("???", (0.0, 0.0, 0.0, 0.0)), **PARCELS}
        assert sorted(table) == sorted(expected)
        for code, (name, rgba) in expected.items():
            assert table[code][0] == name
            np.testing.assert_array_equal(table[code][1], rgba)

    def test_every_file_is_classified_as_what_it_is(self, gifti_export) -> None:
        _, out, _ = gifti_export
        for path in sorted(out.iterdir()):
            entry = classify_gifti(path)
            hemisphere = "left" if "hemi-L" in path.name else "right"
            kind = {"surf": "surface", "shape": "scalar", "label": "label"}[
                path.name.split(".")[-2]
            ]
            assert (entry.kind, entry.hemisphere) == (kind, hemisphere), path.name
            if kind == "surface":
                assert path.name.endswith(f"_{entry.surface}.surf.gii")


# ===========================================================================
# Round trip through the store
# ===========================================================================

def _assert_same_store(first: Path, second: Path) -> None:
    """Same header (bar the recorded source file names) and, hemisphere by
    hemisphere, the same vertices, triangles and maps in source order."""
    a = HeaderRegistry(str(first)).get("surface")
    b = HeaderRegistry(str(second)).get("surface")
    for field in ("source", "space", "geometry", "hemispheres", "alternates",
                  "label_tables", "key_attribute"):
        assert getattr(a, field) == getattr(b, field), field
    assert sorted(a.scalars) == sorted(b.scalars)
    maps = sorted(set(a.scalars) | set(a.label_tables))
    for entry in a.hemispheres:
        hemisphere = entry["hemisphere"]
        coords = sorted(a.alternates)
        one = read_hemisphere(first, hemisphere, attributes=[*coords, *maps])
        two = read_hemisphere(second, hemisphere, attributes=[*coords, *maps])
        np.testing.assert_array_equal(one["vertices"], two["vertices"])
        assert _canonical(one["faces"]) == _canonical(two["faces"])
        for name in [*coords, *maps]:
            assert one["attributes"][name].dtype == two["attributes"][name].dtype, name
            np.testing.assert_array_equal(
                one["attributes"][name], two["attributes"][name], err_msg=name,
            )


class TestRoundTrip:

    def test_reingesting_the_export_rebuilds_the_store(
        self, gifti_export, tmp_path: Path,
    ) -> None:
        store, out, _ = gifti_export
        again = tmp_path / "again.zarrvectors"
        ingest_gifti(out, again, CHUNK)
        _assert_same_store(store, again)

    def test_names_bids_cannot_spell_come_back_unchanged(
        self, rich_export, tmp_path: Path,
    ) -> None:
        store, out, summary = rich_export
        names = {p.name for p in out.iterdir()}
        assert "hemi-L.MyelinMap_BC.func.gii" in names
        assert {"hemi-L.aparc.a2009s.label.gii", "hemi-R.aparc.a2009s.label.gii"} <= names
        assert summary["warnings"] == []

        again = tmp_path / "again.zarrvectors"
        reingested = ingest_gifti(out, again, CHUNK)
        assert reingested["labels"] == ["aparc", "aparc.a2009s"]
        assert reingested["scalars"] == ["MyelinMap_BC", "bold", "thickness"]
        _assert_same_store(store, again)

    def test_a_map_measured_on_one_hemisphere_is_written_for_that_one(
        self, rich_export, tmp_path: Path,
    ) -> None:
        store, out, summary = rich_export
        assert summary["skipped"] == {"right": ["MyelinMap_BC"]}
        assert not (out / "hemi-R.MyelinMap_BC.func.gii").exists()
        (darray,) = nib.load(str(out / "hemi-L.MyelinMap_BC.func.gii")).darrays
        assert _intent(darray) == "NIFTI_INTENT_NONE"
        np.testing.assert_allclose(darray.data, np.arange(N) * 0.5)
        # Re-ingest fills the right hemisphere exactly as the first ingest did.
        again = tmp_path / "again.zarrvectors"
        assert ingest_gifti(out, again, CHUNK)["filled"] == ["MyelinMap_BC"]

    def test_a_multicolumn_map_is_one_array_per_column(self, rich_export) -> None:
        _, out, _ = rich_export
        image = nib.load(str(out / "hemi-R_bold.func.gii"))
        assert len(image.darrays) == FRAMES
        # Unnamed, so the GIFTI ingest keeps the columns together.
        assert all("Name" not in d.meta for d in image.darrays)
        stacked = np.stack([d.data for d in image.darrays], axis=1)
        expected = np.stack([np.arange(N) * (t + 1.0) for t in range(FRAMES)], axis=1)
        np.testing.assert_array_equal(stacked, expected.astype(np.float32))


# ===========================================================================
# FreeSurfer and CIFTI stores
# ===========================================================================

class TestOtherSources:

    def test_a_freesurfer_store_exports_in_the_space_it_was_stored_in(
        self, tmp_path: Path,
    ) -> None:
        store = tmp_path / "bert.zarrvectors"
        ingest_freesurfer(write_freesurfer_subject(tmp_path / "bert"), store, CHUNK)
        out = tmp_path / "gifti"
        summary = export_gifti(store, out, prefix="sub-bert")
        assert summary["space"] == "scanner"
        assert summary["warnings"] == []

        white, faces = sheet("left")
        image = nib.load(str(out / "sub-bert_hemi-L_white.surf.gii"))
        points, triangles = image.darrays
        # c_ras was added at ingest and is neither removed nor added again.
        np.testing.assert_allclose(points.data, white + CRAS, atol=1e-4)
        assert nib.nifti1.xform_codes.label[points.coordsys.dataspace] == "scanner"
        assert points.meta["AnatomicalStructureSecondary"] == "GrayWhite"
        assert not any(key.startswith("VolGeom") for key in points.meta)
        assert _canonical(triangles.data) == _canonical(faces)

        inflated = nib.load(str(out / "sub-bert_hemi-L_inflated.surf.gii")).darrays[0]
        np.testing.assert_allclose(inflated.data, white * 1.5, atol=1e-4)
        assert inflated.coordsys.dataspace == 0

        for name in ("thickness", "curv"):
            (darray,) = nib.load(str(out / f"sub-bert_hemi-L_{name}.shape.gii")).darrays
            assert _intent(darray) == "NIFTI_INTENT_SHAPE"
        np.testing.assert_array_equal(
            nib.load(str(out / "sub-bert_hemi-L_thickness.shape.gii")).darrays[0].data,
            thickness(N),
        )
        label = nib.load(str(out / "sub-bert_hemi-R_aparc.label.gii"))
        np.testing.assert_array_equal(label.darrays[0].data, parcels(N) - 1)
        assert [lab.label for lab in label.labeltable.labels] == [
            name for name, _ in PARCELS.values()
        ]

        again = tmp_path / "again.zarrvectors"
        assert ingest_gifti(out, again, CHUNK)["space"] == "scanner"
        _assert_same_store_geometry(store, again)

    def test_a_surface_ras_store_has_no_nifti_space(self, tmp_path: Path) -> None:
        store = tmp_path / "bert.zarrvectors"
        ingest_freesurfer(write_freesurfer_subject(tmp_path / "bert"), store, CHUNK,
                          space="surface", geometry="white", alternates=[])
        out = tmp_path / "gifti"
        export_gifti(store, out, attribute_names=[])
        (points, _) = nib.load(str(out / "hemi-L_white.surf.gii")).darrays
        assert points.coordsys.dataspace == 0
        np.testing.assert_allclose(points.data, sheet("left")[0], atol=1e-4)

    def test_cifti_maps_keep_their_kind(self, gifti_export, tmp_path: Path) -> None:
        from nibabel import cifti2

        from zarr_vectors_tools.convert.ingest.cifti import attach_cifti

        store = tmp_path / "surf.zarrvectors"
        shutil.copytree(gifti_export[0], store)
        medial = np.arange(0, N, 9)
        cortex = np.setdiff1d(np.arange(N), medial)
        axis = brain_models({"left": N, "right": N}, {"left": cortex, "right": cortex})
        k = 2 * len(cortex)
        attach_cifti(store, write_cifti(
            tmp_path / "sub-01_myelin.dscalar.nii", cifti2.ScalarAxis(["myelin"]), axis,
            np.arange(k, dtype=float)[None, :],
        ))
        attach_cifti(store, write_cifti(
            tmp_path / "rest.dtseries.nii",
            cifti2.SeriesAxis(start=0.0, step=0.72, size=3), axis,
            np.arange(3 * k, dtype=float).reshape(3, k),
        ), name="rest")
        table = {0: ("???", (0, 0, 0, 0)), **PARCELS}
        attach_cifti(store, write_cifti(
            tmp_path / "glasser.dlabel.nii", cifti2.LabelAxis(["glasser"], [table]),
            axis, (np.concatenate([cortex, cortex]) % 3 + 1)[None, :],
        ))

        out = tmp_path / "gifti"
        summary = export_gifti(store, out, hemispheres=["rh"],
                               attribute_names=["myelin", "rest", "glasser"])
        assert summary["hemispheres"] == ["right"]

        (myelin,) = nib.load(str(out / "hemi-R_myelin.func.gii")).darrays
        assert _intent(myelin) == "NIFTI_INTENT_NONE"
        assert np.isnan(myelin.data[medial]).all()

        rest = nib.load(str(out / "hemi-R_rest.func.gii"))
        assert [_intent(d) for d in rest.darrays] == ["NIFTI_INTENT_TIME_SERIES"] * 3
        assert float(rest.meta["TimeStep"]) == pytest.approx(0.72)

        glasser = nib.load(str(out / "hemi-R_glasser.label.gii"))
        codes = glasser.darrays[0].data
        assert np.all(codes[medial] == -1)
        np.testing.assert_array_equal(codes[cortex], cortex % 3 + 1)
        keys = {label.key: label.label for label in glasser.labeltable.labels}
        assert keys[-1] == "no data (medial wall)" and keys[2] == "postcentral"


def _assert_same_store_geometry(first: Path, second: Path) -> None:
    """A FreeSurfer store re-ingested from GIFTI: same surfaces and maps.

    The header's ``source`` and FreeSurfer-only extras differ by design.
    """
    a = HeaderRegistry(str(first)).get("surface")
    b = HeaderRegistry(str(second)).get("surface")
    assert (a.geometry, a.hemispheres, a.label_tables) == (b.geometry, b.hemispheres,
                                                          b.label_tables)
    assert sorted(a.alternates) == sorted(b.alternates)
    names = [*sorted(a.alternates), *sorted(a.scalars), *sorted(a.label_tables)]
    for hemisphere in ("left", "right"):
        one = read_hemisphere(first, hemisphere, attributes=names)
        two = read_hemisphere(second, hemisphere, attributes=names)
        np.testing.assert_array_equal(one["vertices"], two["vertices"])
        for name in names:
            np.testing.assert_array_equal(one["attributes"][name], two["attributes"][name])


# ===========================================================================
# Choosing what to write, and refusals
# ===========================================================================

class TestSelection:

    def test_subsets_and_a_prefix(self, gifti_export, tmp_path: Path) -> None:
        store, _, _ = gifti_export
        out = tmp_path / "subset"
        summary = export_gifti(store, out, hemispheres="lh", surfaces=["coords_inflated"],
                               attribute_names=["aparc"], prefix="sub-01")
        assert sorted(p.name for p in out.iterdir()) == [
            "sub-01_hemi-L_aparc.label.gii", "sub-01_hemi-L_inflated.surf.gii",
        ]
        assert summary["surfaces"] == ["inflated"]
        assert (summary["scalars"], summary["labels"]) == ([], ["aparc"])
        assert summary["vertex_count"] == N

    def test_a_name_ingest_would_change_is_warned_about(
        self, rich_export, tmp_path: Path,
    ) -> None:
        store, _, _ = rich_export
        summary = export_gifti(store, tmp_path / "o", hemispheres=["left"],
                               surfaces=[], attribute_names=["aparc.a2009s", "thickness"],
                               prefix="bert")
        # "bert" is not a BIDS entity, so before a dotted name it reads as
        # part of the map name; before a BIDS-style name it is dropped.
        assert len(summary["warnings"]) == 1
        assert "bert_hemi-L.aparc.a2009s.label.gii" in summary["warnings"][0]

    @pytest.mark.parametrize(("kwargs", "message"), [
        ({"hemispheres": ["middle"]}, "left, right, lh or rh"),
        ({"surfaces": ["sphere"]}, "no 'sphere' surface"),
        ({"attribute_names": ["curv"]}, "'curv' is not at level 0"),
        ({"attribute_names": ["coords_pial"]}, r"surfaces=\['pial'\]"),
        ({"attribute_names": ["zv_join_key"]}, "join key"),
        ({"level": 3}, "level 3 is not in"),
        ({"prefix": "a/b"}, "path separators"),
    ])
    def test_what_the_store_lacks_is_refused_by_name(
        self, gifti_export, tmp_path: Path, kwargs: dict, message: str,
    ) -> None:
        store, _, _ = gifti_export
        with pytest.raises(ExportError, match=message):
            export_gifti(store, tmp_path / "o", **kwargs)

    def test_a_missing_hemisphere_is_refused(self, tmp_path: Path) -> None:
        store = tmp_path / "left.zarrvectors"
        ingest_gifti(write_gifti_subject(tmp_path / "gii", hemispheres=("left",)),
                     store, CHUNK)
        with pytest.raises(ExportError, match=r"no right hemisphere; it has \['left'\]"):
            export_gifti(store, tmp_path / "o", hemispheres=["rh"])

    def test_a_file_name_for_the_output_is_refused(self, gifti_export, tmp_path: Path) -> None:
        store, _, _ = gifti_export
        with pytest.raises(ExportError, match="writes a directory"):
            export_gifti(store, tmp_path / "lh.pial.surf.gii")

    def test_a_store_that_is_not_a_surface_is_refused(self, tmp_path: Path) -> None:
        from zarr_vectors.types.points import write_points

        store = tmp_path / "points.zarrvectors"
        write_points(str(store), np.ones((3, 3), np.float32), chunk_shape=CHUNK)
        with pytest.raises(ExportError, match="not a surface store"):
            export_gifti(store, tmp_path / "o")


@pytest.mark.parametrize(("name", "source", "series", "expected"), [
    ("thickness", "sub-01_hemi-L_thickness.shape.gii", False, "shape"),
    ("MyelinMap_BC", "100307.L.MyelinMap_BC.32k_fs_LR.func.gii", False, "func"),
    ("anything", "x.shape.gii", False, "shape"),
    ("curv", "lh.curv", False, "shape"),
    ("myelin", "sub-01_myelin.dscalar.nii", False, "func"),
    ("corrThickness", None, False, "shape"),
    ("ArealDistortion_FS", None, False, "shape"),
    ("rest", "rest.dtseries.nii", True, "series"),
])
def test_metric_kind(name: str, source: str | None, series: bool, expected: str) -> None:
    kind, intent = metric_kind(name, source, series=series)
    assert {
        "shape": ("shape", "NIFTI_INTENT_SHAPE"),
        "func": ("func", "NIFTI_INTENT_NONE"),
        "series": ("func", "NIFTI_INTENT_TIME_SERIES"),
    }[expected] == (kind, intent)


@pytest.mark.parametrize(("args", "expected"), [
    (("left", "pial", "surf"), "hemi-L_pial.surf.gii"),
    (("right", "thickness", "shape", "sub-01"), "sub-01_hemi-R_thickness.shape.gii"),
    (("left", "aparc.a2009s", "label"), "hemi-L.aparc.a2009s.label.gii"),
    (("left", "very_inflated", "surf"), "hemi-L.very_inflated.surf.gii"),
])
def test_gifti_filename(args: tuple, expected: str) -> None:
    assert gifti_filename(*args) == expected


@pytest.mark.slow
def test_a_coarse_level_is_a_consistent_decimated_mesh(tmp_path: Path) -> None:
    from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

    store = tmp_path / "surf.zarrvectors"
    ingest_gifti(write_gifti_subject(tmp_path / "gii"), store, CHUNK)
    build_pyramid(str(store), factors=[(2.0, 1.0)], method="mesh",
                  cross_level_depth=0, cross_level_storage="none")
    out = tmp_path / "level1"
    summary = export_gifti(store, out, level=1)
    assert summary["level"] == 1

    for hemisphere, tag in (("left", "L"), ("right", "R")):
        coarse = read_hemisphere(store, hemisphere, level=1,
                                 attributes=["coords_pial", "thickness", "aparc"])
        n = len(coarse["vertices"])
        assert 0 < n < N
        mid, tris = nib.load(str(out / f"hemi-{tag}_midthickness.surf.gii")).darrays
        pial = nib.load(str(out / f"hemi-{tag}_pial.surf.gii")).darrays[0]
        np.testing.assert_array_equal(mid.data, coarse["vertices"].astype(np.float32))
        np.testing.assert_array_equal(pial.data, coarse["attributes"]["coords_pial"])
        assert _canonical(tris.data) == _canonical(coarse["faces"])
        assert tris.data.max() < n
        (thick,) = nib.load(str(out / f"hemi-{tag}_thickness.shape.gii")).darrays
        np.testing.assert_array_equal(thick.data, coarse["attributes"]["thickness"])
        (codes,) = nib.load(str(out / f"hemi-{tag}_aparc.label.gii")).darrays
        assert len(codes.data) == n

    # The coarse files are a complete subject in their own right.
    assert ingest_gifti(out, tmp_path / "again.zarrvectors", CHUNK)["vertex_count"] == (
        summary["vertex_count"]
    )


class TestCli:

    def test_an_extensionless_output_exports_gifti(self, tmp_path: Path) -> None:
        from zarr_vectors_tools.cli import main

        source = write_gifti_subject(tmp_path / "gii")
        store = tmp_path / "surf.zarrvectors"
        ingest_gifti(source, store, CHUNK)
        out = tmp_path / "exported"
        assert main([
            "convert", str(store), str(out),
            "--hemisphere", "lh", "--surface", "pial", "--attribute", "thickness",
            "--prefix", "sub-01",
        ]) == 0
        assert sorted(p.name for p in out.iterdir()) == [
            "sub-01_hemi-L_pial.surf.gii", "sub-01_hemi-L_thickness.shape.gii",
        ]

    def test_gifti_options_are_refused_elsewhere(self, tmp_path: Path) -> None:
        from zarr_vectors.types.points import write_points

        from zarr_vectors_tools.cli import main

        store = tmp_path / "p.zv"
        write_points(str(store), np.ones((3, 3), np.float32), chunk_shape=CHUNK)
        with pytest.raises(SystemExit, match="--prefix"):
            main(["convert", str(store), str(tmp_path / "p.csv"), "--prefix", "x"])
