"""An explicit, per-level RDP tolerance, in store units (T06).

The derived tolerance -- half the smallest edge of each level's bin -- is a
multiplier of a grid the user never thinks in.  What a user can say is "this
level may move a streamline by at most half a millimetre", so the pyramid
takes that directly:

* ``build_pyramid(..., rdp_tolerances=[...])`` and
  ``zvtools pyramid / convert --rdp-tolerance T1,T2,...`` set each level's
  Douglas-Peucker tolerance;
* every rdp level records the tolerance it was actually built with, explicit
  or derived, and a refresh rebuilds it at that tolerance;
* anywhere a tolerance would mean nothing -- decimate mode, and every store
  not coarsened by the polyline coarsener -- it is refused by name before a
  level is written, not ignored.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.building import (
    list_resolution_levels,
    open_store,
    read_level_metadata,
)
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.types.meshes import write_mesh
from zarr_vectors.types.points import write_points
from zarr_vectors.types.polylines import read_polylines

from tests._source_helpers import write_polylines_with_segment_id
from zarr_vectors_tools.cli import build_parser, main
from zarr_vectors_tools.multiresolution.coarsen import (
    TOOLS_LEVEL_ATTRS_KEY,
    build_pyramid,
    coarsen_level,
    read_coarsening_record,
)
from zarr_vectors_tools.multiresolution.refresh import rebuild_pyramid_from_level
from zarr_vectors_tools.multiresolution.strategies.polylines import (
    coarsen_polyline_level,
)

CHUNK = (10.0, 10.0, 10.0)
#: A one-unit bin, so a factor of 2 derives a tolerance of 1.0 -- far enough
#: from the explicit values below that "explicit was used" is observable.
BIN = (1.0, 1.0, 1.0)


def _wiggly(n: int = 6, points: int = 200) -> list[np.ndarray]:
    out = []
    for i in range(n):
        t = np.linspace(0.0, 1.0, points)
        out.append(np.column_stack([
            t * 28.0,
            5.0 + 3.0 * np.sin(t * 14.0 + i),
            5.0 + 2.0 * np.cos(t * 11.0 + i),
        ]).astype(np.float32))
    return out


def _streamline_store(path: Path) -> Path:
    write_polylines_with_segment_id(
        str(path), _wiggly(), chunk_shape=CHUNK, bin_shape=BIN,
        geometry_type="streamline",
    )
    return path


def _pyramid(store: Path, factors, **kwargs) -> dict:
    return build_pyramid(
        str(store), factors=factors, cross_level_depth=0,
        cross_level_storage="none", **kwargs,
    )


def _counts(store: Path) -> list[int]:
    root = open_store(str(store))
    return [
        int(read_level_metadata(root, lv).vertex_count)
        for lv in list_resolution_levels(root)
    ]


def _records(store: Path) -> dict[int, dict]:
    root = open_store(str(store))
    return {
        lv: read_coarsening_record(root, lv)
        for lv in list_resolution_levels(root) if lv > 0
    }


def _paths(store: Path, level: int) -> dict[int, np.ndarray]:
    """Each object's full path at ``level``, fragments joined in walk order."""
    result = read_polylines(str(store), level=level)
    return {
        int(oid): np.concatenate(fragments, axis=0).astype(np.float64)
        for oid, fragments in zip(result["object_ids"], result["polylines"])
        if len(fragments)
    }


def _max_deviation(points: np.ndarray, line: np.ndarray) -> float:
    """Largest distance from any of ``points`` to the polyline ``line``."""
    if len(line) == 1:
        return float(np.max(np.linalg.norm(points - line[0], axis=1)))
    a = line[:-1][None, :, :]
    ab = (line[1:] - line[:-1])[None, :, :]
    ap = points[:, None, :] - a
    length_sq = np.maximum(np.sum(ab * ab, axis=2), 1e-30)
    t = np.clip(np.sum(ap * ab, axis=2) / length_sq, 0.0, 1.0)
    nearest = a + t[:, :, None] * ab
    distances = np.linalg.norm(points[:, None, :] - nearest, axis=2)
    return float(distances.min(axis=1).max())


# =====================================================================
# The tolerance is honoured as a distance
# =====================================================================


class TestToleranceIsHonoured:

    def test_a_larger_tolerance_keeps_fewer_vertices(self, tmp_path: Path) -> None:
        tight = _streamline_store(tmp_path / "tight.zv")
        loose = _streamline_store(tmp_path / "loose.zv")
        _pyramid(tight, [(2.0, 1.0)], rdp_tolerances=[0.05])
        _pyramid(loose, [(2.0, 1.0)], rdp_tolerances=[0.5])

        level0, tight_count = _counts(tight)
        loose_count = _counts(loose)[1]
        assert loose_count < tight_count < level0

    def test_every_removed_vertex_stays_within_the_tolerance(
        self, tmp_path: Path,
    ) -> None:
        """The contract a user relies on: no vertex moves further than asked.

        The derived tolerance at this bin and factor is 1.0, so a level that
        silently kept deriving would deviate by up to five times what was
        requested and fail the bound.
        """
        tolerance = 0.2
        store = _streamline_store(tmp_path / "s.zv")
        _pyramid(store, [(2.0, 1.0)], rdp_tolerances=[tolerance])

        fine, coarse = _paths(store, 0), _paths(store, 1)
        assert set(coarse) == set(fine)
        deviations = [_max_deviation(fine[oid], coarse[oid]) for oid in fine]
        # float32 storage: allow for its rounding, nothing more.
        assert max(deviations) <= tolerance + 1e-4, deviations
        # And it was the tolerance that stopped the simplifier, not a floor:
        # the kept lines do use the room they were given.
        assert max(deviations) > tolerance / 4, deviations
        assert sum(len(p) for p in coarse.values()) < sum(
            len(p) for p in fine.values()
        )

    def test_deviation_from_level_zero_is_bounded_by_the_running_sum(
        self, tmp_path: Path,
    ) -> None:
        """Each tolerance is relative to the level below, as documented."""
        tolerances = [0.2, 0.3]
        store = _streamline_store(tmp_path / "s.zv")
        _pyramid(store, [(2.0, 1.0), (2.0, 1.0)], rdp_tolerances=tolerances)

        fine, middle, coarse = (_paths(store, lv) for lv in (0, 1, 2))
        assert max(
            _max_deviation(middle[oid], coarse[oid]) for oid in coarse
        ) <= tolerances[1] + 1e-4
        assert max(
            _max_deviation(fine[oid], coarse[oid]) for oid in coarse
        ) <= sum(tolerances) + 1e-4

    def test_the_explicit_tolerance_replaces_the_derived_one(
        self, tmp_path: Path,
    ) -> None:
        derived = _streamline_store(tmp_path / "derived.zv")
        explicit = _streamline_store(tmp_path / "explicit.zv")
        _pyramid(derived, [(2.0, 1.0), (2.0, 1.0)])
        summary = _pyramid(
            explicit, [(2.0, 1.0), (2.0, 1.0)], rdp_tolerances=[0.05, 0.1],
        )
        assert [s["simplify_epsilon"] for s in summary["level_specs"]] == [0.05, 0.1]
        assert _counts(explicit)[1] > _counts(derived)[1]
        # The factors still set the bins, whichever tolerance simplified.
        for lv in (1, 2):
            assert (
                read_level_metadata(open_store(str(explicit)), lv).bin_shape
                == read_level_metadata(open_store(str(derived)), lv).bin_shape
            )

    def test_a_tolerance_simplifies_even_at_factor_one(
        self, tmp_path: Path,
    ) -> None:
        """A factor of 1 is "keep the bin", not "keep every vertex", once a
        distance has been asked for."""
        store = _streamline_store(tmp_path / "s.zv")
        _pyramid(store, [(1.0, 1.0)], rdp_tolerances=[0.3])
        level0, level1 = _counts(store)
        assert level1 < level0
        assert read_level_metadata(open_store(str(store)), 1).bin_shape == BIN


# =====================================================================
# The tolerance is recorded on the level
# =====================================================================


class TestToleranceIsRecorded:

    def test_explicit_tolerances_are_recorded_per_level(
        self, tmp_path: Path,
    ) -> None:
        store = _streamline_store(tmp_path / "s.zv")
        summary = _pyramid(
            store, [(2.0, 1.0), (2.0, 1.0), (2.0, 1.0)],
            rdp_tolerances=[0.5, 1, 2],
        )
        assert _records(store) == {
            1: {"rdp_tolerance": 0.5, "rdp_tolerance_source": "explicit"},
            2: {"rdp_tolerance": 1.0, "rdp_tolerance_source": "explicit"},
            3: {"rdp_tolerance": 2.0, "rdp_tolerance_source": "explicit"},
        }
        # Writing the record must not disturb core's own level block: the
        # coarsener's level handle predates the final vertex count, and
        # writing through it would put the placeholder 0 back.
        assert _counts(store)[1:] == [
            s["vertex_count"] for s in summary["level_specs"]
        ]
        assert all(count > 0 for count in _counts(store))

    def test_derived_tolerances_are_recorded_too(self, tmp_path: Path) -> None:
        store = _streamline_store(tmp_path / "s.zv")
        _pyramid(store, [(2.0, 1.0), (2.0, 1.0)])
        assert _records(store) == {
            1: {"rdp_tolerance": 1.0, "rdp_tolerance_source": "derived"},
            2: {"rdp_tolerance": 2.0, "rdp_tolerance_source": "derived"},
        }

    def test_an_identity_level_records_no_tolerance(self, tmp_path: Path) -> None:
        store = _streamline_store(tmp_path / "s.zv")
        _pyramid(store, [(1.0, 1.0)])
        assert _records(store) == {
            1: {"rdp_tolerance": None, "rdp_tolerance_source": "derived"},
        }

    def test_a_decimated_level_carries_no_tolerance_record(
        self, tmp_path: Path,
    ) -> None:
        store = _streamline_store(tmp_path / "s.zv")
        _pyramid(store, [(4.0, 1.0)], coarsen_mode="decimate")
        assert _records(store) == {1: {}}

    def test_the_record_lives_beside_core_level_metadata(
        self, tmp_path: Path,
    ) -> None:
        """A key inside ``zarr_vectors_level`` would be dropped the next time
        core restamps that block; a sibling key is not."""
        store = _streamline_store(tmp_path / "s.zv")
        _pyramid(store, [(2.0, 1.0)], rdp_tolerances=[0.5])
        attrs = open_store(str(store))["1"].attrs.to_dict()
        assert "rdp_tolerance" not in attrs["zarr_vectors_level"]
        assert attrs[TOOLS_LEVEL_ATTRS_KEY]["coarsening"]["rdp_tolerance"] == 0.5


# =====================================================================
# Refresh reproduces the tolerance
# =====================================================================


class TestRefreshReproducesTheTolerance:

    def test_an_explicit_tolerance_survives_a_refresh(
        self, tmp_path: Path,
    ) -> None:
        store = _streamline_store(tmp_path / "s.zv")
        _pyramid(store, [(2.0, 1.0), (2.0, 1.0)], rdp_tolerances=[0.05, 0.1])
        before_counts, before_records = _counts(store), _records(store)

        # Guard the guard: were the refresh to fall back to the derived
        # tolerance, it would visibly produce a different level.
        derived = _streamline_store(tmp_path / "derived.zv")
        _pyramid(derived, [(2.0, 1.0), (2.0, 1.0)])
        assert _counts(derived)[1] != before_counts[1]

        rebuild_pyramid_from_level(open_store(str(store), mode="r+"), 0)

        assert _counts(store) == before_counts
        assert _records(store) == before_records

    def test_a_derived_tolerance_stays_derived(self, tmp_path: Path) -> None:
        store = _streamline_store(tmp_path / "s.zv")
        _pyramid(store, [(2.0, 1.0), (2.0, 1.0)])
        before_counts, before_records = _counts(store), _records(store)

        rebuild_pyramid_from_level(open_store(str(store), mode="r+"), 0)

        assert _counts(store) == before_counts
        assert _records(store) == before_records


# =====================================================================
# Refused where it cannot apply
# =====================================================================


def _points_store(path: Path) -> Path:
    rng = np.random.default_rng(0)
    write_points(
        str(path), (rng.random((100, 3)) * 20).astype(np.float32),
        chunk_shape=CHUNK,
    )
    return path


def _mesh_store(path: Path) -> Path:
    vertices = np.array(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int64)
    write_mesh(str(path), vertices, faces, chunk_shape=CHUNK)
    return path


def _graph_store(path: Path) -> Path:
    positions = np.array(
        [[1.0, 1, 1], [2, 2, 2], [12, 2, 2], [15, 5, 5]], dtype=np.float32,
    )
    write_graph(
        str(path), positions, np.array([[0, 1], [1, 2], [2, 3]]),
        chunk_shape=CHUNK,
    )
    return path


def _skeleton_store(path: Path) -> Path:
    pytest.importorskip("zarr_vectors.types.skeletons")
    from tests.test_skeleton_coarsen import _write_two_skeletons

    _write_two_skeletons(str(path))
    return path


class TestRefusals:

    @pytest.mark.parametrize(
        ("make", "coarsener"),
        [
            (_points_store, "per_object"),
            (_mesh_store, "mesh"),
            (_graph_store, "per_object"),
            (_skeleton_store, "skeleton"),
        ],
        ids=["points", "mesh", "graph", "skeleton"],
    )
    def test_stores_without_the_rdp_coarsener_are_refused(
        self, tmp_path: Path, make, coarsener: str,
    ) -> None:
        store = make(tmp_path / "s.zv")
        with pytest.raises(ValueError, match=rf"rdp_tolerances.*'{coarsener}'"):
            _pyramid(store, [(2.0, 1.0)], rdp_tolerances=[0.5])
        assert list_resolution_levels(open_store(str(store))) == [0]

    def test_a_strategy_override_is_checked_not_the_geometry(
        self, tmp_path: Path,
    ) -> None:
        """A streamline store forced onto the per-fragment strategy has no
        tolerance either: that strategy reads ``rdp`` as a stride."""
        store = _streamline_store(tmp_path / "s.zv")
        with pytest.raises(ValueError, match="'per_fragment'"):
            _pyramid(
                store, [(2.0, 1.0)], rdp_tolerances=[0.5], method="per_fragment",
            )
        assert list_resolution_levels(open_store(str(store))) == [0]

    def test_decimate_mode_is_refused(self, tmp_path: Path) -> None:
        store = _streamline_store(tmp_path / "s.zv")
        with pytest.raises(ValueError, match="rdp_tolerances.*decimate"):
            _pyramid(
                store, [(2.0, 1.0)], coarsen_mode="decimate", rdp_tolerances=[0.5],
            )
        assert list_resolution_levels(open_store(str(store))) == [0]

    def test_a_length_mismatch_is_refused_before_any_level(
        self, tmp_path: Path,
    ) -> None:
        store = _streamline_store(tmp_path / "s.zv")
        with pytest.raises(ValueError, match="2 entries for 3 coarser"):
            _pyramid(
                store, [(2.0, 1.0), (2.0, 1.0), (2.0, 1.0)],
                rdp_tolerances=[0.5, 1.0],
            )
        assert list_resolution_levels(open_store(str(store))) == [0]

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, "wide"])
    def test_a_tolerance_that_is_no_distance_is_refused(
        self, tmp_path: Path, bad,
    ) -> None:
        store = _streamline_store(tmp_path / "s.zv")
        with pytest.raises(ValueError, match=r"rdp_tolerances\[1\]"):
            _pyramid(
                store, [(2.0, 1.0), (2.0, 1.0)], rdp_tolerances=[0.5, bad],
            )
        assert list_resolution_levels(open_store(str(store))) == [0]

    def test_a_bare_number_is_refused(self, tmp_path: Path) -> None:
        store = _streamline_store(tmp_path / "s.zv")
        with pytest.raises(ValueError, match="one tolerance per coarser level"):
            _pyramid(store, [(2.0, 1.0)], rdp_tolerances=0.5)

    def test_coarsen_level_refuses_it_on_a_points_store(
        self, tmp_path: Path,
    ) -> None:
        store = _points_store(tmp_path / "p.zv")
        with pytest.raises(ValueError, match="rdp_tolerance does not apply"):
            coarsen_level(
                str(store), 0, 1, coarsen_factor=2.0, rdp_tolerance=0.5,
            )
        assert list_resolution_levels(open_store(str(store))) == [0]

    def test_coarsen_level_applies_it_on_a_streamline_store(
        self, tmp_path: Path,
    ) -> None:
        store = _streamline_store(tmp_path / "s.zv")
        summary = coarsen_level(
            str(store), 0, 1, coarsen_factor=2.0, rdp_tolerance=0.25,
        )
        assert summary["simplify_epsilon"] == 0.25
        assert summary["rdp_tolerance_source"] == "explicit"
        assert _records(store)[1]["rdp_tolerance"] == 0.25

    def test_the_strategy_refuses_an_epsilon_in_decimate_mode(
        self, tmp_path: Path,
    ) -> None:
        """It used to ignore one without a word."""
        store = _streamline_store(tmp_path / "s.zv")
        with pytest.raises(ValueError, match="simplify_epsilon.*decimate"):
            coarsen_polyline_level(
                str(store), 0, 1, coarsen_factor=2.0, coarsen_mode="decimate",
                simplify_epsilon=0.5,
            )
        assert list_resolution_levels(open_store(str(store))) == [0]


# =====================================================================
# TRK inline pyramid
# =====================================================================


class TestTrkInlinePyramid:
    """``ingest_trk_parallel`` builds its pyramid after the whole ingest, so
    a list that cannot apply must fail before the file is even opened --
    these pass a path that does not exist to prove it."""

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"pyramid_rdp_tolerances": [0.5, 1.0],
              "pyramid_coarsen_mode": "decimate"}, "decimate"),
            ({"pyramid_rdp_tolerances": [0.5, 1.0],
              "build_multiscale": False}, "build_multiscale"),
            # The default pyramid is two levels.
            ({"pyramid_rdp_tolerances": [0.5]}, "1 entries for 2 coarser"),
            ({"pyramid_rdp_tolerances": [0.5], "pyramid_factors": [(2.0, 1.0)],
              }, None),
        ],
        ids=["decimate", "no-pyramid", "length", "valid-reaches-the-file"],
    )
    def test_refused_before_the_input_is_read(
        self, tmp_path: Path, kwargs: dict, match: str | None,
    ) -> None:
        from zarr_vectors_tools.convert.ingest.trk_parallel import ingest_trk_parallel

        missing = tmp_path / "missing.trk"
        if match is None:
            # A valid list gets past the check and fails on the file instead.
            with pytest.raises((FileNotFoundError, OSError)):
                ingest_trk_parallel(
                    missing, tmp_path / "o.zv", progress=False, **kwargs,
                )
        else:
            with pytest.raises(ValueError, match=match):
                ingest_trk_parallel(
                    missing, tmp_path / "o.zv", progress=False, **kwargs,
                )
        assert not (tmp_path / "o.zv").exists()


# =====================================================================
# CLI
# =====================================================================


def _help_for(subcommand: str, dest: str) -> str:
    parser = build_parser()
    sub = next(
        a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"
    )
    return next(
        a.help for a in sub.choices[subcommand]._actions if a.dest == dest
    )


class TestCli:

    @pytest.mark.parametrize("subcommand", ["pyramid", "convert"])
    def test_the_flag_help_is_ascii(self, subcommand: str) -> None:
        # A non-ASCII help string crashes argparse on a Windows console.
        assert _help_for(subcommand, "rdp_tolerance").isascii()

    def test_pyramid_sets_and_records_each_level(self, tmp_path: Path) -> None:
        store = _streamline_store(tmp_path / "s.zv")
        rc = main([
            "pyramid", str(store), "--coarsen", "2,2,2", "--sparsity", "1,1,1",
            "--rdp-tolerance", "0.5,1,2",
        ])
        assert rc == 0
        assert {lv: r["rdp_tolerance"] for lv, r in _records(store).items()} == {
            1: 0.5, 2: 1.0, 3: 2.0,
        }
        assert {r["rdp_tolerance_source"] for r in _records(store).values()} == {
            "explicit",
        }

    @pytest.mark.parametrize(
        ("extra", "match"),
        [
            (["--rdp-tolerance", "0.5"], "1 entries but --coarsen has 2"),
            (["--rdp-tolerance", "0.5,0"], "distances > 0"),
            (["--rdp-tolerance", "0.5,nan"], "distances > 0"),
            (["--rdp-tolerance", "0.5,1", "--coarsen-mode", "decimate"],
             "--coarsen-mode decimate"),
        ],
        ids=["length", "zero", "nan", "decimate"],
    )
    def test_pyramid_refuses_a_bad_list(
        self, tmp_path: Path, extra: list[str], match: str,
    ) -> None:
        store = _streamline_store(tmp_path / "s.zv")
        with pytest.raises(SystemExit, match=match):
            main([
                "pyramid", str(store), "--coarsen", "2,2", "--sparsity", "1,1",
                *extra,
            ])
        assert list_resolution_levels(open_store(str(store))) == [0]

    def test_pyramid_refuses_a_store_without_the_rdp_coarsener(
        self, tmp_path: Path,
    ) -> None:
        store = _points_store(tmp_path / "p.zv")
        with pytest.raises(SystemExit, match="--rdp-tolerance.*'per_object'"):
            main([
                "pyramid", str(store), "--coarsen", "2", "--sparsity", "1",
                "--rdp-tolerance", "0.5",
            ])
        assert list_resolution_levels(open_store(str(store))) == [0]

    def test_convert_tck_sets_the_tolerance(self, tmp_path: Path) -> None:
        pytest.importorskip("nibabel")
        from tests.test_streamline_ingest_pyramid import _write_tck

        source = _write_tck(tmp_path / "t.tck", _wiggly(n=3, points=60))
        out = tmp_path / "t.zv"
        rc = main([
            "convert", str(source), str(out), "--chunk-shape", "10,10,10",
            "--coarsen", "2,2", "--sparsity", "1,1", "--rdp-tolerance", "0.25,0.75",
        ])
        assert rc == 0
        assert _records(out) == {
            1: {"rdp_tolerance": 0.25, "rdp_tolerance_source": "explicit"},
            2: {"rdp_tolerance": 0.75, "rdp_tolerance_source": "explicit"},
        }

    def test_convert_refuses_a_format_without_the_rdp_coarsener(
        self, tmp_path: Path,
    ) -> None:
        source = tmp_path / "pts.csv"
        source.write_text("x,y,z\n1,2,3\n4,5,6\n")
        out = tmp_path / "pts.zv"
        with pytest.raises(SystemExit, match="--rdp-tolerance applies to"):
            main([
                "convert", str(source), str(out), "--chunk-shape", "10,10,10",
                "--coarsen", "2", "--sparsity", "1", "--rdp-tolerance", "0.5",
            ])
        assert not out.exists()

    def test_convert_refuses_it_without_a_pyramid(self, tmp_path: Path) -> None:
        pytest.importorskip("nibabel")
        from tests.test_streamline_ingest_pyramid import _write_tck

        source = _write_tck(tmp_path / "t.tck", _wiggly(n=2, points=20))
        out = tmp_path / "t.zv"
        with pytest.raises(SystemExit, match="no pyramid was requested"):
            main([
                "convert", str(source), str(out), "--chunk-shape", "10,10,10",
                "--rdp-tolerance", "0.5",
            ])
        assert not out.exists()

    def test_convert_refuses_a_mismatch_before_ingesting(
        self, tmp_path: Path,
    ) -> None:
        pytest.importorskip("nibabel")
        from tests.test_streamline_ingest_pyramid import _write_tck

        source = _write_tck(tmp_path / "t.tck", _wiggly(n=2, points=20))
        out = tmp_path / "t.zv"
        with pytest.raises(SystemExit, match="2 entries but --coarsen has 1"):
            main([
                "convert", str(source), str(out), "--chunk-shape", "10,10,10",
                "--coarsen", "2", "--sparsity", "1", "--rdp-tolerance", "0.5,1",
            ])
        assert not out.exists()

    def test_export_refuses_it(self, tmp_path: Path) -> None:
        store = _streamline_store(tmp_path / "s.zv")
        with pytest.raises(SystemExit, match="--rdp-tolerance"):
            main([
                "convert", str(store), str(tmp_path / "out.trk"),
                "--rdp-tolerance", "0.5",
            ])

    # A full TRK ingest, so in the slow tier with the rest of them.
    @pytest.mark.slow
    def test_convert_trk_threads_it_to_the_inline_pyramid(
        self, tmp_path: Path,
    ) -> None:
        pytest.importorskip("nibabel")
        from tests.test_cli import _write_smooth_trk

        source = _write_smooth_trk(tmp_path / "s.trk", n=40, npts=40)
        out = tmp_path / "s.zv"
        rc = main([
            "convert", str(source), str(out), "--num-chunks", "27",
            "--workers", "1", "--coarsen", "2,2", "--sparsity", "1,1",
            "--rdp-tolerance", "0.5,2",
        ])
        assert rc == 0
        assert _records(out) == {
            1: {"rdp_tolerance": 0.5, "rdp_tolerance_source": "explicit"},
            2: {"rdp_tolerance": 2.0, "rdp_tolerance_source": "explicit"},
        }
