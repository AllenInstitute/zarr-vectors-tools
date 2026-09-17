"""``zvtools convert`` moves data both ways.

The command used to be ingest-only, and the package's seven exporters were
reachable from Python alone -- so a pipeline driven from the shell had to
drop out to a script for its last step.  The direction now comes from what
``INPUT`` is: a file is read into the store at ``OUTPUT``, a store is written
out to the file at ``OUTPUT``.

What is worth testing here is the seam, not the exporters (which have their
own tests): that the direction is detected, that the format is resolved from
the right side of the command, and that an option which does not apply to the
chosen direction or format is refused by name instead of ignored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zarr_vectors.types.points import write_points
from zarr_vectors.types.polylines import write_polylines

from zarr_vectors_tools.cli import main

CHUNK = (10.0, 10.0, 10.0)


@pytest.fixture
def point_store(tmp_path: Path) -> Path:
    store = tmp_path / "points.zarrvectors"
    positions = np.column_stack([
        np.linspace(0.0, 9.0, 20),
        np.linspace(0.0, 4.0, 20),
        np.full(20, 1.0),
    ]).astype(np.float32)
    write_points(
        str(store), positions, chunk_shape=(5.0, 5.0, 5.0),
        vertex_attributes={"intensity": np.arange(20, dtype=np.float32)},
    )
    return store


@pytest.fixture
def streamline_store(tmp_path: Path) -> tuple[Path, list[np.ndarray]]:
    store = tmp_path / "tracts.zarrvectors"
    polylines = [
        np.column_stack([
            np.linspace(1.0, 9.0, 6), np.full(6, 2.0), np.full(6, 3.0),
        ]).astype(np.float32),
        np.column_stack([
            np.linspace(2.0, 8.0, 4), np.full(4, 4.0), np.full(4, 5.0),
        ]).astype(np.float32),
    ]
    write_polylines(
        str(store), polylines, chunk_shape=CHUNK, geometry_type="streamline",
    )
    return store, polylines


class TestDirection:

    def test_a_file_in_is_still_an_ingest(self, tmp_path: Path) -> None:
        source = tmp_path / "cells.csv"
        source.write_text(
            "x,y,z\n" + "\n".join(f"{i},{i},{i}" for i in range(10)) + "\n",
        )
        store = tmp_path / "cells.zarrvectors"
        assert main([
            "convert", str(source), str(store), "--chunk-shape", "5,5,5",
        ]) == 0
        assert (store / "zarr.json").exists()

    def test_a_store_in_is_an_export(
        self, point_store: Path, tmp_path: Path,
    ) -> None:
        out = tmp_path / "out.csv"
        assert main(["convert", str(point_store), str(out)]) == 0
        assert out.exists()
        assert out.read_text().splitlines()[0].startswith("dim0,dim1,dim2")

    def test_the_round_trip_preserves_the_geometry(
        self, streamline_store: tuple[Path, list[np.ndarray]], tmp_path: Path,
    ) -> None:
        pytest.importorskip("nibabel")
        import nibabel as nib

        store, polylines = streamline_store
        out = tmp_path / "out.trk"
        assert main(["convert", str(store), str(out)]) == 0

        loaded = [np.asarray(s) for s in nib.streamlines.load(str(out)).streamlines]
        assert len(loaded) == len(polylines)
        for written, read_back in zip(polylines, loaded):
            np.testing.assert_allclose(written, read_back, atol=1e-3)


class TestFormatResolution:

    def test_the_output_extension_names_the_format(
        self, point_store: Path, tmp_path: Path,
    ) -> None:
        """On export it is the OUTPUT that says what to write, not the input."""
        pytest.importorskip("plyfile")
        out = tmp_path / "out.ply"
        assert main(["convert", str(point_store), str(out)]) == 0
        assert out.read_bytes().startswith(b"ply")

    def test_an_explicit_format_overrides_the_extension(
        self, point_store: Path, tmp_path: Path,
    ) -> None:
        out = tmp_path / "out.dat"
        assert main([
            "convert", str(point_store), str(out), "--format", "csv",
        ]) == 0
        assert out.read_text().splitlines()[0].startswith("dim0")

    def test_an_unknown_extension_says_what_is_available(
        self, point_store: Path, tmp_path: Path,
    ) -> None:
        with pytest.raises(SystemExit, match="cannot tell what to export"):
            main(["convert", str(point_store), str(tmp_path / "out.bogus")])

    def test_a_format_with_no_exporter_is_named(
        self, point_store: Path, tmp_path: Path,
    ) -> None:
        """``las`` can be ingested but not written."""
        with pytest.raises(SystemExit, match="cannot export to 'las'"):
            main([
                "convert", str(point_store), str(tmp_path / "out.las"),
                "--format", "las",
            ])


class TestOptionsAreRefusedNotIgnored:

    def test_an_ingest_flag_on_an_export(
        self, point_store: Path, tmp_path: Path,
    ) -> None:
        """Silently ignoring it would leave the user believing it applied."""
        with pytest.raises(SystemExit, match="do not apply when exporting"):
            main([
                "convert", str(point_store), str(tmp_path / "o.csv"),
                "--chunk-shape", "5,5,5",
            ])

    def test_pyramid_flags_point_at_the_level_flag(
        self, point_store: Path, tmp_path: Path,
    ) -> None:
        with pytest.raises(SystemExit, match=r"--level"):
            main([
                "convert", str(point_store), str(tmp_path / "o.csv"),
                "--coarsen", "8", "--sparsity", "1",
            ])

    def test_an_option_the_exporter_does_not_take(
        self, tmp_path: Path,
    ) -> None:
        """SWC takes no per-vertex attribute selection; say so."""
        from zarr_vectors.types.graphs import write_graph

        store = tmp_path / "skel.zarrvectors"
        positions = np.array(
            [[1.0, 1, 1], [2, 1, 1], [3, 1, 1]], dtype=np.float32,
        )
        write_graph(
            str(store), positions,
            np.array([[1, 0], [2, 1]], dtype=np.int64),
            chunk_shape=CHUNK, kind="skeleton",
        )
        with pytest.raises(SystemExit, match="--attribute does not apply"):
            main([
                "convert", str(store), str(tmp_path / "o.swc"),
                "--attribute", "radius",
            ])


class TestExportOptions:

    def test_named_attributes_reach_the_file(
        self, point_store: Path, tmp_path: Path,
    ) -> None:
        out = tmp_path / "out.csv"
        assert main([
            "convert", str(point_store), str(out),
            "--attribute", "intensity",
        ]) == 0
        header = out.read_text().splitlines()[0]
        assert header == "dim0,dim1,dim2,intensity"

    def test_a_bounding_box_is_read_as_two_corners(
        self, point_store: Path, tmp_path: Path,
    ) -> None:
        """The flag is flat, the way a box is usually written; the exporter
        takes a (min, max) pair."""
        whole = tmp_path / "whole.csv"
        cropped = tmp_path / "cropped.csv"
        assert main(["convert", str(point_store), str(whole)]) == 0
        assert main([
            "convert", str(point_store), str(cropped),
            "--bbox", "0,0,0,4,4,4",
        ]) == 0
        assert len(cropped.read_text().splitlines()) < len(
            whole.read_text().splitlines()
        )

    def test_an_odd_bounding_box_is_refused(
        self, point_store: Path, tmp_path: Path,
    ) -> None:
        with pytest.raises(SystemExit, match="even number of values"):
            main([
                "convert", str(point_store), str(tmp_path / "o.csv"),
                "--bbox", "0,0,0,4,4",
            ])

    def test_a_coarse_level_can_be_exported(
        self, tmp_path: Path,
    ) -> None:
        """The point of a pyramid is that the coarse levels are usable."""
        from zarr_vectors_tools.multiresolution.coarsen import build_pyramid

        store = tmp_path / "pyr.zarrvectors"
        rng = np.random.default_rng(0)
        write_points(
            str(store), (rng.random((200, 3)) * 20).astype(np.float32),
            chunk_shape=CHUNK, bin_shape=(1.0, 1.0, 1.0),
            object_ids=np.repeat(np.arange(20), 10),
        )
        build_pyramid(
            str(store), factors=[(4.0, 1.0)],
            cross_level_depth=0, cross_level_storage="none",
        )
        level0 = tmp_path / "l0.csv"
        level1 = tmp_path / "l1.csv"
        assert main(["convert", str(store), str(level0), "--level", "0"]) == 0
        assert main(["convert", str(store), str(level1), "--level", "1"]) == 0
        assert len(level1.read_text().splitlines()) < len(
            level0.read_text().splitlines()
        )


class TestHelpIsPrintable:

    def test_convert_help_does_not_crash_on_a_windows_console(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """argparse writes help to the console encoding.

        On Windows that is cp1252, where a single non-ASCII character in a
        help string raises UnicodeEncodeError and the user gets a traceback
        instead of the help.  Several help strings carried arrows and em
        dashes, so ``zvtools convert --help`` was unusable there.
        """
        with pytest.raises(SystemExit) as excinfo:
            main(["convert", "--help"])
        assert excinfo.value.code == 0
        printed = capsys.readouterr().out
        assert printed
        printed.encode("cp1252")  # raises if a stray character came back
