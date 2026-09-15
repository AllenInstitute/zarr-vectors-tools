"""The numpy STL and OBJ parsers against the per-line parsers they replaced.

The ``_old_*`` functions below are frozen copies of the parsers as they
stood before vectorisation (commit 6c199de).  They are the reference: every
test builds a file, runs both, and requires bit-identical arrays -- same
dtype, shape, byte content and ordering -- or the same exception and
message.  They are deliberately not imported from the package, whose own
line-by-line fallback could otherwise drift along with the code it checks.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from zarr_vectors.exceptions import IngestError

import zarr_vectors_tools.headers.registry as registry_module
from zarr_vectors_tools.convert.ingest import _text_tokens
from zarr_vectors_tools.convert.ingest import obj as obj_module
from zarr_vectors_tools.convert.ingest import stl as stl_module

# ---------------------------------------------------------------------------
# Frozen reference parsers (verbatim logic from commit 6c199de)
# ---------------------------------------------------------------------------


def _old_parse_ascii_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    normals: list[list[float]] = []

    with open(path) as f:
        current_normal: list[float] = [0, 0, 0]
        for line in f:
            line = line.strip()
            if line.startswith("facet normal"):
                parts = line.split()
                current_normal = [float(parts[2]), float(parts[3]), float(parts[4])]
                normals.append(current_normal)
            elif line.startswith("vertex"):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return (
        np.array(vertices, dtype=np.float64) if vertices else np.zeros((0, 3)),
        np.array(normals, dtype=np.float64) if normals else np.zeros((0, 3)),
    )


def _old_parse_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        f.read(80)  # header
        n_faces = struct.unpack("<I", f.read(4))[0]

        vertices = np.empty((n_faces * 3, 3), dtype=np.float32)
        normals = np.empty((n_faces, 3), dtype=np.float32)

        for i in range(n_faces):
            nx, ny, nz = struct.unpack("<3f", f.read(12))
            normals[i] = [nx, ny, nz]
            for j in range(3):
                x, y, z = struct.unpack("<3f", f.read(12))
                vertices[i * 3 + j] = [x, y, z]
            f.read(2)  # attribute byte count

    return vertices, normals


def _old_stl_write_args(
    input_path: Path, *, dtype: str = "float32", merge_vertices: bool = True,
    merge_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """The old ``ingest_stl`` up to the ``write_mesh`` call."""
    try:
        if stl_module._is_ascii_stl(input_path):
            raw_verts, raw_normals = _old_parse_ascii_stl(input_path)
        else:
            raw_verts, raw_normals = _old_parse_binary_stl(input_path)
    except IngestError:
        raise
    except Exception as e:
        raise IngestError(f"Failed to parse STL '{input_path}': {e}") from e

    if len(raw_verts) == 0:
        raise IngestError(f"STL file has no triangles: {input_path}")

    np_dtype = np.dtype(dtype)
    n_raw = len(raw_verts)
    n_faces = n_raw // 3
    if merge_vertices:
        positions, faces = stl_module._merge_vertices(raw_verts, n_faces, merge_tolerance)
    else:
        positions = raw_verts
        faces = np.arange(n_raw, dtype=np.int64).reshape(n_faces, 3)
    positions = positions.astype(np_dtype)
    return {"positions": positions, "faces": faces, "kwargs": {}}


def _old_obj_write_args(
    input_path: Path, *, dtype: str = "float32", auto_object_id: bool = False,
) -> dict[str, Any]:
    """The old ``ingest_obj`` up to the ``write_mesh`` call."""
    vertices: list[list[float]] = []
    normals_list: list[list[float]] = []
    faces: list[list[int]] = []
    vertex_object_ids: list[int] = []
    object_names: list[str] = []
    current_obj_id = -1

    try:
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts[0] == "v" and len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    if auto_object_id:
                        vertex_object_ids.append(max(current_obj_id, 0))
                elif parts[0] in ("o", "g") and len(parts) >= 2 and auto_object_id:
                    name = " ".join(parts[1:])
                    if name not in object_names:
                        object_names.append(name)
                    current_obj_id = object_names.index(name)
                elif parts[0] == "vn" and len(parts) >= 4:
                    normals_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == "f" and len(parts) >= 4:
                    face_indices: list[int] = []
                    for p in parts[1:]:
                        idx_str = p.split("/")[0]
                        idx = int(idx_str)
                        if idx < 0:
                            idx = len(vertices) + idx
                        else:
                            idx = idx - 1
                        face_indices.append(idx)

                    if len(face_indices) == 3:
                        faces.append(face_indices)
                    elif len(face_indices) == 4:
                        faces.append(face_indices)
                    elif len(face_indices) > 4:
                        for i in range(1, len(face_indices) - 1):
                            faces.append([
                                face_indices[0],
                                face_indices[i],
                                face_indices[i + 1],
                            ])

    except IngestError:
        raise
    except Exception as e:
        raise IngestError(f"Failed to parse OBJ '{input_path}': {e}") from e

    if not vertices:
        raise IngestError(f"OBJ file has no vertices: {input_path}")

    np_dtype = np.dtype(dtype)
    positions = np.array(vertices, dtype=np_dtype)

    if not faces:
        raise IngestError(f"OBJ file has no faces: {input_path}")

    face_sizes = set(len(f) for f in faces)
    if face_sizes not in ({3}, {4}):
        tri_faces: list[list[int]] = []
        for f in faces:
            if len(f) == 3:
                tri_faces.append(f)
            elif len(f) == 4:
                tri_faces.append([f[0], f[1], f[2]])
                tri_faces.append([f[0], f[2], f[3]])
        faces = tri_faces

    faces_arr = np.array(faces, dtype=np.int64)

    vertex_attributes: dict[str, np.ndarray] | None = None
    if normals_list and len(normals_list) == len(vertices):
        vertex_attributes = {"normal": np.array(normals_list, dtype=np.float32)}

    kwargs: dict[str, Any] = {"vertex_attributes": vertex_attributes}
    if auto_object_id and vertex_object_ids:
        kwargs["object_ids"] = np.array(vertex_object_ids, dtype=np.int64)
    out: dict[str, Any] = {"positions": positions, "faces": faces_arr, "kwargs": kwargs}
    if auto_object_id and object_names:
        out["object_names"] = object_names
    return out


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _assert_identical(new: np.ndarray, old: np.ndarray) -> None:
    assert isinstance(new, np.ndarray)
    assert new.dtype == old.dtype
    assert new.shape == old.shape
    # Compared as bytes so NaN payloads and the sign of zero count too.
    assert np.ascontiguousarray(new).tobytes() == np.ascontiguousarray(old).tobytes()


def _capture_ingest(monkeypatch: pytest.MonkeyPatch, module: Any, call: Any) -> dict[str, Any]:
    """Run an ingest with ``write_mesh`` and the header registry stubbed out."""
    captured: dict[str, Any] = {}

    def fake_write_mesh(store: str, positions: np.ndarray, faces: np.ndarray, **kwargs: Any):
        captured.update(positions=positions, faces=faces, kwargs=kwargs)
        return {}

    class FakeRegistry:
        def __init__(self, path: str) -> None:
            pass

        def add(self, key: str, header: Any) -> None:
            captured["object_names"] = header.object_names

    monkeypatch.setattr(module, "write_mesh", fake_write_mesh)
    monkeypatch.setattr(registry_module, "HeaderRegistry", FakeRegistry)
    call()
    return captured


def _assert_same_outcome(new_call: Any, old_call: Any) -> dict[str, Any] | None:
    """Both calls raise the same error, or both return; returns the new result."""
    try:
        old = old_call()
    except Exception as old_error:
        with pytest.raises(type(old_error)) as new_error:
            new_call()
        assert str(new_error.value) == str(old_error)
        return None
    new = new_call()
    for key in ("positions", "faces"):
        _assert_identical(new[key], old[key])
    assert new.get("object_names") == old.get("object_names")
    for key, old_value in old["kwargs"].items():
        new_value = new["kwargs"].get(key)
        if isinstance(old_value, dict):
            assert new_value is not None and new_value.keys() == old_value.keys()
            for name in old_value:
                _assert_identical(new_value[name], old_value[name])
        elif isinstance(old_value, np.ndarray):
            _assert_identical(new_value, old_value)
        else:
            assert new_value == old_value
    assert ("object_ids" in new["kwargs"]) == ("object_ids" in old["kwargs"])
    return new


def _compare_obj(
    monkeypatch: pytest.MonkeyPatch, path: Path, **kwargs: Any,
) -> dict[str, Any] | None:
    def new() -> dict[str, Any]:
        return _capture_ingest(
            monkeypatch, obj_module,
            lambda: obj_module.ingest_obj(path, path.with_suffix(".zv"), (1.0, 1.0, 1.0), **kwargs),
        )

    return _assert_same_outcome(new, lambda: _old_obj_write_args(path, **kwargs))


def _compare_stl(
    monkeypatch: pytest.MonkeyPatch, path: Path, **kwargs: Any,
) -> dict[str, Any] | None:
    def new() -> dict[str, Any]:
        return _capture_ingest(
            monkeypatch, stl_module,
            lambda: stl_module.ingest_stl(path, path.with_suffix(".zv"), (1.0, 1.0, 1.0), **kwargs),
        )

    return _assert_same_outcome(new, lambda: _old_stl_write_args(path, **kwargs))


# ---------------------------------------------------------------------------
# File generators
# ---------------------------------------------------------------------------

_NEWLINES = ("\n", "\r\n", "\r")


def _number(rng: np.random.Generator, x: float) -> str:
    """One coordinate, in one of the spellings real exporters produce."""
    spellings = (
        repr(x), f"{x:.6f}", f"{x:.3e}", f"{x:g}", f"{x:+.4f}", f"{x:.17g}",
        f"{x:.2E}", str(round(x)), "-0", "-0.0", ".5", "5.", "1e-310",
    )
    return spellings[int(rng.integers(len(spellings)))]


def _gap(rng: np.random.Generator) -> str:
    return (" ", " ", " ", "  ", "\t", " \t ")[int(rng.integers(6))]


def _indent(rng: np.random.Generator) -> str:
    return ("", "", "", "  ", "\t", " \t")[int(rng.integers(6))]


def _write_text(path: Path, rng: np.random.Generator, lines: list[str]) -> Path:
    newline = _NEWLINES[int(rng.integers(len(_NEWLINES)))]
    text = newline.join(lines)
    if rng.random() < 0.7:
        text += newline
    path.write_bytes(text.encode("ascii"))
    return path


def _random_obj(
    path: Path, rng: np.random.Generator, *, n_vertices: int, n_faces: int,
    arities: tuple[int, ...], corner_style: str | None = None,
) -> Path:
    lines = ["# generated OBJ", "", "mtllib scene.mtl"]
    names = ["cell_a", "cell b", "cell_c", "group  with   gaps"]
    declared = 0
    faces_left = n_faces
    with_normals = rng.random() < 0.5
    while declared < n_vertices or faces_left > 0:
        if rng.random() < 0.08:
            keyword = "o" if rng.random() < 0.5 else "g"
            name = names[int(rng.integers(len(names)))]
            lines.append(f"{_indent(rng)}{keyword} {name}")
        batch = int(rng.integers(1, 12)) if declared < n_vertices else 0
        for _ in range(batch):
            xyz = rng.normal(size=3) * 10.0 ** rng.integers(-3, 4)
            coords = _gap(rng).join(_number(rng, float(c)) for c in xyz)
            extra = ""
            if rng.random() < 0.1:
                extra = _gap(rng) + "1.0"  # homogeneous w
            elif rng.random() < 0.1:
                extra = " 0.5 0.25 1"  # vertex colour
            trailing = "  " * int(rng.random() < 0.1)
            lines.append(f"{_indent(rng)}v{_gap(rng)}{coords}{extra}{trailing}")
            if with_normals:
                lines.append("vn " + " ".join(_number(rng, float(c)) for c in rng.normal(size=3)))
            if rng.random() < 0.3:
                lines.append("vt 0.5 0.5")
            declared += 1
        if rng.random() < 0.05:
            lines.append("v 1 2")  # too short: ignored
        if rng.random() < 0.05:
            lines.append("f 1 2")  # too short: ignored
        if rng.random() < 0.05:
            lines.extend(["s off", "usemtl skin", "l 1 2", "   ", "#f 1 2 3"])
        if declared >= 3 and faces_left > 0:
            for _ in range(min(faces_left, int(rng.integers(1, 8)))):
                arity = arities[int(rng.integers(len(arities)))]
                corners = rng.integers(1, declared + 1, size=arity)
                tokens = []
                for c in corners.tolist():
                    ref = str(c) if rng.random() < 0.7 else str(c - declared - 1)
                    style = corner_style or ("v", "v/vt", "v//vn", "v/vt/vn")[int(rng.integers(4))]
                    tokens.append({
                        "v": ref, "v/vt": f"{ref}/{ref}", "v//vn": f"{ref}//{ref}",
                        "v/vt/vn": f"{ref}/1/{ref}",
                    }[style])
                lines.append(f"{_indent(rng)}f{_gap(rng)}{_gap(rng).join(tokens)}")
                faces_left -= 1
    return _write_text(path, rng, lines)


def _random_ascii_stl(path: Path, rng: np.random.Generator, n_faces: int) -> Path:
    lines = ["solid generated part"]
    for _ in range(n_faces):
        i1, i2 = _indent(rng) + "  ", _indent(rng) + "    "
        normal = _gap(rng).join(_number(rng, float(c)) for c in rng.normal(size=3))
        lines.append(f"{i1}facet normal{_gap(rng)}{normal}")
        lines.append(f"{i2}outer loop")
        for _ in range(3):
            xyz = _gap(rng).join(_number(rng, float(c)) for c in rng.normal(size=3) * 50)
            lines.append(f"{i2}  vertex{_gap(rng)}{xyz}")
        lines.append(f"{i2}endloop")
        lines.append(f"{i1}endfacet")
        if rng.random() < 0.05:
            lines.append("")
    lines.append("endsolid generated part")
    return _write_text(path, rng, lines)


def _random_binary_stl(path: Path, rng: np.random.Generator, n_faces: int) -> Path:
    records = np.zeros(n_faces, dtype=stl_module._BINARY_FACET)
    records["normal"] = rng.normal(size=(n_faces, 3))
    records["vertices"] = rng.normal(size=(n_faces, 3, 3)) * 100
    records["attribute"] = rng.integers(0, 2**16, size=n_faces)
    flat = records["vertices"].reshape(-1)
    specials = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-45, 3e30], dtype=np.float32)
    picks = min(20, flat.size)
    flat[rng.integers(flat.size, size=picks)] = rng.choice(specials, size=picks)
    records["vertices"] = flat.reshape(-1, 3, 3)
    with open(path, "wb") as f:
        f.write(b"binary STL from test_mesh_parsers".ljust(80, b"\0"))
        f.write(struct.pack("<I", n_faces))
        f.write(records.tobytes())
    return path


# ---------------------------------------------------------------------------
# Binary STL
# ---------------------------------------------------------------------------


class TestBinarySTL:

    @pytest.mark.parametrize("n_faces", [1, 7, 2000])
    def test_matches_old_parser(self, tmp_path: Path, n_faces: int) -> None:
        path = _random_binary_stl(tmp_path / "b.stl", np.random.default_rng(n_faces), n_faces)
        new_v, new_n = stl_module._parse_binary_stl(path)
        old_v, old_n = _old_parse_binary_stl(path)
        _assert_identical(new_v, old_v)
        _assert_identical(new_n, old_n)

    def test_spans_several_reads(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(stl_module, "_FACETS_PER_READ", 64)
        path = _random_binary_stl(tmp_path / "b.stl", np.random.default_rng(3), 1000)
        for new, old in zip(stl_module._parse_binary_stl(path), _old_parse_binary_stl(path)):
            _assert_identical(new, old)

    @pytest.mark.parametrize("merge_vertices", [True, False])
    def test_ingest_matches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, merge_vertices: bool,
    ) -> None:
        path = _random_binary_stl(tmp_path / "b.stl", np.random.default_rng(11), 300)
        assert _compare_stl(monkeypatch, path, merge_vertices=merge_vertices) is not None

    def test_signalling_nan_is_quieted_as_before(self, tmp_path: Path) -> None:
        path = _random_binary_stl(tmp_path / "b.stl", np.random.default_rng(5), 4)
        raw = bytearray(path.read_bytes())
        raw[84 + 12:84 + 16] = struct.pack("<I", 0x7F800001)  # first vertex x: signalling NaN
        raw[84:88] = struct.pack("<I", 0xFFA00002)  # first normal x
        path.write_bytes(bytes(raw))
        for new, old in zip(stl_module._parse_binary_stl(path), _old_parse_binary_stl(path)):
            _assert_identical(new, old)

    def test_trailing_bytes_ignored(self, tmp_path: Path) -> None:
        path = _random_binary_stl(tmp_path / "b.stl", np.random.default_rng(6), 10)
        path.write_bytes(path.read_bytes() + b"trailing junk")
        for new, old in zip(stl_module._parse_binary_stl(path), _old_parse_binary_stl(path)):
            _assert_identical(new, old)

    @pytest.mark.parametrize("missing", [1, 2])
    def test_missing_final_attribute_word_still_parses(self, tmp_path: Path, missing: int) -> None:
        path = _random_binary_stl(tmp_path / "b.stl", np.random.default_rng(7), 10)
        path.write_bytes(path.read_bytes()[:-missing])
        for new, old in zip(stl_module._parse_binary_stl(path), _old_parse_binary_stl(path)):
            _assert_identical(new, old)

    @pytest.mark.parametrize("cut", [3, 49, 50 * 9 + 10, 50 * 5])
    def test_truncated_file_fails_clearly(self, tmp_path: Path, cut: int) -> None:
        path = _random_binary_stl(tmp_path / "b.stl", np.random.default_rng(8), 10)
        path.write_bytes(path.read_bytes()[:-cut])
        with pytest.raises(IngestError):  # the old parser failed too
            _old_stl_write_args(path)
        with pytest.raises(IngestError, match="truncated.*declares 10 triangles"):
            stl_module.ingest_stl(path, tmp_path / "b.zv", (1.0, 1.0, 1.0))

    def test_corrupt_count_fails_before_allocating(self, tmp_path: Path) -> None:
        path = tmp_path / "b.stl"
        path.write_bytes(b"\0" * 80 + struct.pack("<I", 0xFFFFFFFF) + b"\0" * 50)
        with pytest.raises(IngestError, match="truncated.*4294967295 triangles"):
            stl_module.ingest_stl(path, tmp_path / "b.zv", (1.0, 1.0, 1.0))

    @pytest.mark.parametrize("size", [0, 10, 83])
    def test_shorter_than_header_fails_clearly(self, tmp_path: Path, size: int) -> None:
        path = tmp_path / "b.stl"
        path.write_bytes(b"\x01" * size)
        with pytest.raises(IngestError):
            _old_stl_write_args(path)
        with pytest.raises(IngestError, match="too short for the 84-byte header"):
            stl_module.ingest_stl(path, tmp_path / "b.zv", (1.0, 1.0, 1.0))

    def test_zero_triangles(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "b.stl"
        path.write_bytes(b"\0" * 80 + struct.pack("<I", 0))
        assert _compare_stl(monkeypatch, path) is None

    def test_solid_header_on_binary_file_behaves_as_before(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Sniffed as ASCII, as it always was; whatever that gives must not change.
        path = _random_binary_stl(tmp_path / "b.stl", np.random.default_rng(9), 50)
        raw = bytearray(path.read_bytes())
        raw[:12] = b"solid binary"
        path.write_bytes(bytes(raw))
        _compare_stl(monkeypatch, path)


# ---------------------------------------------------------------------------
# ASCII STL
# ---------------------------------------------------------------------------


class TestAsciiSTL:

    @pytest.mark.parametrize("seed", range(6))
    def test_matches_old_parser(self, tmp_path: Path, seed: int) -> None:
        path = _random_ascii_stl(tmp_path / "a.stl", np.random.default_rng(seed), 40 + 60 * seed)
        assert stl_module._parse_ascii_stl_fast(path) is not None
        for new, old in zip(stl_module._parse_ascii_stl(path), _old_parse_ascii_stl(path)):
            _assert_identical(new, old)

    def test_across_chunk_boundaries(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_text_tokens, "_CHUNK_BYTES", 61)
        path = _random_ascii_stl(tmp_path / "a.stl", np.random.default_rng(21), 120)
        assert stl_module._parse_ascii_stl_fast(path) is not None
        for new, old in zip(stl_module._parse_ascii_stl(path), _old_parse_ascii_stl(path)):
            _assert_identical(new, old)

    @pytest.mark.parametrize("merge_vertices", [True, False])
    def test_ingest_matches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, merge_vertices: bool,
    ) -> None:
        path = _random_ascii_stl(tmp_path / "a.stl", np.random.default_rng(12), 200)
        compared = _compare_stl(
            monkeypatch, path, merge_vertices=merge_vertices, dtype="float64",
        )
        assert compared is not None

    @pytest.mark.parametrize("text", [
        # The line reader's prefix rules, not the STL grammar.
        "solid s\nvertexish 1 2 3\nfacet normalised 4 5 6 7\nfacet  normal 1 x\n"
        "facet\tnormal 1 x\n  vertex 1 2 3 extra tokens\nendsolid",
        "solid s\nvertex 1 2 3\n",
        "solid only\nendsolid only\n",
        "",
        "solid s\n\n\n   \n\t\nvertex -0 +1.5 1E3",
        "solid s\nvertex nan inf -inf\nfacet normal -nan +inf 1e999\n",
        "solid s\nvertex 1_000 2 3\n",  # float() itself does the parsing
    ])
    def test_edge_cases_on_fast_path(self, tmp_path: Path, text: str) -> None:
        path = tmp_path / "a.stl"
        path.write_bytes(text.encode("ascii"))
        assert stl_module._parse_ascii_stl_fast(path) is not None
        for new, old in zip(stl_module._parse_ascii_stl(path), _old_parse_ascii_stl(path)):
            _assert_identical(new, old)

    @pytest.mark.parametrize("text", [
        "solid café\nvertex 1 2 3\n".encode(),  # non-ASCII byte
        b"solid s\nvertex 1\x0c2 3\n",  # form feed is whitespace to str.split
        b"solid s\nvertex 1\xa02 3\n",
    ])
    def test_fallback_inputs_match(self, tmp_path: Path, text: bytes) -> None:
        path = tmp_path / "a.stl"
        path.write_bytes(text)
        assert stl_module._parse_ascii_stl_fast(path) is None
        # Both readers decode with the platform's default encoding, so a byte
        # like 0xa0 reads under cp1252 and is an error under UTF-8 (Linux,
        # or PYTHONUTF8=1).  Either way the two must agree.
        try:
            expected = _old_parse_ascii_stl(path)
        except UnicodeDecodeError:
            with pytest.raises(UnicodeDecodeError):
                stl_module._parse_ascii_stl(path)
            return
        for new, old in zip(stl_module._parse_ascii_stl(path), expected):
            _assert_identical(new, old)

    @pytest.mark.parametrize("text", [
        "solid s\nvertex 1 2\n",
        "solid s\nfacet normal 0 0\n",
        "solid s\nvertex 1 2 three\n",
        "solid s\nvertex 1,0 2 3\n",
        "solid s\nendsolid\n",
    ])
    def test_malformed_input_fails_as_before(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str,
    ) -> None:
        path = tmp_path / "a.stl"
        path.write_text(text)
        assert _compare_stl(monkeypatch, path) is None


# ---------------------------------------------------------------------------
# OBJ
# ---------------------------------------------------------------------------

_ARITIES = {
    "triangles": (3,),
    "quads": (4,),
    "mixed": (3, 4),
    "polygons": (3, 4, 5, 6, 8),
    "polygons_only": (5, 6, 7),
    "quads_and_polygons": (4, 5),
}


class TestOBJ:

    @pytest.mark.parametrize("auto_object_id", [False, True])
    @pytest.mark.parametrize("kind", sorted(_ARITIES))
    def test_matches_old_parser(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, auto_object_id: bool,
    ) -> None:
        rng = np.random.default_rng(sorted(_ARITIES).index(kind))
        path = _random_obj(tmp_path / "m.obj", rng, n_vertices=400, n_faces=600,
                           arities=_ARITIES[kind])
        assert obj_module._parse_obj_fast(path, auto_object_id) is not None
        new = _compare_obj(monkeypatch, path, auto_object_id=auto_object_id)
        assert new is not None
        if kind == "quads":
            assert new["faces"].shape[1] == 4

    @pytest.mark.parametrize("style", ["v", "v/vt", "v//vn", "v/vt/vn"])
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_corner_styles_and_dtypes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, style: str, dtype: str,
    ) -> None:
        path = _random_obj(tmp_path / "m.obj", np.random.default_rng(31), n_vertices=150,
                           n_faces=200, arities=(3, 4, 6), corner_style=style)
        assert obj_module._parse_obj_fast(path, True) is not None
        assert _compare_obj(monkeypatch, path, dtype=dtype, auto_object_id=True) is not None

    @pytest.mark.parametrize("auto_object_id", [False, True])
    def test_across_chunk_boundaries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, auto_object_id: bool,
    ) -> None:
        # Tiny chunks put negative indices, object ids and long lines on
        # either side of a boundary.
        monkeypatch.setattr(_text_tokens, "_CHUNK_BYTES", 37)
        path = _random_obj(tmp_path / "m.obj", np.random.default_rng(41), n_vertices=300,
                           n_faces=400, arities=(3, 4, 5, 9))
        assert obj_module._parse_obj_fast(path, auto_object_id) is not None
        assert _compare_obj(monkeypatch, path, auto_object_id=auto_object_id) is not None

    def test_normals_used_only_when_counts_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        lines = ["v 0 0 0", "v 1 0 0", "v 0 1 0", "vn 0 0 1", "vn 0 0 1", "vn 0 0 1", "f 1 2 3"]
        path = tmp_path / "m.obj"
        path.write_text("\n".join(lines))
        new = _compare_obj(monkeypatch, path)
        assert new is not None and "normal" in new["kwargs"]["vertex_attributes"]
        path.write_text("\n".join([*lines, "vn 1 0 0"]))
        new = _compare_obj(monkeypatch, path)
        assert new is not None and new["kwargs"]["vertex_attributes"] is None

    @pytest.mark.parametrize("text", [
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n",  # index 0 becomes -1, as before
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\nv 5 5 5\nf -1 -2 -3 -4\n",
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/ 2/ 3/\nf 99 -99 000002\n",
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3 4 5 6 7 8 9 10 11\n",
        "g\no\nv 0 0 0\ng a\nv 1 0 0\ng b\ng a\nv 0 1 0\nf 1 2 3\n",
        "# comment only\nv nan inf -0\nv 1 2 3\nv 1 1 1\nf 1 2 3\n",
        "  v   1   2   3   \n\tv\t4\t5\t6\nv 7 8 9\n  f 1 2 3  \n",
        "vn 0 0 1\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1//1 2//1 3//1\n",
        "v 0 0 0\nv 1_0 0 0\nv 0 1 0\nf 1 2 3\n",  # float() itself does the parsing
    ])
    @pytest.mark.parametrize("auto_object_id", [False, True])
    def test_edge_cases_on_fast_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str, auto_object_id: bool,
    ) -> None:
        path = tmp_path / "m.obj"
        path.write_bytes(text.encode("ascii"))
        assert obj_module._parse_obj_fast(path, auto_object_id) is not None
        assert _compare_obj(monkeypatch, path, auto_object_id=auto_object_id) is not None

    @pytest.mark.parametrize("text", [
        b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf +1 +2 +3\n",  # int() takes a plus sign
        b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1_0 2 3\n",  # ... and underscores
        b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0000000000000000001 2 3\n",  # 19 digits
        "o café\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n".encode(),  # non-ASCII
        b"v 0 0 0\nv 1\x0b0 0\nv 0 1 0\nf 1 2 3\n",  # vertical tab
    ])
    @pytest.mark.parametrize("auto_object_id", [False, True])
    def test_fallback_inputs_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: bytes, auto_object_id: bool,
    ) -> None:
        path = tmp_path / "m.obj"
        path.write_bytes(text)
        assert obj_module._parse_obj_fast(path, auto_object_id) is None
        assert _compare_obj(monkeypatch, path, auto_object_id=auto_object_id) is not None

    @pytest.mark.parametrize("text", [
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3 # inline comment\n",
        "v 0 0 0\nv 1 0 0\nv 0 y 0\nf 1 2 3\n",
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1 /2 3\n",
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3.0\n",
        "vn 0 0 one\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n",  # no faces
        "f 1 2 3\n",  # no vertices
        "",
    ])
    def test_malformed_input_fails_as_before(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str,
    ) -> None:
        path = tmp_path / "m.obj"
        path.write_text(text)
        assert _compare_obj(monkeypatch, path) is None

    def test_integer_position_dtype_overflow_raises_as_before(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / "m.obj"
        path.write_text("v 0 0 0\nv 1e12 0 0\nv 0 1 0\nf 1 2 3\n")
        assert obj_module._parse_obj_fast(path, False) is not None
        assert _compare_obj(monkeypatch, path, dtype="int32") is None
