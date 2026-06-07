"""Reusable real-flywire cutout ingest + equivalence manifest.

Used to prove the precomputed-skeleton ingest migration (core -> tools) is
behaviour-preserving: the SAME cached cutout is ingested before and after the
move and the resulting stores are byte-compared.

Subcommands:
  fetch  <cache.pkl>                 read the cutout from GCS once, cache it
  ingest <cache.pkl> <out> <man.json> ingest the cached cutout -> store + manifest

The ingest entry points are imported import-location-agnostically, so the
exact same script runs against the pre-migration (zarr_vectors.ingest) and
post-migration (zarr_vectors_tools.ingest) module locations.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys


def _imports():
    """Import the ingest API from whichever location currently provides it."""
    try:
        from zarr_vectors_tools.ingest.precomputed_skeletons import (  # type: ignore
            InMemoryFragsReader, PrecomputedFragsReader, SkeletonInfo,
            enumerate_frag_keys, run_ingest,
        )
        src = "zarr_vectors_tools.ingest"
    except ImportError:
        from zarr_vectors.ingest.precomputed_skeletons import (  # type: ignore
            InMemoryFragsReader, PrecomputedFragsReader, SkeletonInfo,
            enumerate_frag_keys, run_ingest,
        )
        src = "zarr_vectors.ingest"
    return dict(
        InMemoryFragsReader=InMemoryFragsReader,
        PrecomputedFragsReader=PrecomputedFragsReader,
        SkeletonInfo=SkeletonInfo,
        enumerate_frag_keys=enumerate_frag_keys,
        run_ingest=run_ingest,
        src=src,
    )


def _env_ints(name: str, default: list[int]) -> list[int]:
    v = os.environ.get(name)
    return [int(x) for x in v.split(",")] if v else default


def _env_floats(name: str, default: list[float]) -> list[float]:
    v = os.environ.get(name)
    return [float(x) for x in v.split(",")] if v else default


# All overridable via env so the same script drives any cutout, e.g.:
#   ANCHOR=17398,10448,3088 COUNTS=4,4,2 STRIDES=8,8,8 SPF=1,1,4 \
#     python ingest_cutout.py run /tmp/flywire_skel_aligned.zv
URL = os.environ.get("ZV_URL", "gs://flywire_v141_m783/skeletons_mip_1")
ANCHOR = tuple(_env_ints("ANCHOR", [17910, 8912, 3088]))
COUNTS = tuple(_env_ints("COUNTS", [2, 2, 1]))
STRIDES = _env_ints("STRIDES", [8, 8, 8, 4])
CSF = _env_ints("CSF", [2, 2, 2, 2])
SPF = _env_floats("SPF", [1.0, 1.0, 1.0, 4.0])


def fetch(cache_path: str) -> None:
    api = _imports()
    reader = api["PrecomputedFragsReader"](URL)
    info = reader.info
    keys = api["enumerate_frag_keys"](info, ANCHOR, COUNTS)
    chunks = {k: reader.read_chunk(k) for k in keys}
    payload = {
        "resolution_nm": tuple(info.resolution_nm),
        "chunk_size_nm": tuple(info.chunk_size_nm),
        "vertex_attributes": info.vertex_attributes,
        "keys": keys,
        "chunks": chunks,  # dict[key -> dict[seg_id -> {vertices,edges,radius,...}]]
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    nseg = sum(len(c) for c in chunks.values())
    nv = sum(len(p["vertices"]) for c in chunks.values() for p in c.values())
    print(f"cached {len(keys)} chunks, {nseg:,} segments, {nv:,} verts -> {cache_path}")


def _tree_hash(out_dir: str) -> dict[str, str]:
    """Map every file under out_dir to its sha256 (relative paths)."""
    import os
    hashes: dict[str, str] = {}
    for root, _dirs, files in os.walk(out_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, out_dir)
            h = hashlib.sha256()
            with open(fp, "rb") as f:
                for blk in iter(lambda: f.read(1 << 20), b""):
                    h.update(blk)
            hashes[rel] = h.hexdigest()
    return hashes


def ingest(cache_path: str, out_dir: str, manifest_path: str) -> None:
    import shutil
    api = _imports()
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    info = api["SkeletonInfo"](
        base_url="mem://cutout",
        resolution_nm=tuple(payload["resolution_nm"]),
        chunk_size_nm=tuple(payload["chunk_size_nm"]),
        vertex_attributes=payload["vertex_attributes"],
    )
    reader = api["InMemoryFragsReader"](info, payload["chunks"])
    keys = payload["keys"]
    res = payload["resolution_nm"]
    bounds = (
        [ANCHOR[a] * res[a] for a in range(3)],
        [(ANCHOR[a] + COUNTS[a] * 512) * res[a] for a in range(3)],
    )
    shutil.rmtree(out_dir, ignore_errors=True)
    summary = api["run_ingest"](
        reader, out_dir, keys, bounds_nm=bounds,
        strides=STRIDES, chunk_scale_factors=CSF, sparsity_factors=SPF,
        progress=False,
    )
    # Drop the non-deterministic 'pyramid' nested timing if present; keep counts.
    levels = summary.get("pyramid", {}).get("levels", [])
    manifest = {
        "ingest_src": api["src"],
        "summary": {k: v for k, v in summary.items() if k != "pyramid"},
        "levels": [
            {kk: lv[kk] for kk in ("vertex_count", "fragment_count", "object_count")
             if kk in lv}
            for lv in levels
        ],
        "tree_sha256": _tree_hash(out_dir),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"ingested via {api['src']} -> {out_dir}")
    print(f"  summary: {manifest['summary']}")
    print(f"  levels:  {manifest['levels']}")
    print(f"  files:   {len(manifest['tree_sha256'])} (manifest -> {manifest_path})")


def run(out_dir: str) -> None:
    """Full from-scratch ingest straight from the precomputed layer (the real
    production path: PrecomputedFragsReader → run_ingest → pyramid)."""
    import shutil
    api = _imports()
    reader = api["PrecomputedFragsReader"](URL)
    info = reader.info
    keys = api["enumerate_frag_keys"](info, ANCHOR, COUNTS)
    res = tuple(info.resolution_nm)
    bounds = (
        [ANCHOR[a] * res[a] for a in range(3)],
        [(ANCHOR[a] + COUNTS[a] * info.chunk_size_voxels[a]) * res[a] for a in range(3)],
    )
    print(f"ingest via {api['src']}: {URL}")
    print(f"  anchor={ANCHOR} counts={COUNTS} -> {len(keys)} .frag keys")
    print(f"  strides={STRIDES} csf={CSF} spf={SPF}")
    print(f"  bounds_nm={bounds}  -> {out_dir}")
    shutil.rmtree(out_dir, ignore_errors=True)
    summary = api["run_ingest"](
        reader, out_dir, keys, bounds_nm=bounds,
        strides=STRIDES, chunk_scale_factors=CSF, sparsity_factors=SPF,
        progress=True,
    )
    print("summary:", {k: v for k, v in summary.items() if k != "pyramid"})
    for lv in summary.get("pyramid", {}).get("levels", []):
        print("  level:", {k: lv[k] for k in
                           ("vertex_count", "fragment_count", "object_count")
                           if k in lv})
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "fetch":
        fetch(sys.argv[2])
    elif cmd == "ingest":
        ingest(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "run":
        run(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(2)
