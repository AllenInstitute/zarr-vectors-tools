# Findings for zarr-vectors, from the merged-links migration + parallel pyramid work

Reported by `zarr-vectors-tools`. Verified against branch `links-offset-merge`
(working tree) with zarr 3.2.1.

Most findings from this effort have now been **fixed in core** (see below).
Two remain open; neither is a links regression, and neither is reachable from
core's own tests — they need a downstream caller (a parallel writer, or a
composite store).

---

## Fixed in core (this branch) — concurrency of decentralized writes

`nonempty_chunks` is one attribute shared by every cell of an array, and array
attributes live in the array's `zarr.json`. So a per-cell write that stamps the
manifest is a read-modify-write of array-wide state: two workers writing
**disjoint cells** still collide on the `zarr.json` atomic rename. On Windows
the loser dies with `PermissionError [WinError 5]`; everywhere else it silently
loses the manifest entry while the payload sits on disk (`list_chunks` trusts
the manifest, so the cell becomes invisible).

Core already had the right protocol — `record_presence=False` on the worker,
coordinator re-derives via `derive_nonempty_chunks` / `finalize_links`. Three
holes made it unusable in practice; all three are now closed:

1. **`write_chunk_links` dropped the flag on its sidecar.** It threaded
   `record_presence` into the links cell but not into the `LINK_FRAGMENTS`
   write, so opting out still stamped `link_fragments`. (One-line fix; proven
   with a single-process repro where `link_fragments` was the only array whose
   manifest was non-empty.)
2. **`write_chunk_fragment_attributes` had no opt-out at all** — added
   `record_presence`, mirroring `write_chunk_vertices`.
3. **`create_links_array` re-stamped the family group's `zarr.json` on every
   offsets-array creation.** Decentralized writers each creating a *different*
   offsets array all rewrote `links/<delta>/zarr.json` and raced on it. It now
   stamps the family policy only when absent (a coordinator's
   `create_links_family`, or the first creator, establishes it) and checks the
   existing policy instead of blindly overwriting.

Regression test added:
`tests/test_links_merge_regressions.py::TestDecentralizedManifestProtocol::test_record_presence_false_is_honoured_by_every_per_chunk_writer`
— asserts `vertices`, `vertex_fragments`, `links/<delta>/<offsets>`,
`link_fragments` and `fragment_attributes/*` all report an empty
`nonempty_chunks` when written with `record_presence=False`.

**Residual, not fixed:** creating the *same* offsets array concurrently is still
not safe (a TOCTOU window between the existence check and the `zarr.json`
rename). Downstream works around it by having the coordinator pre-create every
offsets array before dispatching workers, so workers only ever write cells. A
general fix would make array creation tolerate a concurrent creator (catch the
rename failure and re-check existence) — worth considering, since every
decentralized writer hits it.

---

## Open: `composite` does not allocate its per-chunk arrays

**Severity: composite stores cannot be written at all.** Unrelated to the links
merge; confirmed still failing with my changes stashed.

`zarr_vectors/composite.py` writes per-geometry arrays such as `vertices_graph`,
but nothing allocates them for the single-array layout, so the first write
raises core's own error:

```
StoreError: Cannot write to 'vertices_graph' in 0: no chunk array at that path.
Per-chunk arrays must be allocated first (see arrays._ensure_array_dir /
create_sharded_chunk_array).
```

`_is_per_chunk_array` does not recognise the `<array>_<geometry>` naming, so
`_ensure_array_dir` never gives it a grid. Surfaced by
`tests/integration/test_lazy_sharding_rechunk.py::TestCompositeStore::test_composite_pipeline`.

## Open: a core test still assumes the pre-merge links shape

`tests/integration/test_lazy_sharding_rechunk.py::TestBornShardedWrites::test_graph_born_sharded`
asserts every node is a `zarr.Array` and trips on `0/links/0`:

```
AssertionError: ('0/links/0', <class 'zarr.core.group.Group'>)
```

That is the merged layout working as designed — `links/<delta>` is a **group**
whose children are one array per offsets segment. The test predates the merge
and needs updating (it should walk into the segments, or skip family groups).
Also confirmed pre-existing with my changes stashed.

---

## Minor asks (non-blocking)

- **No level-wide manifest coordinator.** `Group.rebuild_nonempty_manifests()`
  does not exist — only per-array `derive_nonempty_chunks`. Downstream
  re-implements the walk (`zarr_vectors_tools/_manifests.py`), including
  re-implementing the private `_is_per_chunk_array` predicate and skipping
  sharded arrays (`derive_nonempty_chunks` is documented unsharded-only, and
  would rewrite a sharded array's manifest to empty).
- **`iter_link_cells` decodes, so there is no O(cells) enumerator.** Every
  parallel coarsener needs "which cells exist" without paying to decode them, to
  bucket cells to targets in one pass. Decoding makes that O(records). Downstream
  rebuilds it from `list_link_offsets` + `list_chunks` + each array's `offsets`
  meta (`_links.list_link_cells`), re-implementing the private
  `_cell_endpoint_chunks`, and only for `delta == 0` (anchoring for `delta != 0`
  needs the private `_link_scales`). A public
  `list_link_cells(level_group, *, delta=0, involves=None)` would close it.
- **Doc drift (behaviour correct):** `reorder_vertices_implicit`'s docstring
  still cites `cross_chunk_links/0`; `_ensure_array_dir`'s example still names
  `cross_chunk_links`.

- **`batched_writes`' `compressor` docstring is wrong.** It claims "the default
  ``None`` resolves to zarr v3's default (``bytes`` + ``zstd``)". It does not —
  `resolve_compressor(None)` returns `['bytes']`, which
  `codecs_for_create_array` reduces to `[]`, i.e. **no compression**.
  `resolve_compressor`'s own docstring is the correct one. Worth fixing, since
  the two are read together and only one can be believed.

- **`create_store` warm-creating `vertices` silently defeats a downstream
  compressor.** A chunk array's codec pipeline is fixed at creation, and
  `create_store` warm-creates the `vertices`/`vertex_fragments` pair; every
  later `create_vertices_array` then short-circuits on the existing array. So a
  writer that opens a codec session around *its own* `create_*` calls compresses
  the small arrays and silently leaves the largest one raw. Fixed downstream by
  adding `compressor=` to `create_store` (the only place that can set it), but
  the sharp edge is general: any array warm-created without a codec can never
  gain one. Worth either documenting on `create_store` or making the warm create
  lazy.

---

## Note on `read_chunk_links` (fixed earlier — recorded for the changelog)

Worth keeping a regression test for, because the failure was silent and
shape-dependent. The reader defaulted `link_width` from the family group meta
(the **logical** width) and decoded at `ncols=link_width`, while `links_has_perm`
makes the **physical** width `1 + L` for a non-intra, `delta == 0`,
undirected-canonical array. Since `(1+L)·N ≡ N (mod L)` it raised iff
`N % L != 0` and **silently returned `(1+L)N/L` fabricated rows** iff
`N % L == 0` — for `link_width=2`, silent on every even record count:

```
N=1 truth=1 -> raised: Element count 3 is not divisible by ncols=2
N=2 truth=2 -> SILENT, 3 rows: [[0,0],[0,0],[1,1]]   # 2 real records
N=4 truth=4 -> SILENT, 6 rows
```

Core never hit it because its call sites all pass intra offsets. The current
code reads the array's `has_perm` stamp and decodes at `1 + L`; re-verified
correct for N=1..4.
