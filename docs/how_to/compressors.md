# Choosing a compressor

Compression is off by default. Every write path in this package accepts a
`compressor=` argument that is forwarded, unchanged, to core's
`zarr_vectors.encoding.compression.resolve_compressor`; the CLI exposes
the same choice as `--compressor {none,zstd,blosc}`.

The decision is a straightforward trade of write throughput against
stored size — but it interacts with one structural rule that catches
people out, so read the next section before you pick.

## The rule that governs everything: codecs are fixed at creation

A Zarr array's codec pipeline is written into its `zarr.json` when the
array is **created**, and cannot be changed afterwards. Three consequences
follow, and all three are load-bearing in this package.

**1. The compressor only has to be active around array creation.**
Every later per-cell write — including writes from parallel worker
processes — encodes to match the array's existing pipeline automatically.
There is nothing to thread through to the workers. This is why the write
paths wrap only their `create_*` calls, and use `nullcontext()` when no
compressor was requested, so the default path stays byte-for-byte what it
was before the option existed.

**2. Level 0's codec does not propagate to coarser levels.** Each pyramid
level creates its own arrays, so `build_pyramid` forwards `compressor=`
to every level explicitly. If you build level 0 compressed and then run
`zvtools pyramid` without `--compressor`, you get a store whose coarse
levels are raw.

:::{warning}
`zvtools pyramid`'s `--compressor` defaults to `none`, the same as
`convert`. Match the value level 0 was written with, or the store ends up
non-uniform:

```bash
zvtools convert tracts.trk out.zarrvectors --num-chunks 5000 --compressor zstd
zvtools pyramid out.zarrvectors --coarsen 8,8 --sparsity 2,4 --compressor zstd
#                                                            ^^^^^^^^^^^^^^^^^ not optional
```
:::

**3. A warm-created array can never gain a codec later.** `create_store`
warm-creates `vertices` and `vertex_fragments`. If the compressor is not
set at that point, the store's two largest arrays stay raw forever, no
matter what you pass downstream. This was a real bug; it is fixed by
`create_store` taking `compressor=` directly.

## What it costs and what it saves

The one measured figure in this codebase, from `ingest_trk_parallel`'s
docstring:

| Setting | Stored size | Write path |
| --- | --- | --- |
| `None` (default) | `n_vertices × ndim × itemsize` bytes exactly — the same payload a `.trk` holds | fastest; raw bytes |
| `"zstd"` | ~2.4× smaller on HCP tract coordinates | slower, synchronous codec encode |
| `"blosc"` | comparable to `zstd` | slower, synchronous codec encode |

So: roughly **halves** a streamline store, at the cost of a slower write.

## When to compress

**Compress** when the store is going to a cloud object store and will be
read many more times than written, when you are storing tractography or
skeleton coordinates (spatially coherent data compresses well), or when
storage cost dominates.

**Don't compress** when you are iterating locally on ingest parameters and
write time dominates your loop, or when the data is high-entropy.

That second case is not hypothetical. The CLI tests
(`tests/test_cli.py:264-350`) carry a pointed comment: on synthetic
*random* coordinates zstd produces a **larger** store than raw, because
there is nothing to exploit and you pay the framing overhead. The tests
therefore use realistic tract-like data, where coordinates advance about
one voxel per step. If you are benchmarking compression on made-up data,
you will measure the wrong thing.

## The `None` gotcha

:::{warning}
`resolve_compressor(None)` returns `['bytes']` — that is **no
compression**, not Zarr v3's default `bytes`+`zstd` pipeline.

Core's `batched_writes` docstring claims otherwise. It is wrong;
`resolve_compressor`'s own docstring is the correct one. If you have been
relying on "the default compresses", you have a raw store. Check it:

```python
import zarr
arr = zarr.open(f"{store_path}/0/vertices")
print([c["name"] for c in arr.metadata.to_dict()["codecs"]])
# ['bytes']          -> raw
# ['bytes', 'zstd']  -> compressed
```
:::

Note that `--compressor none` is translated to Python `None` before the
call, specifically so the no-compressor path stays identical to the
historical default.

## Verifying it actually applied

Because the codec is per-array and set at creation, "did it work?" is a
question about the *right* array. A compressor can land on a small
sibling array and miss `vertices` entirely — that is what the test
`test_trk_compressor_reaches_the_vertices_array` guards against. Check
`vertices`, and check every level:

```python
import zarr

for level in range(3):
    arr = zarr.open(f"{store_path}/{level}/vertices")
    names = [c["name"] for c in arr.metadata.to_dict()["codecs"]]
    print(level, names)
```

Compression is lossless, so the geometry is unaffected —
`test_trk_compressor_roundtrips_identically` asserts that a `zstd` store
and a raw store read back equal.

## Dependencies

`pyproject.toml` declares no `blosc`, `zstd`, or `numcodecs` dependency
directly; the codecs arrive transitively with `zarr>=3.0`. You do not
need to install anything extra to use `--compressor zstd`.

## See also

- [Parallel workflows](parallelism.md) — the other half of write throughput.
- [Building pyramids](../multiresolution/building_pyramids.md) — where per-level forwarding happens.
- [The `zvtools` CLI](../getting_started/cli.md)
- [Upstream findings](../upstream/links-merge-findings.md) — where the `None` docstring bug is recorded.
