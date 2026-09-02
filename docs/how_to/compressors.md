# Choosing a compressor

:::{important}
**The codec pipeline is owned by the parent package.** What codecs a Zarr
Vectors store may use, what each `compressor=` value resolves to, and why
the default is uncompressed are specified at
{zvpy}`the codec pipeline <spec/foundations/codec_pipeline.html>`. That page is the authority
on the values; this one covers only how they are threaded through this
package's workflows and CLI.
:::

Compression is off by default. Every write path in this package accepts a
`compressor=` argument that is forwarded, unchanged, to the parent
package's writers; the CLI exposes the same choice as
`--compressor {none,zstd,blosc}`.

## The rule that governs everything

A Zarr array's codec pipeline is written into its `zarr.json` when the
array is **created**, and cannot be changed afterwards — see
{zvpy}`the codec pipeline <spec/foundations/codec_pipeline.html>`. Three consequences follow,
and all three are load-bearing in *this* package:

**1. The compressor only has to be active around array creation.**
Every later per-cell write — including writes from parallel worker
processes — encodes to match the array's existing pipeline automatically.
There is nothing to thread through to the workers. This is why the write
paths here wrap only their creation calls, and use `nullcontext()` when no
compressor was requested, so the default path stays byte-for-byte what it
was before the option existed.

**2. Level 0's codec does not propagate to coarser levels.** Each pyramid
level creates its own arrays, so `build_pyramid` forwards `compressor=` to
every level explicitly.

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

**3. A warm-created array can never gain a codec later.** Store creation
warm-creates `vertices` and `vertex_fragments`. If the compressor is not
set at that point, the store's two largest arrays stay raw forever, no
matter what you pass downstream — which is why the compressor has to be
supplied at store-creation time rather than at first write.

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

That second case is not hypothetical. The CLI tests carry a pointed
comment: on synthetic *random* coordinates zstd produces a **larger** store
than raw, because there is nothing to exploit and you pay the framing
overhead. The tests therefore use realistic tract-like data, where
coordinates advance about one voxel per step. If you are benchmarking
compression on made-up data, you will measure the wrong thing.

## The default is uncompressed

:::{warning}
`--compressor none` — and the Python `None` it is translated to — means
**no compression**, not Zarr v3's `bytes`+`zstd` default pipeline. The
resolved pipeline is `[{"name": "bytes"}]`; the table at
{zvpy}`the codec pipeline <spec/foundations/codec_pipeline.html>` gives the mapping for every
accepted value.

If you have been assuming "the default compresses", you have a raw store.
:::

## Verifying it actually applied

Because the codec is per-array and set at creation, "did it work?" is a
question about the *right* array. A compressor can land on a small sibling
array and miss `vertices` entirely — that is what the test
`test_trk_compressor_reaches_the_vertices_array` guards against. Check
`vertices`, and check every level:

```python
import zarr

for level in range(3):
    arr = zarr.open(f"{store_path}/{level}/vertices")
    names = [c["name"] for c in arr.metadata.to_dict()["codecs"]]
    print(level, names)
    # ['bytes']          -> raw
    # ['bytes', 'zstd']  -> compressed
```

Compression is lossless, so the geometry is unaffected —
`test_trk_compressor_roundtrips_identically` asserts that a `zstd` store
and a raw store read back equal.

## Dependencies

`pyproject.toml` declares no `blosc`, `zstd`, or `numcodecs` dependency
directly; the codecs arrive transitively with `zarr>=3.0`. You do not need
to install anything extra to use `--compressor zstd`.

## See also

- **Parent package:** {zvpy}`the codec pipeline <spec/foundations/codec_pipeline.html>` — the codec pipeline itself.
- [Parallel workflows](parallelism.md) — the other half of write throughput.
- [Building pyramids](../multiresolution/building_pyramids.md) — where per-level forwarding happens.
- [The `zvtools` CLI](../getting_started/cli.md)
