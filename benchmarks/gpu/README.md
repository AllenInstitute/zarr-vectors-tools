# GPU read benchmark

One question: **how much faster does a whole-level read get on a CUDA
device, and from what size?** It times `read_cells` over every cell of a
point-cloud store three ways:

| impl | call | what happens |
|---|---|---|
| `cpu` | `read_cells(lg, cells, arrays)` | read and decode on the host |
| `gpu (host decode)` | `device="cuda", decode="host"` | decode on the host, then one upload per array |
| `gpu (device decode)` | `device="cuda", decode="device"` | stored bytes to the device, decoded there (nvCOMP for zstd) |

Each size is run twice, on an uncompressed store and on a zstd one. Each
point is the **mean of 5 timed runs** after one untimed warm-up, with a
95 % Student's-t band. Every GPU result is checked element-wise against
the host result before its row is written.

Layout and style follow [`../paper/`](../paper/). Measuring and plotting
are separate scripts, each panel is its own file with its key beside it,
and times are plotted in milliseconds. `make_figure.py` imports the paper
figure's helpers rather than copying them, so if you restyle the paper,
these panels change with it. The domain, the 125-cell chunk grid and the
point generator come from `../paper/_harness.py`.

```
run_sweep.py     measure -> results/measurements.csv + results/environment.json
make_figure.py   plot    -> results/figure.{png,pdf} + results/panels/*.{png,pdf}
                            + results/supplementary_table.{md,csv}
```

`results/figure.{png,pdf}` is all four panels on one lettered sheet with
one key shared below. Time is on the top row and speed-up on the bottom;
the left column is uncompressed and the right column is zstd. The panels
are also written as separate files, the way the paper suite does it, for
laying out elsewhere.

```
results/panels/A_read_uncompressed      + _key    time vs vertices, log-log
               B_read_zstd              + _key
               C_speedup_uncompressed   + _key    cpu time / gpu time
               D_speedup_zstd           + _key
```

## Running

The sweep needs cupy and nvCOMP's Python bindings, and a `zarr_vectors`
with the GPU extension. `runtime_capabilities()` must report
`device_decode` and `gpu_codecs`, or the script stops before it starts.
That extension is on the `gpu-backend` branch of `zarr-vectors-py`, so
point `PYTHONPATH` at a checkout of it:

```bash
PYTHONPATH=/path/to/zarr-vectors-py@gpu-backend \
  python benchmarks/gpu/run_sweep.py          # 10^3..10^7, ~2 min
python benchmarks/gpu/make_figure.py          # seconds
```

`--quick` runs two small sizes. `--sizes 1e5,1e6` sweeps exactly those
sizes. `--workdir DIR` puts the scratch stores somewhere else; the
largest are ~0.3 GB. `environment.json` records the GPU, the CUDA driver
and runtime, cupy, nvCOMP and the `zarr_vectors` commit next to what the
paper sweep records.

## What the committed run shows

RTX A2000 12GB, `zarr_vectors` `06a3bc8`:

- **Device decode is the only GPU mode worth having for bulk reads.** At
  10⁷ vertices it is 2.3× faster than the CPU uncompressed (182 → 79 ms)
  and 3.0× faster with zstd (252 → 84 ms). It pulls clear from about
  3·10⁶ vertices.
- **Host decode never wins.** It does the CPU's work and then pays for
  the upload, so it runs 0.9–1.0× at every size where the timing is not
  noise. Use it only when the caller needs device arrays and the store is
  small.
- **Below ~10⁶ vertices everything takes 8–30 ms** and the modes are
  within a few milliseconds of each other. That regime is per-call
  overhead, not decode. The uncompressed device-decode curve sits
  1.2–1.7× lower there. With zstd the device decode is *slower* at 10³
  (0.7×), because nvCOMP's launch cost is larger than the whole host
  decode.
- **With zstd on the device, the stored size barely matters.** At 10⁷
  the zstd read (84 ms) costs about the same as the uncompressed one
  (79 ms). On the host, zstd adds 40 %.

This is a single-machine sanity sweep, not an authoritative number.
The store is in the page cache after the warm-up, so this measures
decode and assembly, not disk or network I/O.
