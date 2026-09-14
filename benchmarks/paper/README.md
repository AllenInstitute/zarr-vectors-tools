# Paper benchmarks

Eight panels, written as separate files, and one supplementary table.
This suite exists to
answer the three claims the Results section makes — that zarr-vectors
**works across geometry types**, is **efficient in space**, and is
**comparable to or faster than** the single-file format it replaces —
and nothing else. For exploratory sweeps see the sibling suites in
[`../formats/`](../formats/) and [`../internals/`](../internals/).

Unlike those two, this suite is **scripts, not notebooks**. Measurement
and plotting are separated on purpose: the sweep can be re-run on
reference hardware with no plotting stack installed, and the figure can
be restyled for a journal's column width without re-running anything.

```
run_sweep.py         measure -> results/measurements.csv + results/environment.json
run_large.py         the same sweep, one decade further -> results/large/
make_figure.py       plot    -> results/panels/*.{png,pdf} + results/supplementary_table.{md,csv}
profile_hotspots.py  why the timings look the way they do (see below)
_harness.py          timing, statistics, synthetic datasets
_formats.py          the competitor formats
```

**The panels are separate files, not a composed sheet.** Which panel
goes where, how many columns, and what the letters are is a layout
decision that belongs to whoever assembles the figure, so
`make_figure.py` writes one file per panel at one journal column (the
storage bars take two) and leaves the composition alone:

```
results/panels/A_storage_per_vertex.{png,pdf}   +  A_storage_per_vertex_key
                B_size_ratio                       B_size_ratio_key
                C_write_all                        C_write_all_key
                D_read_all                         D_read_all_key
                E_read_one                         E_read_one_key
                F_read_region                      F_read_region_key
                F2_read_region_equal_box           F2_read_region_equal_box_key
                G_vs_precomputed                   G_vs_precomputed_key
                H_read_one_vs_chunks               H_read_one_vs_chunks_key
                H2_read_one_vs_fragments           H2_read_one_vs_fragments_key
```

**Both region panels are log-log on all three scales** — size across the
bottom, query time on the left, box volume on the right. They have to be
once the sweep runs to 10⁷: the times on one panel span 2.7 ms to 55 s,
and the box shrinks from the whole domain to ~0.0014 % of it. A linear
pair flattens everything under the mesh series onto the axis.
`_region_panel(..., linear=True)` restores the old linear scales, which
read well only over a narrow range of N.

**F comes in two versions and only one belongs in the figure.** `F` sizes
the query box to ~100 *objects*, which is a different box per geometry —
100 mesh patches is 14 400 vertices and 100 points is 100 — so its volume
curve is one per geometry and the three series answer differently sized
questions. `F2` sizes it to ~100 *vertices* instead: one box volume for
all three at every N, one volume curve, and the same expected payload,
since every geometry spreads N vertices over the same domain. Pick `F` if
the question is "fetch a fixed number of objects", `F2` if it is "read a
fixed region".

**Each key is its own file, not burnt into its panel.** An in-panel
legend fixes how much of the plot area is spent on identification and
where it sits — decisions that belong to the layout the panels end up in,
not to the script that draws them. Every panel therefore ships
`<panel>_key.png` / `.pdf`, cropped to the legend and nothing else, so it
can be placed once for a row of panels or beside each one. `--keys` draws
them inside the panels as well, for reading a panel on its own;
`--letters` stamps A–G, `--formats` picks the output formats, and
`--combined` still writes the single-sheet `figure1.{png,pdf}`.

**Times are plotted in milliseconds.** The operations compared run from a
20 µs seek-and-read to a 20 s mesh write, and milliseconds put the middle
of that range in whole numbers. `measurements.csv` and the supplementary
table stay in seconds, which is the unit they were measured in.

## Running

```bash
pip install -e ".[all]" matplotlib pandas
python benchmarks/paper/run_sweep.py        # ~30 min
python benchmarks/paper/make_figure.py      # seconds
```

The two slow parts are the 10⁶ mesh writes (about 18 s each, five times
over) and precomputed's `by_id` index in the chunked block, which is one
file per annotation — a million files per write at the top size.

`--quick` shrinks the sweep to two small vertex sizes and two small
object counts for a smoke test. `--sizes 1000,10000,1000000` sweeps
exactly those vertex counts instead of the default decades.
`--workdir DIR` puts the scratch stores somewhere with room — the sweep
touches a few GB of transient files at the top size, and **it must be a
filesystem that takes a million small files in one directory**: the
precomputed competitor writes one file per annotation and fails outright
on NTFS-via-`ntfs3`. `environment.json` records `workdir_fstype` so a
run's numbers carry the filesystem they were measured on.
`--blocks storage,timing,spatial,chunked,fidelity` runs a subset. To
re-run one block into an existing sweep, use `--replace`: it drops the
rows the new run supersedes — matched on (block, geometry, operation,
implementation), so a sweep that now covers different sizes does not
leave the old sizes behind — and carries the rest over. `--append` adds
rows without dropping anything, which is only right when the new rows are
a series the file does not already hold. Either way `environment.json`
records which blocks were re-run under `mixed_run`, and keeps the
superseded package versions under `previous_run` if they changed.

### The long run — 10³ to 10⁷

`run_sweep.py` stops at 10⁶ because that is what a decade costs: one more
point per line roughly triples the wall time, and most runs of this
script are checking that nothing regressed rather than producing a
figure. The extra decade is not dropped, only moved.

```bash
python benchmarks/paper/run_large.py --workdir /scratch   # ~1.5 h
python benchmarks/paper/make_figure.py --results benchmarks/paper/results/large
```

`run_large.py` is a driver, not a second implementation — it runs
`run_sweep.py` once per block over `LARGE_SIZES`, so a point on a curve
means the same thing at 10⁷ as it does at 10³. What it adds is what a
multi-hour run needs:

- **Its own results directory.** Output goes to `results/large/`, so an
  interrupted or mis-configured run cannot overwrite the committed
  reference artefact. `--into-main` writes to `results/` instead, for
  when the large sweep *is* the artefact.
- **Resumability.** Each block is a separate process appending to one
  CSV, and `run_sweep.py` writes its CSV once at the end — so a block
  contributes either all its rows or none, and `--resume` restarts a
  killed run at the block it died on. A block that fails does not stop
  the ones after it; the summary names it.
- **A time budget you can read afterwards.** Every block is timed, and
  `results/large/run_large.log` keeps the per-block breakdown and total.

`--dry-run` prints the plan and the exact commands without measuring
anything. `--blocks`, `--geometries`, `--sizes` and `--workdir` pass
through to the sweep; `--figure` regenerates the panels when it finishes.

Three competitors stay capped below 10⁷, and each cap skips a point whose
shape is already settled at 10⁶ while its cost is superlinear in the
*writer* rather than in the format: GraphML above 10⁵ (networkx builds
the whole XML tree in memory), the per-point object index above 10⁶
(quadratic in points per chunk — see below), and precomputed annotations
above 10⁶ (`by_id` is one file per annotation, so 10⁷ is ten million
files, per repeat). Every cap logs its skip as the sweep reaches it, so
the omission is in the run's own output.

**What the top point actually costs.** One repeat of each operation at
N = 10⁷, measured on the reference machine, is what sets the run's
length — the sweep takes 2 to 3 samples of most of these:

| | point cloud | streamlines | meshes |
|---|---|---|---|
| generate | 0.1 s | 4.7 s | 0.7 s |
| ZV write all | 4.6 s | 44.4 s | 153.2 s |
| ZV read all | 0.1 s | 7.7 s | 41.5 s |
| ZV fetch one | <0.1 s | <0.1 s | 53.3 s |
| ZV replace one | <0.1 s | 9.8 s | — |
| competitor write | 0.2 s (PLY) | 10.3 s (TRK) | 3.0 s (STL) |
| competitor read | 0.1 s (PLY) | 2.6 s (TRK) | 0.6 s (STL) |
| store on disk | 0.10 GB | 0.12 GB | 0.10 GB |
| competitor on disk | 0.12 GB (PLY) | 0.12 GB (TRK) | 0.84 GB (STL) |

Mesh `fetch_one` is the outlier, and it is the curve continuing rather
than a new cliff: `read_mesh` has no `object_ids=` filter, so a patch
arrives inside its chunks and is filtered locally, and at a fixed
125-chunk grid those chunks get denser with N. The `detail` column
records the amplification per row — 1× at 10³, 10× at 10⁵, 70× at 10⁶,
and ~700× at 10⁷. See "Selective reads for polylines and meshes are
dominated by Python assembly" below.

Budget **around 1.5 hours** in total, and expect a **peak RSS of about
12 GB** during the mesh writes — the generator holds vertices, faces and
object ids for 10⁷ vertices while the writer builds its chunks. This is
not a sweep for a 16 GB laptop.

**The storage bars retarget themselves.** `A_storage_per_vertex` and
`B_size_ratio` are drawn at the largest N in the file, so plotting
`results/large/` moves them from 10⁶ to 10⁷ — where the three capped
competitors above are absent and simply have no bar. That is the right
default (the bars should show the largest measured dataset) but it does
mean the per-point-index price and the GraphML bar are on the 10⁶ figure
and not on the 10⁷ one. If the figure wants them, draw the storage
panels from `results/` and the curves from `results/large/`; which N a
panel targets is a layout decision, which is why `make_figure.py` takes
`--results` rather than deciding for you.

Repeat counts step down at the top size: the timing block takes
`LARGE_TIMING_RUNS = 3` samples rather than 5, and `n_runs()` drops to 2
for whole-dataset writes. The CSV records `n_runs` per row, so the wider
confidence interval at 10⁷ is visible rather than implied.

The blocks that hold N fixed rather than sweeping it — the
selected-fraction spatial sweep at 10⁶, fidelity at 10⁵, and the
chunk-count and fragmentation blocks at 10⁶ — are re-measured at those
same fixed sizes. Leaving them out would leave `results/large/` unable to
draw half the panels; re-measuring them is what makes every number in
that directory one machine state.

## What the panels show

The figure asks one question — **at what dataset size does zarr-vectors
overtake the alternative?** — and the answer is the crossing of each pair
of curves. Those crossings are computed, by log-log interpolation over
the sizes both curves cover, but they are **not drawn**: a vertical rule
marking a point on one pair crossed every other curve in the panel, and
these panels are read for their slopes. `make_figure.py` prints them
instead, and the table has the numbers:

```
crossovers (zarr-vectors overtakes to the right):
  Read one object       streamlines       4k
  read one region       point cloud     105k
  vs precomputed        write all         5k
```

| Row | Opponent | Panels |
|---|---|---|
| 1 · size | the alternative encoding | **A** bytes per vertex · **B** size ratio vs N |
| 2 · write and read | one file per dataset (PLY, TRK, STL) | **C** write everything · **D** read everything · **E** read one object |
| 3 · chunked | single files, then precomputed annotations | **F** / **F2** read one region · **G** write everything, read everything, read one |
| 4 · the partition itself | zarr-vectors against its own configurations | **H** read one object vs chunk count |

**H is the only panel with no opponent.** Every other panel runs on one
grid — 200-unit chunks over a 1000-unit domain, 125 of them, whatever the
dataset size — so nothing else in the sweep says what the chunk count
itself costs. H holds the dataset at 10⁶ vertices and moves the grid
instead, from 8 chunks to 8000, fetching one object out of each. There is
no competitor because a single file has no partition to vary; the
comparison is against the store's own other configurations. As measured,
the answer is that it barely matters at this size: a point fetch is
2.7–4.0 ms across three decades of chunk count, a streamline improves
from 24 ms at 8 chunks to 9.4 ms at 1000 and worsens to 14 ms at 8000,
and a mesh sits at 2.4–3.8 s throughout. `bin_shape` is held at a quarter
of the chunk side so the sweep moves one thing.

**H2 is the panel that answers "does crossing chunks cost?"** H cannot:
its objects are 32-unit random walks against chunks of 50 to 500 units,
so they span a mean of 1.1 cells at the coarse end and 3.1 at the fine
end, and its curves are dominated by how much payload a chunk holds
rather than by fragmentation. H2 fixes the grid at 125 chunks and the
dataset at 10⁶ vertices and varies the *object* instead, holding the
payload constant so only the fragmentation moves: a 128-vertex streamline
probe stretched across 1 to 125 cells, a 144-vertex mesh patch stretched
across 1 to 25, and — since a point can never straddle a cell — boxes of
equal volume for the point cloud, each selecting about 1000 points from
more and more cells.

The cost is real and close to linear in fragments. A streamline goes
**10.3 ms at one chunk to 165 ms at 125**, returning the same 128
vertices every time; the point-cloud query goes 3.9 ms to 18.0 ms over
1 to 25 cells for the same ~1000 points. The mesh line is the odd one:
3.6 s to 4.3 s while the vertices it had to read rose 21-fold, because
`read_mesh` has no object filter and is bound by Python face reassembly
rather than by either chunk selection or payload — the same conclusion
the sections below reach from the other direction, and the reason
`fetch_overshoot` is recorded per row.

Every panel is swept by **vertex count**, one point per decade. Objects
move with it at a fixed conversion, because each geometry's objects are a
fixed size — 1 vertex per point, 11.5 on average per streamline, exactly
144 per mesh patch — and the key names that conversion, so a point on the
streamline curve at 10⁵ vertices is also 8.7 k streamlines. Holding
object size fixed is what makes "more data" and "more objects" the same
axis; a sweep that grew the objects instead would be measuring something
else, and one that grew the object *count* would put the mesh curve at
144 k–144 M vertices and the point curve at 10³–10⁶, two lines that
barely overlap.


**G asks C, D and E again of a harder opponent, and that contrast is the
argument.** A single file has to read the whole thing to answer anything,
so it loses any partial-read comparison eventually and the only question
is where. Precomputed is chunked too — it prunes by chunk exactly as
zarr-vectors does, and its `by_id` index resolves one annotation in a
single file read — so the same three operations against it are the
comparison that says whether the layout earns its keep against something
built on the same idea. Nothing structural is handicapped there; what is
left is the layout.

Panel G plots those two and nothing else. Parquet (one row group per
spatial cell, pruned on footer statistics), plain Zarr and HDF5 are
measured on the same grid and are in the supplementary table, but two of
them cannot answer "read one" at all, and on the panel they were extra
lines making a point the table already makes. What precomputed does not
do is share storage between the two access patterns: the spatial index
and `by_id` are independent copies of every annotation, which is where
its size goes.

Single-object replacement and the query-size sweep are still measured and
are in the supplementary table; they are not plotted. The figure carries
the comparisons the Results section argues from.

**Read one** means: get exactly one object back, by the cheapest route
the public API offers. Points are addressed by position
(`read_points(bbox=...)`), streamlines by id
(`read_polylines(object_ids=[...])`); `read_mesh` has no `object_ids=`
filter, so a mesh patch is fetched through the chunks it touches and
filtered locally. All three return the whole object and nothing less, but
the mesh route reads up to 70× the object's own vertices to do it —
`fetch_overshoot` is recorded per row so that series is read for what it
is. Each object is the same size at every point on the sweep (1 vertex,
11.5 mean, and exactly 144 respectively), which is what makes the object
axis meaningful: panel E varies how many objects the store holds — 10⁶
vertices is 83 k streamlines or 6.9 k mesh patches — and never how big
the one being fetched is.

The figure has no panel for the first claim — that the format works
across geometry types — because correctness is not a curve. It is
measured instead as a round-trip fidelity check (write, read back,
compare sorted coordinates) for all five geometries and for every binary
competitor, and it is the first block of the supplementary table.

**Panel F carries a second vertical axis: the volume of the box.** The
box is resized at every N to hold about 100 objects, so the answer stays
the same size while the volume it is cut from shrinks — the whole domain
at 10³ vertices, 0.01 % of it at 10⁶. Without that on the panel, a
falling query-time curve reads as a speedup when it is really the same
work over a smaller box; with it, the prediction and its test are in one
frame. The volume curve is per geometry, because "100 objects" is 100
points but 14 400 mesh vertices, and it is drawn thin with a light fill
under it so it stays context rather than a fourth series.

Panel F is the load-bearing one, because it isolates the prediction the
layout is built on: a single-file format has to touch the whole file to
answer any spatial question, so its cost should track the size of the
haystack, while a chunked store touches only the chunks the box
intersects, so its cost should track the size of the answer. Read it
alongside the section below — as measured, that separation holds for
points and does not yet hold for polylines or meshes.

## Design rules

**One point per decade, 10³ to 10⁶.** `SIZES` is four vertex counts and
`OBJECT_SIZES` four object counts, a decade apart. These curves are close
to straight on log-log, so resolving *inside* a decade bought precision
on a slope that was never in doubt while making every panel dense enough
to be hard to read; span is what carries the argument, and where the span
stops is a cost decision — a 10⁷ point roughly triples the sweep's wall
time for one more dot per line — a 10⁷ mesh point alone is 14.4 M
vertices and over half an hour of repeats. A crossing is still located
by log-log interpolation between the two points that bracket it, which is
what it has always been at any spacing.

**One sweep variable, and it is vertices.** Every block sweeps `SIZES`,
so every geometry covers the same 10³–10⁶ range and the curves are read
against each other directly. Objects come along at a fixed conversion
because objects are a fixed size — the alternative, sweeping the object
*count*, puts the mesh curve at 144 k–144 M vertices against the point
curve at 10³–10⁶, and two curves that barely overlap cannot be compared
at a glance. Crossovers are still computed on the sizes the two curves
have in common, since not every competitor is measured at every size.

**Three guards on the top of the range, and each says so in the log.**
Only one of them drops a point at 10³–10⁶: GraphML stops at 10⁵ nodes,
because the networkx serialiser builds the whole XML tree in memory and
takes minutes beyond that. The other two are why the sweep stops at 10⁶
rather than going further — precomputed annotations (`by_id` is one file
per annotation, so 10⁷ points is ten million files per write, which
failed outright on an NTFS scratch disk at 10⁶) and the per-point object
index (the writer groups vertices by object with one full-array mask per
object, quadratic in the points per chunk: ~15 s at 10⁶ and about half an
hour at 10⁷). Raise `SIZES` and they bind; nothing is silently truncated
either way, because the sweep prints what it dropped and why.

**Repeats adapt to cost: at least five, more while they are cheap.**
`H.repeat` takes `TIMING_RUNS = 5` samples and then keeps going while the
samples so far total under half a second, up to 200. A 20 s mesh write is
therefore measured five times and a 9 ms TRK write a few hundred, for the
same half-second of extra wall time. Five *fixed* repeats of everything
was the earlier setting and it does not survive contact with the cheap
end of the range: one scheduler stall in five samples of a
millisecond-scale call produces a confidence interval wider than the mean
— a statement about the sample size, not about the operation. Intervals
are drawn as a filled band around each line, at low alpha so they read as
tolerance rather than as a second series, and the band's lower edge is
floored at a twentieth of the value so an interval that ran out of
samples cannot flood the panel.

**The palette is validated, not chosen.** One blue-to-magenta arc,
five fixed slots — deep blue, azure, violet, pink, deep magenta — checked
as a set for lightness band, chroma floor, colour-blind separation across
*every* pair rather than neighbouring ones, and 3:1 contrast against
white. The worst all-pairs distance is 9.2 ΔE under protanopia, above
the 8 ΔE bar, so identity survives colour-blind readers without leaning
on the markers to disambiguate. In the panels that compare
implementations rather than geometries the same arc does double duty:
blues are zarr-vectors, magentas are what it replaces, light is
uncompressed and dark is compressed, so a reader can see which side of
the argument a bar is on before reading the key. Colour is not the only
encoding in any case — line style and marker fill carry the
store-versus-alternative split, panel B gives each geometry its own
marker, and every panel is labelled.

**Panel A names every competitor, not just the smallest.** The Results
section argues against particular formats — `.trk` for streamlines,
`.obj` and `.stl` for meshes — so each is drawn and labelled even where
another encoding of the same geometry happens to be smaller. Panel B's
ratio uses whichever single file is smallest, which is the conservative
choice for a claim about storage, and it plots only the single-file
ratios — precomputed is a chunked layout, not the file anyone would
otherwise ship, and its ratio lives in the supplementary table. Parity is
a thick grey rule behind the curves, unlabelled: where a geometry crosses
from "bigger than the file" to "smaller" is the panel's whole content and
does not need naming.

**Every competitor is measured compressed as well as raw, binary
included.** An earlier version of this suite gzipped only the text
formats and compared a Zstd store against uncompressed binaries — the
same asymmetry it was written to avoid, and it flattered the store.
TRK gzips to 10.92 bytes per vertex and TRX has its own lossless deflate
setting at 10.84, both below anything zarr-vectors managed on the same
data (11.92 with Blosc/Zstd-9/bitshuffle, 12.04 with plain Zstd). TRX
also supports float16 positions at 5.17 bytes per vertex, which is
measured and reported as lossy — max coordinate error 0.25 units — and
is not used in any headline comparison, because the store it is being
compared against is lossless.

**Both sides are built from the same arrays.** `_harness.py` generates a
dataset once; the zarr-vectors store and the competitor file are written
from it. A size or timing difference is never an artefact of different
input data.

**Every competitor gets its best implementation, not a typical one.**
No reader in `_formats.py` loops over lines in Python: PLY and STL are
`np.frombuffer` over the record buffer, OBJ is written with `np.savetxt`,
the TRK bounding-box filter is a vectorised reduction over the flat point
buffer rather than a loop over streamlines. Where a format's on-disk
structure permits a cheap single-object edit, it gets it — replacing one
point in a PLY is an in-place 12-byte overwrite at a computed offset, and
replacing one patch in an STL overwrites that patch's contiguous triangle
run. Only TRK is charged a full rewrite, because variable-length records
genuinely cannot be located or resized without walking the file.

**Both sides return materialised arrays.** `np.frombuffer` hands back a
read-only view over the file buffer, which is faster than anything a
chunked store can return but is not the same product. The binary readers
copy into owned arrays so the comparison is like for like.

**The competitor bar is the best of the available formats.** Panel A and
the ratio in panel B compare against whichever single-file encoding is
*smallest* for that geometry, and the supplementary table lists all of
them. Text formats are also reported gzipped, so the size result is not
merely "binary beats ASCII".

**Uniform random point positions.** This is the worst case for both
compression (no spatial correlation to exploit) and chunk locality
(every chunk equally populated), so the point-cloud numbers are a lower
bound on what real data does.

**One deployed configuration everywhere.** Every store in the timing
blocks uses `compressor='zstd'` and a fixed 200-unit chunk shape.
Geometries whose objects are intrinsic — streamlines, skeletons, graphs,
meshes — carry an object index; a point cloud does not, because a point
has no id other than where it is, so a single point is addressed
spatially. The storage block measures the uncompressed store
and the opposite index choice for points and meshes, so the price of each
is in the supplementary table.

## What the current numbers say that the design does not

Everything below is reproducible: the query numbers from
`results/measurements.csv`, the loop costs from `python
profile_hotspots.py`. These are properties of today's implementation,
not of the layout.

### `bbox=` did not prune chunks — fixed

[`types/points.py:594`](../../../zarr-vectors-py/zarr_vectors/types/points.py)
computed its chunk whitelist only when the caller passed `chunks=`:

```python
chunk_set = None
if chunks is not None:                       # bbox alone never got here
    chunk_set = set(resolve_chunk_keys(..., bbox=bbox, chunks=chunks))
```

`resolve_chunk_keys` was correct and cheap all along — a 0.1 % box
resolves 125 chunks down to 1 in 0.4 ms — it simply was not consulted, so
the prefetch plan was built for every chunk in the level and the box was
applied as a mask afterwards. Reading one point cost 75 % of reading the
entire store, at every size.

Widening the guard to `if chunks is not None or bbox is not None` makes
reading one point **~5 ms, flat in N**, against ~80 ms before:

| N | before | after | read all |
|---|---|---|---|
| 10³ | 79.9 ms | 4.7 ms | 84.2 ms |
| 10⁴ | 76.1 ms | 4.8 ms | 76.7 ms |
| 10⁵ | 61.0 ms | 4.8 ms | 89.8 ms |
| 10⁶ | 87.8 ms | 5.0 ms | 86.4 ms |

That also restores the sensible ordering against streamlines: 5.0 ms for
a 1-vertex point versus 8.0 ms for a 12-vertex streamline, the difference
being the streamline's manifest decode.

The table above is the record of that one fix, measured when it was made.
The whole sweep has since been re-measured against a further-optimised
working tree, and the current CSV is faster again on both sides of it:
reading one point is **2.2–2.4 ms, flat from 10³ to 10⁶ vertices**, while
reading the whole store over the same range goes 7.5 ms → 22 ms, i.e.
nearly flat until the payload itself starts to matter. The ordering still
holds — 2.3 ms for a 1-vertex point against 3.3 ms for a 12-vertex
streamline at 10³, and 2.4 ms against 7.8 ms at 10⁶.

### `bbox=` returned nothing at all on attribute-binned stores — fixed

Found while fixing the above. Any store written with `chunk_by_attribute`
prefixes every chunk key with a bin axis, giving arity 4, while the
bbox-to-chunk mapping emits spatial coords of arity 3. The intersection
was therefore always empty and `read_points(bbox=...)` returned **zero
rows** — 0 instead of 156 in the reproduction — with no error. Both bbox
branches had it, and so did `resolve_chunk_keys`, which `read_mesh`,
`read_polylines` and `read_graph` all go through.

Fixed by comparing on the trailing spatial dims (an identity for
un-binned keys), plus a fallback so bin-level targeting defers to
chunk-level targeting when keys carry the extra axis rather than
addressing nothing. Verified from the full domain down to a 0.01 % box on
plain and attribute-binned stores alike; 333 core unit tests pass.

### Selective reads for polylines and meshes are dominated by Python assembly

The same workaround does nothing for polylines (1.1×) or meshes (1.0×),
because their cost is not chunk selection. At 10⁶ vertices `read_mesh`
walks 1.68 M single-face Python tuples out of `read_links`, doing three
dict lookups and three adds each; `read_polylines` decodes one manifest
block per object. Reassembling the same faces per *cell* in numpy —
one broadcast add per cell instead of per face — produces an identical
face set 4.5× faster (5.8 s → 1.3 s).

### Bulk write is per-record Python, not I/O or compression

Zstd versus no codec on a 10⁶-point write is 14.67 s versus 14.54 s, and
the 125 actual zarr chunk writes are ~2.5 s of a ~20 s profiled run. The
rest is interpreter time in a handful of identifiable loops:

| # | Site | 10⁶-vertex cost | Vectorised | Speedup |
|---|---|---|---|---|
| 1 | `types/points.py:437` per-object grouping (`mask = ids == obj_id`, once per object) | 7.27 s | 0.51 s | 14.4× |
| 2 | `core/arrays.py:1676` + `encoding/fragments.py:593` manifest encode | 1.83 s | 0.16 s | 11.6× |
| 3 | `types/meshes.py:277` + `spatial/boundary.py:563` face partitioning | 12.86 s | 2.06 s | 6.2× |
| 4 | `core/arrays.py:4476` + `types/meshes.py:541` face reassembly on read | 5.77 s | 1.27 s | 4.5× |
| 5 | `spatial/boundary.py:27` per-polyline boundary split | 1.51 s | 0.27 s | 5.7× |

`profile_hotspots.py` runs all five and asserts the vectorised form
produces output identical to the current one — same groups, byte-identical
manifests, equal bucket dicts, equal face sets, equal segment lists. The
speedups are therefore a measured floor, not an estimate.

### A per-point object index is expensive and usually unnecessary

Giving every point its own object id costs 36× on write (14.88 s vs
0.41 s), 16× on read, 4.7 bytes per point, and turns a single-point
edit from a flat 12 ms into 5.9 s at 10⁶ points, because index
maintenance is redone in proportion to the store rather than the edit
(visible in the `replace one object` rows of the supplementary table).
A point is better addressed by position — `read_points(bbox=...)` to find
it and `VertexRef.from_position` to edit it — which is what this suite
now does. The storage block keeps the indexed variant so the price stays
on record.

None of this is visible at the sizes
[`../internals/08_edit_operations.ipynb`](../internals/08_edit_operations.ipynb)
covers (50 000 objects), which is why it had not shown up before.

## Caveats

- **Fixed chunk shape across the whole size sweep.** A 200-unit chunk
  over a 1000-unit domain is 125 chunks whether the dataset has 10³ or
  10⁶ vertices, so the small-N end pays fixed per-chunk metadata against
  almost no payload. That is the honest "one store, growing dataset"
  scenario and it is what produces the crossover in panel B, but it is
  not the best zarr-vectors can do at small N — sizing chunks to a target
  occupancy recovers most of it. See
  [`../internals/06_chunk_shape.ipynb`](../internals/06_chunk_shape.ipynb).
- **Meshes have no per-object read.** `read_mesh` rejects `object_ids=`,
  so panel E fetches every chunk the object touches and filters down to
  the object locally. The result is exact — 144 of 144 vertices at every
  size — but the read that produced it is not: `fetch_overshoot` records
  how many vertices had to be read per vertex delivered, and on the
  sweep it runs **1× at 10³ and 10⁴ vertices, 10× at 10⁵ and 70× at
  10⁶**, because chunk occupancy grows while the chunk count does not.
  Panel E's mesh series is therefore mostly that amplification past 10⁴,
  and it is why the mesh curve climbs while the point and streamline
  curves stay nearly flat. There is also no public
  whole-object mesh *replacement*, which is why the replace measurement
  covers points and streamlines only.
- **Replacing an object remaps its id.** The edit path is copy-on-write:
  `EditSession` writes the new fragment and issues a new object id
  (`EditReport.oid_remap`). The timing covers the write; readers holding
  the old id must consult the remap.
- **The scratch filesystem is part of the measurement.** A chunked store
  is thousands of small files, so `--workdir` decides as much as the code
  does. `environment.json` records `workdir_fstype` for this reason: the
  committed run's timing blocks were measured on `ntfs3` and the chunked
  block on `ext4`, after precomputed's one-file-per-annotation `by_id`
  index failed outright on NTFS at 10⁶ points. The bulk numbers agree
  with an earlier ext4 run to within their intervals — a store's 125
  chunk files are not where NTFS hurts — but a comparison that turns on
  small-file cost should be re-run with one `--workdir` throughout.
- **Wall time and on-disk bytes only.** No memory profiling, no cold-cache
  measurements — every read here is served from a warm page cache, which
  flatters the single-file formats more than it flatters the store.
- **Local filesystem only.** Nothing here says anything about object
  storage, where the per-request latency that chunking amortises is the
  dominant term. See
  [`../internals/03_backends.ipynb`](../internals/03_backends.ipynb).
- **Check `environment.json` before comparing across blocks.** A block
  can be re-measured on its own with `--replace`, and if the installed
  package changed in between, the file then holds two generations of
  measurement. That case is recorded explicitly: `mixed_run` lists the
  blocks re-run and `previous_run` keeps the superseded versions. Absent
  those two keys, every row came from one run against one install.
  Package hashes carry a `-dirty` suffix when the tree they were measured
  from had uncommitted work in it, which a bare commit hash would hide.
- **Repeat counts vary by two orders of magnitude across the table.**
  Five samples of a 20 s mesh write, 200 of a 1.7 ms TRK write; the
  `runs` column says which. That is deliberate — it is what keeps the
  cheap end's intervals meaningful — but it means two rows' intervals are
  not equally well determined, and the widest ones left are all in the
  chunked block, which is sampled by the same rule at a lower floor.
- **Synthetic data, `SEED = 0`.** Real datasets have different sparsity
  and chunk occupancy and therefore different slopes.
- **GraphML is capped at 10⁵ nodes.** The networkx serialiser builds the
  whole XML tree in memory and takes minutes beyond that; the point is
  dropped and the omission logged rather than silently distorting the
  sweep.
- **Single machine, single run.** `results/environment.json` records the
  hardware, OS and package versions. Numbers are not portable across
  machines; the shapes of the curves are.
