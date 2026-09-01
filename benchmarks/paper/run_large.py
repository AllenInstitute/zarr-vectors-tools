#!/usr/bin/env python
"""Drive ``run_sweep.py`` over one more decade, 10^3 to 10^7.

    python run_large.py                         # the whole thing, hours
    python run_large.py --workdir /hdd/scratch  # scratch off the system disk
    python run_large.py --resume                # pick up where a kill left off
    python run_large.py --dry-run               # print the plan, measure nothing

``run_sweep.py`` stops at 10^6 on purpose: it is mostly run to check that
nothing regressed, and one more decade roughly triples its wall time.
This is the same measurement code driven over ``LARGE_SIZES`` instead --
no separate implementation, no second definition of what a point on a
curve means.  What it adds is the machinery a multi-hour run needs and a
twenty-minute one does not:

**Its own results directory.**  Output goes to ``results/large/`` by
default, so the committed 10^3..10^6 reference artefact is never
overwritten by a run that was interrupted, run on the wrong machine, or
run against a dirty working tree.  ``make_figure.py --results
results/large`` plots it; ``--into-main`` writes to ``results/`` instead,
for when the large sweep *is* the artefact.

**Block-by-block, resumable.**  Each block is a separate ``run_sweep.py``
process appending to one CSV.  That is not just tidiness: ``run_sweep``
writes its CSV once, at the end, so a block either contributes all its
rows or none of them -- there is no such thing as half a block in the
file.  ``--resume`` therefore only has to ask which block names are
already present, and a run killed four hours in restarts at the block it
died on rather than at the beginning.

**A time budget you can see.**  Every block is timed and the running
total is printed, so an unattended run leaves behind a record of where
the hours went -- which is the number anyone deciding whether to re-run
this actually wants.

What the extra decade costs, and what it does not
-------------------------------------------------
Three competitors are already capped below 10^7 by ``run_sweep``, and
the caps are deliberate rather than incidental -- each one skips a point
whose *shape* is already settled and whose cost is superlinear in a way
that says more about the writer than about the format:

* ``GraphML`` above 10^5 -- networkx builds the whole XML tree in memory.
* ``zarr-vectors (zstd, per-point object index)`` above 10^6 -- the
  writer groups vertices by object with one full-array mask per object,
  which is quadratic in points per chunk.
* ``precomputed annotations`` above 10^6 -- ``by_id`` is one file per
  annotation, so 10^7 is ten million files, per repeat.

Each is logged as a skip when the sweep reaches it, so the omission is
in the run's own output and not only in this docstring.

The blocks that hold N fixed rather than sweeping it -- the
selected-fraction spatial sweep at 10^6, fidelity at 10^5, and the
chunk-count and fragmentation blocks at 10^6 -- are re-measured at those
same fixed sizes.  They are not free, but leaving them out would leave
``results/large/`` unable to draw half the panels, and re-measuring them
here is what makes every number in that directory one machine state.

What to have spare
------------------
Roughly **1.5 hours**, a few GB of scratch, and about **12 GB of RAM**.
Memory is the constraint that bites without warning: at 10^7 the mesh
generator holds vertices, faces and object ids for ten million vertices
while the writer builds its chunks, and that peak is the whole run's
peak.  The single slowest measurement is one mesh write at 153 s, taken
three times per block that writes one.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LARGE = RESULTS / "large"

# Blocks in the order they run, with the ``block`` column values each one
# writes.  The order is cheapest-and-most-load-bearing first, so a run
# that is killed early still leaves the panels that matter most; the
# column values are what ``--resume`` matches on.
#
# ``chunks`` and ``fragments`` do not sweep N at all -- they hold 10^6
# and move the chunk grid instead -- so the extra decade does not reach
# them and they cost here exactly what they cost in the normal sweep.
BLOCKS = [
    ("storage",   ("storage",),
     "on-disk bytes for every (geometry, N, format)"),
    ("timing",    ("bulk", "object"),
     "write/read everything, fetch one, replace one"),
    ("spatial",   ("spatial_fraction", "spatial_fixed", "spatial_volume"),
     "bbox queries, swept three ways"),
    ("chunked",   ("chunked",),
     "versus Parquet / plain Zarr / HDF5 / precomputed"),
    ("fidelity",  ("fidelity",),
     "round trip, per geometry (fixed at 10^5)"),
    ("chunks",    ("chunks",),
     "chunk-count sweep (fixed at 10^6)"),
    ("fragments", ("fragments",),
     "fragmentation sweep (fixed at 10^6)"),
]

# Scratch headroom to ask for before starting.  At 10^7 one STL is
# 0.84 GB (measured) and one OBJ about the same, the timing block holds a
# store and a competitor file at once, and ``H.repeat`` writes a fresh
# copy per repeat before tearing it down.  Peak is a few GB; this asks
# for enough margin that a run does not die at hour one on ENOSPC.
MIN_FREE_GB = 25


def log(msg: str, fh=None) -> None:
    print(msg, flush=True)
    if fh is not None:
        fh.write(msg + "\n")
        fh.flush()


def hms(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def blocks_present(csv_path: Path) -> set[str]:
    """``block`` column values already in an existing measurements CSV."""
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as fh:
        return {r["block"] for r in csv.DictReader(fh) if r.get("block")}


def free_gb(path) -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def large_sizes() -> list[int]:
    """The sweep's own definition of the large size list.

    Imported rather than restated: one place decides what "10^3 to 10^7"
    means, and it is the script doing the measuring.
    """
    sys.path.insert(0, str(HERE))
    import run_sweep
    return list(run_sweep.LARGE_SIZES)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None,
                    help="measurements.csv to write (default: "
                         "results/large/measurements.csv)")
    ap.add_argument("--into-main", action="store_true",
                    help="write into results/ rather than results/large/, "
                         "making the large sweep the reference artefact")
    ap.add_argument("--workdir", default=None,
                    help="parent directory for scratch stores; put this on "
                         "a disk with room, the 10^7 competitors are "
                         "gigabytes")
    ap.add_argument("--blocks", default=None,
                    help="comma-separated subset of "
                         + ",".join(b for b, _, _ in BLOCKS))
    ap.add_argument("--geometries", default=None,
                    help="comma-separated subset, passed through to "
                         "run_sweep.py")
    ap.add_argument("--sizes", default=None,
                    help="override the swept sizes (default: run_sweep's "
                         "LARGE_SIZES, 10^3..10^7)")
    ap.add_argument("--resume", action="store_true",
                    help="skip blocks already present in the output CSV. A "
                         "block's rows are written all at once, so a killed "
                         "run never leaves a half-measured block behind")
    ap.add_argument("--fresh", action="store_true",
                    help="delete the output CSV first, rather than "
                         "appending to it")
    ap.add_argument("--figure", action="store_true",
                    help="run make_figure.py against the results when the "
                         "sweep finishes")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the commands, measure nothing")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter to run the sweep with (default: this "
                         "one). The reference environment is the one whose "
                         "package set environment.json records")
    args = ap.parse_args()

    default_dir = RESULTS if args.into_main else LARGE
    out = (Path(args.out).resolve() if args.out
           else default_dir / "measurements.csv")
    results = out.parent
    sizes = args.sizes or ",".join(str(n) for n in large_sizes())

    wanted = [b for b in BLOCKS]
    if args.blocks:
        keep = {b.strip() for b in args.blocks.split(",") if b.strip()}
        unknown = keep - {b for b, _, _ in BLOCKS}
        if unknown:
            ap.error(f"unknown block(s): {', '.join(sorted(unknown))}")
        wanted = [b for b in BLOCKS if b[0] in keep]

    if args.fresh and not args.dry_run:
        out.unlink(missing_ok=True)
        (out.parent / "environment.json").unlink(missing_ok=True)

    skipped = []
    if args.resume:
        if args.geometries:
            # The presence check reads the ``block`` column and nothing
            # else, so a block measured for one geometry looks done for
            # all of them.  Worth saying out loud rather than silently
            # skipping the geometries that were never measured.
            print("WARNING: --resume matches on block name only, so a block "
                  "already measured for some other --geometries will be "
                  "skipped rather than filled in.\n")
        done = blocks_present(out)
        still = []
        for name, produces, note in wanted:
            if done.issuperset(produces):
                skipped.append(name)
            else:
                still.append((name, produces, note))
        wanted = still

    scratch = args.workdir or "/tmp"
    avail = free_gb(scratch)

    print(f"sweep      {sizes}")
    print(f"out        {out}")
    print(f"scratch    {scratch}  ({avail:.0f} GB free)")
    print(f"python     {args.python}")
    if skipped:
        print(f"resuming   skipping {', '.join(skipped)} (already in the CSV)")
    print(f"blocks     {len(wanted)}")
    for name, _, note in wanted:
        print(f"  {name:<10s} {note}")
    print()

    if avail < MIN_FREE_GB:
        print(f"WARNING: {avail:.0f} GB free under {scratch}, "
              f"and the 10^7 competitors want ~{MIN_FREE_GB} GB of "
              f"headroom.  Pass --workdir to point somewhere with room.\n")

    if not wanted:
        print("nothing to do")
        return 0

    def command(name: str, first: bool) -> list[str]:
        cmd = [args.python, str(HERE / "run_sweep.py"),
               "--sizes", sizes, "--out", str(out), "--blocks", name]
        if args.workdir:
            cmd += ["--workdir", args.workdir]
        if args.geometries:
            cmd += ["--geometries", args.geometries]
        # The first block into a file that does not yet exist writes it;
        # everything after appends.  ``--replace`` rather than ``--append``
        # so that re-running a block over one already in the file
        # supersedes it instead of leaving two generations of the same
        # series behind -- which is what a resumed or repeated run does.
        if not (first and not out.exists()):
            cmd += ["--replace"]
        return cmd

    if args.dry_run:
        for i, (name, _, _) in enumerate(wanted):
            print(" ".join(command(name, i == 0)))
        return 0

    # After the dry-run return, so that printing the plan leaves nothing
    # behind -- an empty ``results/large/`` from a dry run reads as a
    # sweep that ran and produced nothing.
    results.mkdir(parents=True, exist_ok=True)

    logfile = results / "run_large.log"
    started = time.time()
    times: list[tuple[str, float]] = []
    failed: list[str] = []

    with open(logfile, "a") as fh:
        log(f"=== run_large {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"sizes={sizes} out={out} ===", fh)
        interrupted = False
        for i, (name, _, _) in enumerate(wanted):
            cmd = command(name, i == 0)
            log(f"\n--- block {i + 1}/{len(wanted)}: {name} "
                f"(elapsed {hms(time.time() - started)}) ---", fh)
            log("    " + " ".join(cmd), fh)
            t0 = time.time()
            try:
                proc = subprocess.run(cmd, cwd=str(HERE))
            except KeyboardInterrupt:
                # Ctrl-C reaches the child too, which dies before writing
                # its CSV -- so this block contributed nothing and
                # ``--resume`` will re-run it.  Say so, and still print
                # the summary for the blocks that did finish.
                dt = time.time() - t0
                log(f"--- {name} INTERRUPTED after {hms(dt)}; it wrote "
                    f"nothing and --resume will re-run it ---", fh)
                interrupted = True
                break
            dt = time.time() - t0
            times.append((name, dt))
            if proc.returncode != 0:
                failed.append(name)
                log(f"--- {name} FAILED (exit {proc.returncode}) after "
                    f"{hms(dt)}; continuing with the next block ---", fh)
            else:
                log(f"--- {name} done in {hms(dt)} ---", fh)

        total = time.time() - started
        log("\n=== summary ===", fh)
        for name, dt in times:
            log(f"  {name:<10s} {hms(dt):>9s}  "
                f"{100 * dt / max(total, 1e-9):5.1f}%", fh)
        log(f"  {'TOTAL':<10s} {hms(total):>9s}", fh)
        log(f"wrote {out}", fh)
        log(f"wrote {logfile}", fh)
        if failed:
            log(f"FAILED blocks: {', '.join(failed)}  "
                f"(re-run with --resume to retry only those)", fh)
        if interrupted:
            log("interrupted -- re-run the same command with --resume to "
                "continue from the block that did not finish", fh)

    if args.figure and not failed and not interrupted:
        fig = [args.python, str(HERE / "make_figure.py"),
               "--results", str(results)]
        print("\n" + " ".join(fig))
        subprocess.run(fig, cwd=str(HERE))

    return 1 if (failed or interrupted) else 0


if __name__ == "__main__":
    sys.exit(main())
