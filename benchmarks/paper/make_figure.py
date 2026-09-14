#!/usr/bin/env python
"""Turn ``results/measurements.csv`` into the paper figure and its
supplementary table.

    python make_figure.py

Reads only the CSV, so the figure can be restyled or regenerated without
re-running the sweep.  Outputs, all under ``results/``:

``panels/X.png`` / ``panels/X.pdf``          one file per panel
``panels/X_key.png`` / ``X_key.pdf``        that panel's key, on its own
``supplementary_table.md``                  every number behind it, plus
``supplementary_table.csv``                 what the panels have no room for

**Panels are written separately, not as one sheet, and so are their
keys.**  Composition -- which panel goes where, how many columns, what
the letters are, how much of the plot area identification is allowed to
cover -- is a layout decision that belongs to whoever is assembling the
figure, and a pre-composed sheet with legends burnt into the panels
forces all of it.  Each panel is therefore its own file at one journal
column (the storage bars take two) and each key is its own file beside
it, cropped to the legend and nothing else.  ``--keys`` draws them inside
the panels too, for reading one alone; ``--letters`` stamps A..G;
``--combined`` still writes the old single-sheet ``figure1.{png,pdf}``.

Times are plotted in **milliseconds**.  The operations compared run from
a 20 us seek-and-read to a 20 s mesh write, and milliseconds put the
middle of that range in whole numbers.  ``measurements.csv`` and the
supplementary table stay in seconds, which is what was measured.

The panels group into three questions.  The grouping is what the letters
mean and what ``--combined`` lays out; assembled another way, the panels
still answer these three.

    how big is it    A bytes per vertex        B size ratio vs N
    vs a single file C write all  D read all   E read one
    vs precomputed   F read a region (vs files)
                     G write all, read all, read one

Every panel is swept by **vertex** count, one point per decade from 10^3
to 10^6.  Objects move with it at a fixed conversion, because each
geometry's objects are a fixed size -- 1 vertex per point, 12 per
streamline, 144 per mesh patch -- and the key names that conversion, so
a point on the streamline curve at 10^5 vertices is also 8.7 k
streamlines.

Every line panel is log-log.  The **crossings** -- the size at which the
zarr-vectors curve overtakes its opponent, which is the number the
Results section is actually asking for -- are computed on the sizes the
two curves have in common and *printed* rather than drawn: a vertical
rule marking one pair's crossing ran through every other curve in the
panel, and these panels are read for their slopes.

C, D, E and G ask the same questions of different opponents, and that is
the point.  A single file has to read everything to answer anything, so
it loses any partial-read comparison eventually and the only question is
where.  G asks the harder one: precomputed annotations chunk the same
domain the same way and resolve one annotation from a ``by_id`` index in
a single file read, so nothing structural is being handicapped and what
is left is the layout.  Parquet, plain Zarr and HDF5 are measured on the
same grid and are in the supplementary table.

The region query appears once, in **F**.  Repeating it against the
chunked layouts would have said the same thing twice, so **G** takes the
bulk operations instead and **H** takes single-object access, which only
precomputed can answer by id.

Single-object replacement and the query-size sweep are still measured --
they are in ``measurements.csv`` and in the supplementary table -- but
they are not plotted.

Colour encodes geometry and line style encodes implementation
(solid + filled marker = zarr-vectors, dashed + open marker = the
single-file competitor), so the same key reads across every line panel.

The palette is one blue-to-magenta arc with five fixed slots, validated
as a set rather than chosen by eye: every pair -- not just neighbours --
clears the colour-blind separation bar (worst 9.2 dE, protanopia) and
every slot clears 3:1 contrast against white.  Colour is never the only
encoding regardless: marker fill and line style carry the
store-versus-alternative split, and panel B gives each geometry its own
marker.  In the panels that compare
implementations rather than geometries the same arc says which side is
which: blues are zarr-vectors, magentas are what it replaces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"

ZV = "zarr-vectors"
ZV_NONE = "zarr-vectors (no codec)"
ZV_ZSTD = "zarr-vectors (zstd)"
ZV_NOIDX = "zarr-vectors (zstd, no object index)"

# Vertices per object, per geometry -- mirrors ``_harness.VERTS_PER_OBJECT``
# and is what makes the object axis comparable across geometries.  Read
# from the CSV where possible (see ``verts_per_object``), these are the
# fallback for an older measurements file.
VERTS_PER_OBJECT = {"points": 1, "streamlines": 12, "skeletons": 200,
                    "graphs": 1, "meshes": 144}

GEOM_ORDER = ["points", "streamlines", "skeletons", "graphs", "meshes"]
GEOM_LABEL = {"points": "point cloud", "streamlines": "streamlines",
              "skeletons": "skeletons", "graphs": "graphs", "meshes": "meshes"}

# Categorical palette: one blue-to-magenta arc, five fixed slots.
# Validated as a set (lightness band, chroma floor, CVD separation across
# *all* pairs rather than neighbours only, normal-vision floor, and 3:1
# contrast against a white surface) -- the worst all-pairs CVD distance
# is 9.2 dE in protanopia, above the 8 dE bar, so identity survives
# colour-blind readers and greyscale printing without relying on the
# markers.  Deep blue and deep magenta are the two ends because they are
# the two series that appear together most: point cloud and streamlines.
GEOM_COLOR = {
    "points": "#0F4FA8",       # deep blue
    "meshes": "#4F97DE",       # azure
    "skeletons": "#A33BDE",    # violet
    "graphs": "#E85BB0",       # pink
    "streamlines": "#B00060",  # deep magenta
}

# Marker per geometry, for the panels where every line is the same
# implementation and the marker is therefore free to carry identity as
# well (panel B).  Elsewhere the marker encodes zarr-vectors vs the
# alternative and this is not used.
GEOM_MARKER = {"points": "o", "streamlines": "s", "skeletons": "D",
               "graphs": "^", "meshes": "v"}

# Ink, not series colour, for anything that is text.
INK = "#1F2933"
INK_MUTED = "#6B7280"

TIMED = ["points", "streamlines", "meshes"]

# Every timing measurement is recorded in seconds and plotted in
# milliseconds: the operations being compared run from ~20 us (a seek and
# a 12-byte read) to ~20 s (a million-vertex mesh write), and milliseconds
# put the interesting middle of that range in whole numbers rather than
# in four leading zeros.
MS = 1000.0
T_LABEL = "time (ms)"

mpl.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})


# --------------------------------------------------------------------
# Data shaping
# --------------------------------------------------------------------

def load():
    df = pd.read_csv(RESULTS / "measurements.csv")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["ci_hw"] = pd.to_numeric(df["ci_hw"], errors="coerce").fillna(0.0)
    # Both sweep axes, coerced up front: a blank anywhere in either
    # column would otherwise turn it into an object column and every
    # comparison and sort against it into a type error at plot time.
    for col in ("n_vertices", "n_objects", "n_target"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def is_zv(series):
    return series.str.startswith(ZV)


def best_competitor(storage: pd.DataFrame) -> pd.DataFrame:
    """Smallest *single-file* competitor per (geometry, N), raw and gzipped.

    Taking the *minimum* rather than a nominated format is the
    conservative choice: every geometry is compared against whichever
    single-file encoding happens to serve it best.
    """
    out = []
    for (geom, n), grp in storage.groupby(["geometry", "n_target"]):
        # Chunked formats are in the storage block too, but panel B's
        # denominator is "the single file you would otherwise have", so
        # they are excluded here and compared on their own line instead.
        comp = grp[~is_zv(grp["impl"])
                   & ~grp["detail"].astype(str).str.startswith("chunked")]
        raw = comp[~comp["impl"].str.contains(r"\(gzip\)")]
        gz = comp[comp["impl"].str.contains(r"\(gzip\)")]
        row = {"geometry": geom, "n_target": n,
               "n_vertices": grp["n_vertices"].iloc[0]}
        if len(raw):
            best = raw.loc[raw["value"].idxmin()]
            row["comp_raw"] = best["value"]
            row["comp_raw_label"] = best["impl"]
        if len(gz):
            best = gz.loc[gz["value"].idxmin()]
            row["comp_gz"] = best["value"]
            row["comp_gz_label"] = best["impl"]
        for label, key in ((ZV_NONE, "zv_none"), (ZV_ZSTD, "zv_zstd"),
                           (ZV_NOIDX, "zv_noidx")):
            hit = grp[grp["impl"] == label]
            if len(hit):
                row[key] = hit["value"].iloc[0]
        out.append(row)
    return pd.DataFrame(out)


def precomputed_ratio(storage):
    """zarr-vectors (Zstd) ÷ precomputed annotations, per dataset size."""
    rows = []
    for n, grp in storage[storage["geometry"] == "points"].groupby("n_target"):
        g = grp.set_index("impl")
        if ZV_ZSTD not in g.index or "precomputed annotations" not in g.index:
            continue
        rows.append({"n_vertices": g["n_vertices"].iloc[0],
                     "ratio": (g.loc[ZV_ZSTD, "value"]
                               / g.loc["precomputed annotations", "value"])})
    return pd.DataFrame(rows).sort_values("n_vertices") if rows else None


def series(df, block, geom, op, impl_is_zv, xkey="n_vertices"):
    sub = df[(df["block"] == block) & (df["geometry"] == geom)
             & (df["op"] == op)]
    sub = sub[is_zv(sub["impl"]) == impl_is_zv].sort_values(xkey)
    return sub


def _band(ax, x, y, hw, color, alpha):
    """Confidence band around a line, floored so it stays on the axis.

    Five repeats of a millisecond-scale call can produce a half-width
    wider than the mean, and ``y - hw`` is then negative -- impossible
    for a duration, and on a log axis it fills everything below the point
    down to the floor, which reads as a measurement rather than as an
    interval that ran out of samples.  The lower edge is clipped to a
    twentieth of the value: still visibly enormous, but bounded.
    """
    y, hw = np.asarray(y, float), np.asarray(hw, float)
    if not np.any(np.isfinite(hw) & (hw > 0)):
        return
    ax.fill_between(x, np.maximum(y - hw, y * 0.05), y + hw, color=color,
                    alpha=alpha, linewidth=0, zorder=2)


def _line(ax, x, y, hw, geom, zv, label=None):
    """One measured series: line, markers, and its uncertainty.

    Uncertainty is a filled band around the line, at low alpha so it
    reads as tolerance rather than as a second series.  On four points
    per line the band is an interpolation between four intervals, which
    is what a reader of a log-log trend line expects it to be.
    """
    color = GEOM_COLOR[geom]
    x, y = np.asarray(x, float), np.asarray(y, float)
    hw = np.asarray(hw, dtype=float)
    ax.plot(x, y, color=color, marker="o" if zv else "s",
            markersize=6.0, linewidth=2.25,
            linestyle="-" if zv else "--",
            markerfacecolor=color if zv else "white",
            markeredgecolor=color, markeredgewidth=1.5, label=label,
            zorder=3)
    _band(ax, x, y, hw, color, 0.16)


def _axes(ax, xlabel, ylabel, title, letter, xlog=True, ylog=True):
    """Scales, labels, grid and letter for one panel.

    Log on both axes is the default and what every panel here wants --
    the series span four decades of size and five of time.  ``_region_panel``
    can still be asked for linear scales via its ``linear`` flag, which
    nothing does by default.
    """
    ax.set_xscale("log" if xlog else "linear")
    ax.set_yscale("log" if ylog else "linear")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _log_grid(ax, xlog=xlog, ylog=ylog)
    _letter(ax, letter)


def crossover(x, y_zv, y_other):
    """Dataset size where the two curves cross, by log-log interpolation.

    Returns ``None`` when they never cross in the measured range -- one
    side won everywhere, which is a result rather than a missing value.
    Uses the *last* sign change, so a curve that wobbles across parity at
    small N still reports the size past which the answer is settled.
    """
    x = np.asarray(x, float)
    d = np.log10(np.asarray(y_zv, float)) - np.log10(np.asarray(y_other, float))
    ok = np.isfinite(d)
    if ok.sum() < 2:
        return None
    x, d = x[ok], d[ok]
    sign_change = np.flatnonzero(d[:-1] * d[1:] < 0)
    if not len(sign_change):
        return None
    i = sign_change[-1]
    t = d[i] / (d[i] - d[i + 1])
    return float(10 ** (np.log10(x[i]) + t * (np.log10(x[i + 1]) - np.log10(x[i]))))


def crossover_of(zv, other, xkey="n_vertices"):
    """Crossing of two measured series, on the sizes they have in common.

    The two sides of a comparison do not always cover the same sizes --
    the object sweep stops where a geometry's vertex budget runs out,
    and the point cloud is measured at sizes the chunked competitors are
    not -- so the curves are joined on the sizes both actually have
    rather than assumed to be index-aligned.  Fewer than two shared
    sizes is not a crossing, it is not enough data to look for one.
    """
    a = zv.drop_duplicates(subset=[xkey]).set_index(xkey)["value"]
    b = other.drop_duplicates(subset=[xkey]).set_index(xkey)["value"]
    shared = sorted(set(a.index) & set(b.index))
    if len(shared) < 2:
        return None
    return crossover(np.asarray(shared, float),
                     a.loc[shared].to_numpy(), b.loc[shared].to_numpy())


# Crossings are still computed -- they are the number the Results
# section argues from -- but they are no longer drawn.  A vertical rule
# per series crossed every curve in the panel to mark a point on one of
# them, and the panels are read for their slopes.  ``make_figure.py``
# prints the crossings instead, and they are in the supplementary table.
CROSSINGS = []


def _note_crossover(panel, label, xc):
    if xc is not None:
        CROSSINGS.append((panel, label, xc))


def _si(v):
    if v >= 1e6:
        return f"{v/1e6:.1f}M"
    if v >= 1e3:
        return f"{v/1e3:.0f}k"
    return f"{v:.0f}"


def _log_grid(ax, xlog=True, ylog=True):
    """Decade lines plus the 2..9 subdivisions between them.

    Without the minor lines a reader has to guess where a point sits
    inside a decade, which on a log axis is most of the information.  A
    linear axis gets evenly divided minor ticks instead, since log
    subdivisions on it would be nonsense.
    """
    ax.grid(True, which="major", alpha=0.28, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.4)
    log_sub = mpl.ticker.LogLocator(base=10.0,
                                    subs=tuple(np.arange(2, 10) * 0.1),
                                    numticks=100)
    for axis, is_log in ((ax.xaxis, xlog), (ax.yaxis, ylog)):
        axis.set_minor_locator(log_sub if is_log
                               else mpl.ticker.AutoMinorLocator(5))
        axis.set_minor_formatter(mpl.ticker.NullFormatter())


def _letter(ax, letter):
    """Panel letter at a constant offset from the axes corner.

    An axes-fraction offset would drift with panel width -- on the
    double-width storage panel it lands inside the plot, on a narrow one
    it lands in the neighbouring gap and looks like it labels that panel.

    ``None`` draws nothing, which is the default when panels are written
    as separate files: the letters belong to whatever layout they are
    assembled into, and stamping them here would fix an order the
    assembler has not chosen yet.
    """
    if letter is None:
        return
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(-34, 10), textcoords="offset points",
                fontsize=11, fontweight="bold", va="top", ha="left",
                color=INK)


# --------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------

# Competitor formats to show in panel A, per geometry, in bar order.
# Named explicitly rather than reduced to "the smallest one": the paper
# argues against particular formats -- .trk for streamlines, .obj and
# .stl for meshes -- so each has to be visible even where another
# encoding of the same geometry happens to be smaller.
PANEL_A_FORMATS = {
    # Each geometry gets its canonical format, that format compressed,
    # and where it exists a third encoding worth naming: precomputed for
    # points, TRX's own deflate for streamlines, OBJ against STL.
    # Both precomputed variants are listed because the full one (with its
    # ``by_id`` index) is not measured at the top size -- one file per
    # annotation does not scale there -- and the spatial-only variant is,
    # so the format is still named wherever the panel lands.
    "points": ["PLY (gzip)", "precomputed annotations",
               "precomputed (spatial only)", "Parquet"],
    "streamlines": ["TRK", "TRK (gzip)", "TRX deflate"],
    "skeletons": ["SWC", "SWC (gzip)"],
    "graphs": ["CSV edge-list", "CSV edge-list (gzip)"],
    "meshes": ["STL", "OBJ", "OBJ (gzip)"],
}
# Panel A encodes implementation, not geometry: blues are the store,
# magentas are the file it replaces, light is uncompressed and dark is
# compressed.  The reader can tell which side of the argument a bar is on
# before reading the key.
ZV_BARS = [(ZV_NONE, "zarr-vectors, no codec", "#9EC4EE"),
           (ZV_ZSTD, "zarr-vectors, Zstd", "#0F4FA8")]
COMP_RAW, COMP_GZ = "#F0A0CE", "#B00060"


# Long implementation names do not fit a rotated bar label at column
# width; these are what the bars are called in the panel.
BAR_NAME = {"precomputed annotations": "precomputed",
            "precomputed (spatial only)": "precomputed (spatial)",
            "precomputed (spatial only) (gzip)": "precomputed (spatial) gz"}


def _bar_name(impl):
    return BAR_NAME.get(impl,
                        impl.replace(" edge-list", "").replace(" (gzip)", " gz"))


def panel_storage_bars(ax, storage, best, env, letter=None, key=True):
    """A -- bytes per vertex at the largest size measured."""
    n_max = int(storage["n_target"].max())
    sub = storage[storage["n_target"] == n_max]
    geoms = [g for g in GEOM_ORDER if g in set(sub["geometry"])]

    rows = {}
    for g in geoms:
        grp = sub[sub["geometry"] == g].set_index("impl")
        nv = float(grp["n_vertices"].iloc[0])
        bars = []
        for impl, label, color in ZV_BARS:
            if impl in grp.index:
                bars.append((grp.loc[impl, "value"] / nv, label, color, None))
        for impl in PANEL_A_FORMATS.get(g, []):
            if impl not in grp.index:
                continue
            gz = "(gzip)" in impl
            bars.append((grp.loc[impl, "value"] / nv,
                         "single file, gzip" if gz else "single file",
                         COMP_GZ if gz else COMP_RAW, _bar_name(impl)))
        rows[g] = bars

    pitch = max(len(b) for b in rows.values())
    w = 0.86 / pitch
    legend_of = {}
    for i, g in enumerate(geoms):
        bars = rows[g]
        left = i - (len(bars) - 1) / 2 * w
        for j, (val, label, color, name) in enumerate(bars):
            x = left + j * w
            ax.bar(x, val, width=w * 0.92, color=color, edgecolor="white",
                   linewidth=0.4)
            legend_of.setdefault(label, color)
            if name:
                ax.text(x, val, name, ha="center", va="bottom", fontsize=5.8,
                        rotation=90, color=INK_MUTED)

    ax.set_xticks(range(len(geoms)))
    ax.set_xticklabels([GEOM_LABEL[g] for g in geoms], rotation=20, ha="right")
    ax.set_ylabel("bytes per vertex")
    ax.set_title(f"Storage per vertex  (N \u2248 {n_max:,})")
    ax.set_ylim(0, max(v for b in rows.values() for v, *_ in b) * 1.22)
    # Fixed order: the two store configurations, then raw file, then
    # compressed file -- so the key reads the same way every rebuild
    # regardless of which format each geometry happens to list first.
    order = [lbl for _, lbl, _ in ZV_BARS] + ["single file",
                                              "single file, gzip"]
    handles = [Patch(facecolor=legend_of[lbl], label=lbl)
               for lbl in order if lbl in legend_of]
    if key:
        ax.legend(handles=handles, loc="upper left", frameon=False,
                  handlelength=1.2, fontsize=6.8)
    ax.grid(True, which="major", axis="y", alpha=0.25)
    ax.grid(True, which="minor", axis="y", alpha=0.12)
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", bottom=False)
    _letter(ax, letter)
    return handles


def panel_size_ratio(ax, best, letter=None, key=True):
    """B -- when does the store become smaller than the file?"""
    for g in GEOM_ORDER:
        sub = best[(best["geometry"] == g)].dropna(
            subset=["zv_zstd", "comp_raw"]).sort_values("n_vertices")
        if not len(sub):
            continue
        # Every line here is the same implementation, so the marker is
        # free to carry identity alongside the colour.
        ax.plot(sub["n_vertices"], sub["zv_zstd"] / sub["comp_raw"],
                marker=GEOM_MARKER[g], markersize=5.25, linewidth=2.1,
                color=GEOM_COLOR[g], label=GEOM_LABEL[g], zorder=3)
    # Parity, unlabelled: a thick grey rule behind the curves, so where a
    # geometry crosses from "bigger than the file" to "smaller" is read
    # off the picture rather than off the axis.
    ax.axhline(1.0, color="0.62", linewidth=2.6, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("vertices")
    ax.set_ylabel("store ÷ best single file")
    ax.set_title("Size ratio vs dataset size")
    # Panel B builds its own axes rather than going through ``_axes``,
    # so it has to ask for the decade subdivisions explicitly -- without
    # them a reader has to guess where a ratio sits inside a decade,
    # which on a log axis is most of the information.
    _log_grid(ax)
    handles = ax.get_legend_handles_labels()[0]
    if key:
        # Panel B is the one panel whose key does not go top left: its
        # curves converge on the right and leave that corner occupied.
        leg = ax.legend(loc="best", frameon=True, framealpha=0.88,
                        handlelength=1.4, ncol=1, fontsize=6.4,
                        labelcolor=INK, borderpad=0.32, labelspacing=0.26)
        leg.get_frame().set_edgecolor("none")
    _letter(ax, letter)
    return handles


def panel_lines(ax, df, block, op, geoms, title, letter, ylabel=T_LABEL,
                opponent_block=None, opponent_impl=None,
                xkey="n_vertices", xlabel="vertices"):
    """One op, one panel: zarr-vectors against its opponent, per geometry.

    ``opponent_block`` / ``opponent_impl`` let row 3 draw the same
    zarr-vectors curves against precomputed instead of the single file,
    so the two rows are directly comparable.  ``xkey`` picks the sweep
    variable: the object-swept panels plot ``n_objects``, the rest plot
    ``n_vertices``.
    """
    for row, g in enumerate(geoms):
        zv = series(df, block, g, op, True, xkey)
        if opponent_impl is None:
            other = series(df, block, g, op, False, xkey)
        else:
            sub = df[(df["block"] == (opponent_block or block))
                     & (df["geometry"] == g) & (df["op"] == op)
                     & (df["impl"] == opponent_impl)]
            other = sub.sort_values(xkey)
        for s_, is_zv in ((zv, True), (other, False)):
            if not len(s_):
                continue
            _line(ax, s_[xkey].to_numpy(), s_["value"].to_numpy() * MS,
                  s_["ci_hw"].to_numpy() * MS, g, is_zv)
        if len(zv) and len(other):
            _note_crossover(title, GEOM_LABEL[g],
                            crossover_of(zv, other, xkey))
    _axes(ax, xlabel, ylabel, title, letter)


# Same five validated slots, re-used for the chunked-format comparison.
# zarr-vectors keeps the blue it has in every other panel so the eye
# carries it across; precomputed takes the far end of the arc because it
# is the opponent the row is really about.
CHUNKED_STYLE = {
    "zarr-vectors": ("#0F4FA8", "o", "-"),
    "plain Zarr": ("#4F97DE", "D", "-"),
    "precomputed annotations": ("#B00060", "s", "--"),
    "Parquet": ("#A33BDE", "^", "--"),
    "HDF5": ("#E85BB0", "v", "--"),
}


# Which chunked layouts the panels draw.  All four competitors are
# measured and all four are in the supplementary table; the panel plots
# the two that are alternative *layouts for vector data* -- precomputed,
# which chunks spatially exactly as zarr-vectors does, and Parquet, the
# columnar file a table of points would otherwise live in.  Plain Zarr
# and HDF5 are the same dense-array container underneath the same manual
# grid, so on the panel they were two more lines making the same point.
CHUNKED_PLOTTED = ("zarr-vectors", "precomputed annotations", "Parquet")


def _chunked_curves(df, op, only=CHUNKED_PLOTTED):
    """zarr-vectors plus the chunked competitors ``only`` names."""
    block = {"write_full": "bulk", "read_full": "bulk",
             "fetch_one": "object"}[op]
    zv = df[(df["block"] == block) & (df["geometry"] == "points")
            & (df["op"] == op) & is_zv(df["impl"])].sort_values("n_vertices")
    out = {"zarr-vectors": zv} if len(zv) else {}
    ch = df[(df["block"] == "chunked") & (df["op"] == op)]
    for fmt in CHUNKED_STYLE:
        if fmt == "zarr-vectors" or (only is not None and fmt not in only):
            continue
        sub = ch[ch["impl"] == fmt].sort_values("n_vertices")
        if len(sub):
            out[fmt] = sub
    return out


def _draw_chunked(ax, curves, *, dashed, label_fmt=True):
    for label, sub in curves.items():
        color, marker, _ = CHUNKED_STYLE[label]
        x, y = sub["n_vertices"].to_numpy(), sub["value"].to_numpy()
        hw = sub["ci_hw"].to_numpy()
        ax.plot(x, y, color=color, marker=marker, markersize=5.7,
                linewidth=2.25, linestyle="--" if dashed else "-",
                label=label if label_fmt else None,
                markerfacecolor="white" if dashed else color,
                markeredgecolor=color, markeredgewidth=1.5, zorder=3)
        _band(ax, x, y, hw, color, 0.14)


# The three operations panel G draws, in legend order.  Line style and
# marker carry the operation; colour carries the implementation.  Nothing
# here is dotted, because a dotted vertical line is the crossover mark.
VS_PC_OPS = [
    ("write_full", "write all", "--", "o", False),
    ("read_full", "read all", "-", "s", True),
    ("fetch_one", "read one", "-.", "^", True),
]
VS_PC_IMPL = [("zarr-vectors", GEOM_COLOR["points"]),
              ("precomputed annotations", GEOM_COLOR["streamlines"])]


def panel_vs_precomputed(ax, df, letter="G", key=True):
    """G -- the same three operations, against precomputed annotations.

    Precomputed is the comparison that matters here because it is built
    on the same idea: it chunks the same domain the same way, and its
    ``by_id`` index resolves one annotation in a single file read.  So
    the three questions the single-file panels ask -- write everything,
    read everything, read one -- get asked again of a layout that has no
    structural disadvantage, and what is left is the layout itself.
    Parquet, plain Zarr and HDF5 are measured too and are in the
    supplementary table; two of them cannot answer "read one" at all.

    Colour is the implementation and line style is the operation, the
    reverse of the single-file panels, because both sides here are
    chunked stores and neither is "the alternative encoding".
    """
    for row, (op, op_label, ls, marker, filled) in enumerate(VS_PC_OPS):
        curves = _chunked_curves(df, op, only=("zarr-vectors",
                                               "precomputed annotations"))
        for impl, color in VS_PC_IMPL:
            sub = curves.get(impl)
            if sub is None or not len(sub):
                continue
            x = sub["n_vertices"].to_numpy()
            y = sub["value"].to_numpy() * MS
            hw = sub["ci_hw"].to_numpy() * MS
            ax.plot(x, y, color=color, marker=marker, markersize=5.7,
                    linewidth=2.25, linestyle=ls,
                    markerfacecolor=color if filled else "white",
                    markeredgecolor=color, markeredgewidth=1.5, zorder=3)
            _band(ax, x, y, hw, color, 0.14)
        zv, pc = curves.get("zarr-vectors"), curves.get(
            "precomputed annotations")
        if zv is not None and pc is not None and len(zv) and len(pc):
            _note_crossover("vs precomputed", op_label,
                            crossover_of(zv, pc))

    handles = [Line2D([], [], color=c, linewidth=2.4,
                      label="zarr-vectors" if impl == "zarr-vectors"
                      else "precomputed")
               for impl, c in VS_PC_IMPL]
    handles += [Line2D([], [], color="0.35", linestyle=ls, marker=m,
                       markersize=5.4, linewidth=2.1, label=lbl,
                       markerfacecolor="0.35" if filled else "white",
                       markeredgecolor="0.35")
                for _op, lbl, ls, m, filled in VS_PC_OPS]
    if key:
        _draw_key(ax, handles)
    _axes(ax, "vertices", T_LABEL,
            "Write, read, read one\nvs precomputed annotations", letter)
    return handles


def _volume_axis(ax, curves, *, per_geometry, log=True):
    """Right-hand axis: the volume of the queried box, as % of domain.

    ``curves`` is ``[(colour, x, percent), ...]``.  Drawn behind the
    timing curves and in muted ink, because it is the context that makes
    them readable rather than a series competing with them: without it a
    falling query-time curve looks like a speedup when it is really the
    same work over a smaller box.
    """
    ax2 = ax.twinx()
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)
    lo = 100.0
    for color, x, y in curves:
        lo = min(lo, float(np.nanmin(y)))
        ax2.fill_between(x, np.full_like(y, 1e-9), y, color=INK_MUTED,
                         alpha=0.055 if per_geometry else 0.09, linewidth=0)
        ax2.plot(x, y, color=color, linewidth=1.4 if not per_geometry else 1.0,
                 alpha=0.5, linestyle="-")
    if log:
        ax2.set_yscale("log")
        ax2.set_ylim(lo / 3.0, 140.0)
    else:
        ax2.set_ylim(0.0, 105.0)
    ax2.set_ylabel("box volume  (% of domain)", color=INK_MUTED, fontsize=7.5)
    ax2.tick_params(axis="y", colors=INK_MUTED, labelsize=6.5)
    ax2.spines["right"].set_color(INK_MUTED)
    ax2.grid(False)
    return ax2


def _region_panel(ax, df, block, title, letter, key, *, per_geometry,
                  linear=False):
    """The region-query panel, in its two variants.

    ``per_geometry`` says whether the box was sized per geometry (~100
    objects each, so three volume curves) or once for all of them (~100
    vertices, so one).

    Both y axes are logarithmic: query time on the left and box volume on
    the right.  They have to be, now that the sweep runs to 10^7 -- the
    times on one panel span from 2.7 ms to 55 s, four and a half decades,
    and on a linear axis everything below the mesh series is flattened
    onto the x axis.  The right axis spans as far: sizing the box to ~100
    objects takes it from the whole domain down to about 0.0014 % of it.
    ``linear=True`` restores the old linear pair for both, which is
    readable only over a narrow range of N.

    Size is logarithmic either way -- the sweep is a decade apart per
    point, and a linear size axis stacks four of the five measurements
    against the left spine.
    """
    sub = df[df["block"] == block]
    vol = []
    for g in TIMED:
        v = sub[(sub["geometry"] == g) & is_zv(sub["impl"])].sort_values(
            "n_vertices")
        frac = pd.to_numeric(v["detail"], errors="coerce")
        if not len(v) or frac.isna().all():
            continue
        vol.append((GEOM_COLOR[g] if per_geometry else INK_MUTED,
                    v["n_vertices"].to_numpy(float),
                    frac.to_numpy(float) * 100.0))
    if vol and not per_geometry:
        vol = vol[:1]          # one box for every geometry: draw it once
    if vol:
        _volume_axis(ax, vol, per_geometry=per_geometry, log=not linear)

    for g in TIMED:
        curves = {}
        for zv in (True, False):
            s_ = sub[(sub["geometry"] == g)
                     & (is_zv(sub["impl"]) == zv)].sort_values("n_vertices")
            if not len(s_):
                continue
            curves[zv] = s_
            _line(ax, s_["n_vertices"].to_numpy(),
                  s_["value"].to_numpy() * MS, s_["ci_hw"].to_numpy() * MS,
                  g, zv)
        if len(curves) == 2:
            _note_crossover(title, GEOM_LABEL[g],
                            crossover_of(curves[True], curves[False]))
    _axes(ax, "vertices", T_LABEL, title, letter, ylog=not linear)
    handles = _series_handles(TIMED, extra=[
        Line2D([], [], color=INK_MUTED, linewidth=1.4, alpha=0.6,
               label="box volume (right axis)")])
    if key:
        _draw_key(ax, handles)
    return handles


def panel_fixed(ax, df, letter="F", key=False):
    """F -- constant-size *answer*, growing haystack.

    The box is resized at every N to hold about 100 objects, so the
    answer stays the same size while the volume it is cut from shrinks --
    from the whole domain down to 0.01 % of it.  A chunked store should
    track the answer and a single file should track the haystack, and the
    two axes together are that prediction and its test in one frame.

    The cost is that "100 objects" is a different box per geometry -- 100
    mesh patches is 14 400 vertices and 100 points is 100 -- so the three
    series are each answering a differently sized question.
    ``panel_equal_box`` is the same measurement with that removed.
    """
    return _region_panel(ax, df, "spatial_fixed",
                         "Read one region\n(box sized to ~100 objects)",
                         letter, key, per_geometry=True)


def panel_equal_box(ax, df, letter="F2", key=False):
    """F, alternative -- one box volume for all three geometries.

    Sized to ~100 *vertices* rather than ~100 objects, so at every N the
    three geometries are queried with the same box and expect the same
    payload: every geometry spreads N vertices over the same domain, so a
    box holding 100 point-cloud vertices holds about 100 streamline or
    mesh vertices too.  What differs between the series is then only how
    those vertices are grouped, which is the thing being compared -- and
    the volume is a single curve rather than one per geometry.
    """
    return _region_panel(ax, df, "spatial_volume",
                         "Read one region\n(equal box, ~100 vertices)",
                         letter, key, per_geometry=False)


def panel_chunks(ax, df, letter="H", key=False):
    """H -- what the partition itself costs, dataset held fixed.

    Every other panel runs on one grid: 200-unit chunks over a 1000-unit
    domain, 125 of them, whatever the dataset size.  This one holds the
    dataset at 10^6 vertices and moves the grid instead, from 8 chunks to
    8000, and fetches one object out of each.  zarr-vectors only -- there
    is no competitor here, because a single file has no partition to
    vary; the comparison is against the store's own other configurations.

    Read it as the shape of a trade-off, not as a recommendation: coarse
    chunks make a fetch read more than it needs, fine chunks make it pay
    per-chunk metadata, and where the minimum sits depends on the
    geometry -- which is the point of drawing all three.
    """
    sub = df[(df["block"] == "chunks") & (df["op"] == "fetch_one")]
    for g in TIMED:
        s_ = sub[sub["geometry"] == g].sort_values("n_target")
        if not len(s_):
            continue
        _line(ax, s_["n_target"].to_numpy(), s_["value"].to_numpy() * MS,
              s_["ci_hw"].to_numpy() * MS, g, True)
    n = sub["n_vertices"].max()
    _axes(ax, "chunks over the domain", T_LABEL,
            f"Read one object vs chunk count\n(N = {_si(n)} vertices)",
            letter)
    handles = [Line2D([], [], color=GEOM_COLOR[g], marker="o", markersize=5.1,
                      linewidth=2.1, markerfacecolor=GEOM_COLOR[g],
                      markeredgecolor=GEOM_COLOR[g], label=GEOM_LABEL[g])
               for g in TIMED]
    if key:
        _draw_key(ax, handles)
    return handles


def panel_fragments(ax, df, letter="H2", key=False):
    """H, alternative -- the same grid sweep, but for objects that span.

    H fetches the generator's own objects, which are small enough to sit
    inside one cell at any grid it sweeps -- a 12-vertex streamline is
    ~32 units across against chunks of 50 to 500 -- so its curves are
    about how much payload a chunk carries, not about fragmentation.
    This panel sweeps the same grid over the same 10^6-vertex datasets
    and fetches a planted probe instead: one point, one 1024-vertex
    streamline running across the domain, and one 2304-vertex patch
    spanning 600 units.

    The point is the control: one chunk at every grid, so its line is the
    per-chunk floor and falls as finer chunks bring less of everything
    else with them.  The streamline and the patch occupy more cells the
    finer the grid gets -- 8 to 98 and 8 to 247 respectively -- and their
    lines are what reassembling one object out of that many costs.  The
    span and the vertices actually read are on every row of the CSV.
    """
    sub = df[df["block"] == "fragments"]
    for g in TIMED:
        s_ = sub[sub["geometry"] == g].sort_values("n_target")
        if not len(s_):
            continue
        _line(ax, s_["n_target"].to_numpy(), s_["value"].to_numpy() * MS,
              s_["ci_hw"].to_numpy() * MS, g, True)
    _axes(ax, "chunks over the domain", T_LABEL,
          "Read one large object vs chunk count\n(N = 1M vertices)", letter)
    handles = [Line2D([], [], color=GEOM_COLOR[g], marker="o", markersize=5.1,
                      linewidth=2.1, markerfacecolor=GEOM_COLOR[g],
                      markeredgecolor=GEOM_COLOR[g], label=GEOM_LABEL[g])
               for g in TIMED]
    if key:
        _draw_key(ax, handles)
    return handles


def _per_object(df, geom):
    """``"12 vertices/object"`` -- measured, not assumed.

    Taken from the rows themselves so the key cannot drift from what was
    generated: the streamline generator draws 8 to 15 vertices per
    object, so the honest number is the median of what it produced
    rather than the nominal 12 it was asked for.
    """
    sub = df[(df["geometry"] == geom) & (df["n_objects"] > 0)]
    if not len(sub):
        v = VERTS_PER_OBJECT.get(geom, 1)
    else:
        v = float((sub["n_vertices"] / sub["n_objects"]).median())
    v = round(v, 1)
    if v == int(v):
        v = int(v)
    return f"{v} vertex/object" if v == 1 else f"{v} vertices/object"


def _series_handles(geoms, *, alt_label="the alternative", extra=None):
    """The key entries a geometry panel needs: colours, then line styles."""
    handles = [Line2D([], [], color=GEOM_COLOR[g], marker="o", markersize=5.1,
                      linewidth=2.1, markerfacecolor=GEOM_COLOR[g],
                      markeredgecolor=GEOM_COLOR[g], label=GEOM_LABEL[g])
               for g in geoms]
    handles += [
        Line2D([], [], color="0.35", linestyle="-", marker="o", markersize=5.1,
               linewidth=2.1, markerfacecolor="0.35", markeredgecolor="0.35",
               label="zarr-vectors"),
        Line2D([], [], color="0.35", linestyle="--", marker="s",
               linewidth=2.1, markersize=5.1, markerfacecolor="white",
               markeredgecolor="0.35", label=alt_label),
    ]
    return handles + list(extra or ())


def _draw_key(ax, handles, *, loc="upper left", ncol=1):
    """Put a key inside the panel -- only when asked for."""
    leg = ax.legend(handles=handles, loc=loc, frameon=True, framealpha=0.88,
                    fontsize=5.9, handlelength=1.7, borderpad=0.32,
                    labelspacing=0.26, ncol=ncol, labelcolor=INK)
    leg.get_frame().set_edgecolor("none")
    return leg


# --------------------------------------------------------------------
# Panels as separate files
# --------------------------------------------------------------------

# One journal column for a line panel; the storage bars need three
# geometries x up to five bars and get two.
SIZE_LINE = (3.7, 3.1)
SIZE_WIDE = (7.6, 3.3)


def _ctx(df, env):
    storage = df[df["block"] == "storage"]
    return {"df": df, "env": env, "storage": storage,
            "best": best_competitor(storage),
            "pc_ratio": precomputed_ratio(storage)}


def _draw_a(ax, c, letter=None, key=True):
    return panel_storage_bars(ax, c["storage"], c["best"], c["env"],
                              letter=letter, key=key)


def _draw_b(ax, c, letter=None, key=True):
    return panel_size_ratio(ax, c["best"], letter=letter, key=key)


# Every panel carries its own key, top left, so it reads on its own;
# ``key.png``/``key.pdf`` is the same key as a standalone asset for an
# assembled figure, and ``--no-keys`` drops the in-panel copies.  Panel B
# is the exception on placement: its curves converge on the right and
# leave the top left occupied.
def _draw_c(ax, c, letter=None, key=True):
    panel_lines(ax, c["df"], "bulk", "write_full", TIMED,
                "Write the whole dataset", letter)
    handles = _series_handles(TIMED)
    if key:
        _draw_key(ax, handles)
    return handles


def _draw_d(ax, c, letter=None, key=True):
    panel_lines(ax, c["df"], "bulk", "read_full", TIMED,
                "Read the whole dataset", letter)
    handles = _series_handles(TIMED)
    if key:
        _draw_key(ax, handles)
    return handles


def _draw_e(ax, c, letter=None, key=True):
    panel_lines(ax, c["df"], "object", "fetch_one", TIMED,
                "Read one object", letter)
    handles = _series_handles(TIMED)
    if key:
        _draw_key(ax, handles)
    return handles


def _draw_f(ax, c, letter=None, key=True):
    return panel_fixed(ax, c["df"], letter=letter, key=key)


def _draw_f2(ax, c, letter=None, key=True):
    return panel_equal_box(ax, c["df"], letter=letter, key=key)


def _draw_h(ax, c, letter=None, key=True):
    return panel_chunks(ax, c["df"], letter=letter, key=key)


def _draw_h2(ax, c, letter=None, key=True):
    return panel_fragments(ax, c["df"], letter=letter, key=key)


def _draw_g(ax, c, letter=None, key=True):
    return panel_vs_precomputed(ax, c["df"], letter, key=key)


# (letter, filename stem, figure size, draw function).  The letter is the
# position this panel held in the assembled figure and is written only
# when asked for; the stem is what the file is called either way.
PANELS = [
    ("A", "A_storage_per_vertex", SIZE_WIDE, _draw_a),
    ("B", "B_size_ratio", SIZE_LINE, _draw_b),
    ("C", "C_write_all", SIZE_LINE, _draw_c),
    ("D", "D_read_all", SIZE_LINE, _draw_d),
    ("E", "E_read_one", SIZE_LINE, _draw_e),
    ("F", "F_read_region", SIZE_LINE, _draw_f),
    ("G", "G_vs_precomputed", SIZE_LINE, _draw_g),
    # The alternative F: same measurement, one box volume for all three
    # geometries.  Written alongside F rather than instead of it, because
    # which one belongs in the figure depends on whether the question is
    # "fetch a fixed number of objects" or "read a fixed region".
    ("F", "F2_read_region_equal_box", SIZE_LINE, _draw_f2),
    ("H", "H_read_one_vs_chunks", SIZE_LINE, _draw_h),
    # The alternative H: the grid held fixed and the object varied, which
    # is the question "does crossing chunk boundaries cost?" -- H itself
    # cannot answer it, because its objects fit inside one chunk.
    ("H", "H2_read_one_vs_fragments", SIZE_LINE, _draw_h2),
]


def render_panels(df, env, outdir, formats=("png", "pdf"), letters=False,
                  keys=False):
    """Write every panel as its own file, each with its key beside it.

    Panel ``X`` produces ``X.png``/``X.pdf`` and ``X_key.png``/``X_key.pdf``.
    Keeping the key out of the panel is what makes the panel placeable:
    an in-panel legend fixes how much of the plot area is spent on
    identification and where, decisions that belong to the layout the
    panels end up in.  ``keys=True`` draws it inside as well, for reading
    one panel on its own.
    """
    c = _ctx(df, env)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for letter, stem, size, draw in PANELS:
        fig, ax = plt.subplots(figsize=size)
        handles = draw(ax, c, letter=letter if letters else None, key=keys)
        fig.tight_layout(pad=0.35)
        for fmt in formats:
            path = outdir / f"{stem}.{fmt}"
            fig.savefig(path, bbox_inches="tight")
            written.append(path)
        plt.close(fig)
        written += _write_key(handles, outdir, f"{stem}_key", formats)
    return written


def _write_key(handles, outdir, stem, formats):
    """One key, cropped to itself, as its own file.

    Sized from the entry count and saved with a tight bounding box, so
    the result is the legend and nothing else -- a graphic to place, not
    a figure with a legend in the middle of it.
    """
    if not handles:
        return []
    fig = plt.figure(figsize=(2.4, 0.19 * len(handles) + 0.12))
    leg = fig.legend(handles=handles, loc="center", frameon=False,
                     fontsize=7.0, handlelength=2.6, labelspacing=0.42,
                     borderpad=0.2, labelcolor=INK)
    leg.get_frame().set_edgecolor("none")
    out = []
    for fmt in formats:
        path = outdir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", transparent=(fmt == "pdf"))
        out.append(path)
    plt.close(fig)
    return out


def build_figure(df, env):
    """Every panel in one sheet -- kept for `--combined`."""
    c = _ctx(df, env)

    # Six columns so a panel can be one or two standard widths.  Storage
    # carries five geometries x up to five bars and needs the room; the
    # rest are single curves and do not.
    fig = plt.figure(figsize=(13.5, 10.5))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1], hspace=0.46,
                          wspace=0.75)
    axes = [fig.add_subplot(gs[0, 0:4]), fig.add_subplot(gs[0, 4:6])]
    axes += [fig.add_subplot(gs[1, col:col + 2]) for col in (0, 2, 4)]
    axes += [fig.add_subplot(gs[2, col:col + 2]) for col in (0, 2, 4)]
    # The sheet has eight slots and the panel list decides how many are
    # used; drop the rest rather than leaving empty framed boxes.
    for ax in axes[len(PANELS):]:
        fig.delaxes(ax)
    axes = axes[:len(PANELS)]

    for ax, (letter, _stem, _size, draw) in zip(axes, PANELS):
        # No per-panel key in the sheet: one shared legend below serves
        # them all, which is the whole reason the sheet is denser than
        # the separate panels.
        draw(ax, c, letter=letter, key=False)
    for ax in axes[6:]:
        ax.set_facecolor("#FBFAFD")

    handles = [Patch(facecolor=GEOM_COLOR[g],
                     label=f"{GEOM_LABEL[g]} ({_per_object(df, g)})")
               for g in TIMED]
    handles += [
        Line2D([], [], color="0.25", linestyle="-", marker="o", markersize=4,
               markerfacecolor="0.25", markeredgecolor="0.25",
               label="zarr-vectors"),
        Line2D([], [], color="0.25", linestyle="--", marker="s", markersize=4,
               markerfacecolor="white", markeredgecolor="0.25",
               label="the alternative"),
        Line2D([], [], color="0.4", linestyle=":", linewidth=1.2,
               label="crossover: zarr-vectors wins to the right"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.005), labelcolor=INK)
    fig.suptitle(
        "At what dataset size does zarr-vectors overtake the alternative?",
        fontsize=11.5, y=0.997, color=INK)
    for ax, text in zip((axes[0], axes[2], axes[5]),
                        ("size", "vs one file", "vs chunked")):
        ax.annotate(text, xy=(0, 0.5), xycoords="axes fraction",
                    xytext=(-52, 0), textcoords="offset points",
                    rotation=90, ha="center", va="center", fontsize=9.5,
                    color=INK, fontweight="bold")
    return fig


# --------------------------------------------------------------------
# Supplementary table
# --------------------------------------------------------------------

OP_LABEL = {
    "write_full": "write whole dataset (s)",
    "read_full": "read whole dataset (s)",
    "fetch_one": "fetch one object (s)",
    "replace_one": "replace one object (s)",
    "bbox_query": "bounding-box query (s)",
}


def supplementary(df) -> pd.DataFrame:
    """One tidy table: every figure number, plus the variants it omits."""
    rows = []

    storage = df[df["block"] == "storage"]
    for (geom, n), grp in storage.groupby(["geometry", "n_target"]):
        nv = int(grp["n_vertices"].iloc[0])
        zv = grp[grp["impl"] == ZV_ZSTD]["value"]
        ref = float(zv.iloc[0]) if len(zv) else np.nan
        for _, r in grp.sort_values("value").iterrows():
            rows.append({
                "geometry": GEOM_LABEL[geom], "vertices": nv,
                "objects": int(r["n_objects"]),
                "measurement": "on-disk size (bytes)",
                "implementation": r["impl"],
                "value": f"{int(r['value']):,}",
                "per_vertex": round(r["value"] / nv, 2),
                "ratio_to_zarr_vectors": (round(r["value"] / ref, 3)
                                          if ref == ref else ""),
                "runs": 1,
                "note": "" if pd.isna(r["detail"]) else str(r["detail"]),
            })

    fid = df[df["block"] == "fidelity"]
    for _, r in fid.iterrows():
        rows.append({
            "geometry": GEOM_LABEL[r["geometry"]],
            "vertices": int(r["n_vertices"]), "objects": int(r["n_objects"]),
            "measurement": "round-trip max |Δ| (store units)",
            "implementation": r["impl"],
            "value": ("exact" if r["value"] == 0 else f"{r['value']:.3g}"),
            "per_vertex": "", "ratio_to_zarr_vectors": "", "runs": 1,
            "note": "" if pd.isna(r["detail"]) else str(r["detail"]),
        })

    timing = df[df["block"].isin(["bulk", "object", "spatial_fixed",
                                  "spatial_fraction", "spatial_volume",
                                  "chunks", "fragments"])].copy()
    # The spatial blocks hold several measurements per (geometry, N, op) --
    # one per box size -- so the box size has to join the grouping key or
    # every ratio in those blocks would be taken against the first box.
    timing["_key"] = np.where(
        timing["block"].isin(["spatial_fraction", "spatial_fixed",
                              "spatial_volume", "chunks", "fragments"]),
        timing["detail"].astype(str), "")
    for (block, geom, n, op, _), grp in timing.groupby(
            ["block", "geometry", "n_target", "op", "_key"]):
        zvr = grp[is_zv(grp["impl"])]
        ref = float(zvr["value"].iloc[0]) if len(zvr) else np.nan
        for _, r in grp.iterrows():
            note = "" if pd.isna(r["detail"]) else str(r["detail"])
            if block == "spatial_fraction":
                note = f"volume fraction {note}"
            elif block == "spatial_fixed":
                note = f"box sized to ~100 objects (fraction {note})"
            elif block == "spatial_volume":
                note = f"equal box, ~100 vertices (fraction {note})"
            elif block == "chunks":
                note = f"{int(r['n_target']):,} chunks over the domain; {note}"
            elif block == "fragments":
                note = f"crosses {int(r['n_target'])} chunks; {note}"
            rows.append({
                "geometry": GEOM_LABEL[geom],
                "vertices": int(r["n_vertices"]),
                "objects": int(r["n_objects"]),
                "measurement": OP_LABEL.get(op, op),
                "implementation": r["impl"],
                "value": f"{r['value']:.5f} ± {r['ci_hw']:.5f}",
                "per_vertex": "",
                "ratio_to_zarr_vectors": (round(r["value"] / ref, 3)
                                          if ref == ref and ref > 0 else ""),
                "runs": int(r["n_runs"]), "note": note,
            })

    out = pd.DataFrame(rows)
    order = {g: i for i, g in enumerate(GEOM_LABEL.values())}
    out["_g"] = out["geometry"].map(order)
    out = out.sort_values(["_g", "vertices", "measurement", "implementation"])
    return out.drop(columns="_g")


def _to_markdown(table: pd.DataFrame) -> str:
    """Minimal pipe-table renderer -- avoids a ``tabulate`` dependency."""
    cols = list(table.columns)
    cells = [[("" if v != v else str(v)) if not isinstance(v, str) else v
              for v in row] for row in table.itertuples(index=False)]
    widths = [max(len(c), *(len(r[i]) for r in cells)) if cells else len(c)
              for i, c in enumerate(cols)]
    def line(vals):
        return "| " + " | ".join(v.ljust(w) for v, w in zip(vals, widths)) + " |"
    out = [line(cols), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [line(r) for r in cells]
    return "\n".join(out)


def write_supplementary(table, env):
    table.to_csv(RESULTS / "supplementary_table.csv", index=False)
    header = [
        "# Supplementary Table 1",
        "",
        "Every measurement behind the main figure, plus the zarr-vectors",
        "configurations the figure has no room for (uncompressed, and",
        "without the per-object index). Times are the mean of *runs*",
        "repeats with a 95 % Student's-t half-width.",
        "",
        "The whole-dataset and per-object timings sweep the **object**",
        "count with each object held at a fixed size (1 vertex per point,",
        "12 per streamline, 144 per mesh patch); the size, region-query",
        "and chunked-format rows sweep the **vertex** count. The",
        "`vertices` and `objects` columns give both either way.",
        "",
        "`ratio_to_zarr_vectors` is the row divided by the zarr-vectors",
        "value for the same geometry, size and measurement — the Zstd store",
        "for size rows, the deployed store for timing rows. Below 1 means",
        "zarr-vectors was beaten; above 1 means it won.",
        "",
        f"Machine: {env.get('platform')} · {env.get('cpu_count')} logical CPUs · "
        f"Python {env.get('python')}",
        f"zarr-vectors {env.get('zarr_vectors')} (`{env.get('zarr_vectors_git')}`) · "
        f"zarr {env.get('zarr')} · numcodecs {env.get('numcodecs')} · "
        f"nibabel {env.get('nibabel')}",
        f"Domain {env.get('domain')}³ · chunk_shape {tuple(env.get('chunk_shape', ()))} · "
        f"bin_shape {tuple(env.get('bin_shape', ()))} · seed {env.get('seed')}",
        "",
    ]
    md = _to_markdown(table)
    (RESULTS / "supplementary_table.md").write_text(
        "\n".join(header) + "\n" + md + "\n")


# --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=None,
                    help="directory holding measurements.csv (default: ./results)")
    ap.add_argument("--panels", default=None,
                    help="directory for the individual panel files "
                         "(default: <results>/panels)")
    ap.add_argument("--formats", default="png,pdf",
                    help="comma-separated output formats (default: png,pdf)")
    ap.add_argument("--letters", action="store_true",
                    help="stamp A..H on the panels; off by default, since "
                         "the letters belong to the layout they end up in")
    ap.add_argument("--keys", action="store_true",
                    help="also draw each key inside its panel; every key "
                         "is written as its own <panel>_key file either "
                         "way")
    ap.add_argument("--combined", action="store_true",
                    help="also write the eight panels as one sheet, "
                         "figure1.png/pdf")
    ap.add_argument("--no-table", action="store_true",
                    help="skip the supplementary table")
    args = ap.parse_args()
    global RESULTS
    if args.results:
        RESULTS = Path(args.results).resolve()
    panels_dir = Path(args.panels).resolve() if args.panels else RESULTS / "panels"
    formats = tuple(f.strip().lstrip(".") for f in args.formats.split(",")
                    if f.strip())

    df = load()
    env = json.loads((RESULTS / "environment.json").read_text())

    written = render_panels(df, env, panels_dir, formats=formats,
                            letters=args.letters, keys=args.keys)
    for path in written:
        print(f"wrote {path}")

    # The crossings are no longer drawn on the panels, so they are
    # reported here instead: the size at which the zarr-vectors curve
    # crosses its opponent, which is the number the Results section
    # argues from.
    if CROSSINGS:
        print("\ncrossovers (zarr-vectors overtakes to the right):")
        # Deduplicated: --combined draws every panel a second time.
        for panel, label, xc in dict.fromkeys(CROSSINGS):
            print(f"  {panel.replace(chr(10), ' '):38s} {label:14s} "
                  f"{_si(xc):>7s}")

    if args.combined:
        fig = build_figure(df, env)
        for fmt in formats:
            path = RESULTS / f"figure1.{fmt}"
            fig.savefig(path, bbox_inches="tight")
            print(f"wrote {path}")
        plt.close(fig)

    if not args.no_table:
        table = supplementary(df)
        write_supplementary(table, env)
        print(f"wrote {RESULTS / 'supplementary_table.md'} ({len(table)} rows)")
        print(f"wrote {RESULTS / 'supplementary_table.csv'}")


if __name__ == "__main__":
    main()
