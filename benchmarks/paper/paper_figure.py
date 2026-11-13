"""Assemble the paper's benchmark figure (panels a-h) from a results CSV.

``make_figure.py`` writes each panel as its own file; this script lays
the same panels out as the paper's figure:

    Data Size by Geometry Types   a  storage per vertex, 1M vertices
                                  b  size relative to the smallest alternative
    Read and write benchmarks     c  write all   d  read all   e  read one
    Impact of Chunk Size          f  read region
                                  g  read one object vs total chunks,
                                     chunked by space and by id
    Alternative Format            h  read / write all vs precomputed

Panels a-f and h are drawn by ``make_figure``'s own panel functions;
only their titles are the figure's.  Panel g is drawn here.

The timing block also records variant rows (no codec, by id) under
``zarr-vectors (...)`` labels; they are left out of c-f, which plot the
deployed configuration against the alternative.

Usage:
    python paper_figure.py [--results results/large] [--out figure.pdf]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import make_figure as mf

HERE = Path(__file__).resolve().parent
ID = "zarr-vectors (chunked by id)"
TIMING_BLOCKS = ("bulk", "object", "spatial_fixed", "spatial_fraction",
                 "spatial_volume")
SMALL_MAX = 1_000_000          # a, b and h stop at 10^6, as in the paper
HEADING = dict(fontsize=15, fontweight="bold", color="0.35", ha="left",
               va="bottom")


def load(results: Path):
    df = mf.pd.read_csv(results / "measurements.csv")
    df["value"] = mf.pd.to_numeric(df["value"], errors="coerce")
    df["ci_hw"] = mf.pd.to_numeric(df["ci_hw"], errors="coerce").fillna(0.0)
    for col in ("n_vertices", "n_objects", "n_target"):
        df[col] = mf.pd.to_numeric(df[col], errors="coerce")
    env_path = results / "environment.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else {}
    return df, env


def main_series(df):
    """Timing rows of the deployed configuration and the alternatives."""
    variant = (df["block"].isin(TIMING_BLOCKS)
               & df["impl"].str.startswith("zarr-vectors (", na=False))
    return df[~variant]


def title(ax, text):
    ax.set_title(text, fontsize=11, fontweight="bold", color="0.35")


def panel_g(ax, df, letter="g"):
    """Read one object vs total chunks: chunked by space and by id."""
    sub = df[(df["block"] == "chunks") & (df["op"] == "fetch_one")]
    for g in mf.TIMED:
        color = mf.GEOM_COLOR[g]
        for impl, ls, mk in (("zarr-vectors", "-", "o"), (ID, "-.", "^")):
            s_ = sub[(sub["geometry"] == g) & (sub["impl"] == impl)]
            s_ = s_.sort_values("n_target")
            if not len(s_):
                continue
            x = s_["n_target"].to_numpy(float)
            y = s_["value"].to_numpy(float) * mf.MS
            ax.plot(x, y, color=color, linestyle=ls, marker=mk,
                    markersize=6.0, linewidth=2.25, markerfacecolor=color,
                    markeredgecolor=color, zorder=3)
            mf._band(ax, x, y, s_["ci_hw"].to_numpy(float) * mf.MS, color,
                     0.16)
    mf._axes(ax, "total chunks", mf.T_LABEL, "", letter)
    handles = [Line2D([], [], color=mf.GEOM_COLOR[g], marker="o",
                      linewidth=2.1, markersize=5.1, label=mf.GEOM_LABEL[g])
               for g in mf.TIMED]
    handles += [
        Line2D([], [], color="0.35", linestyle="-", marker="o",
               linewidth=2.1, markersize=5.1, label="chunked by space"),
        Line2D([], [], color="0.35", linestyle="-.", marker="^",
               linewidth=2.1, markersize=5.1, label="chunked by id"),
    ]
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 6)          # headroom for the key
    mf._draw_key(ax, handles, loc="upper center", ncol=2)


def build(df, env):
    timing = main_series(df)
    small = df[df["n_target"].isna() | (df["n_target"] <= SMALL_MAX)
               | ~df["block"].isin(["storage", "mesh_precomputed"])]
    c_small = mf._ctx(small, env)
    c_time = mf._ctx(timing, env)
    # h compares against precomputed annotations, measured up to 10^6.
    c_pc = mf._ctx(timing[timing["n_target"].isna()
                          | (timing["n_target"] <= SMALL_MAX)], env)

    fig = plt.figure(figsize=(13.0, 13.4))
    gs = fig.add_gridspec(
        4, 3, height_ratios=[1.0, 1.08, 0.16, 1.08],
        hspace=0.5, wspace=0.32, left=0.06, right=0.97, top=0.92,
        bottom=0.05)

    # Data size by geometry type.
    ax_a = fig.add_subplot(gs[0, 0:2])
    mf._draw_a(ax_a, c_small, letter="a", key=True)
    title(ax_a, "Storage Size\nstore of 1M vertices")
    ax_a.tick_params(axis="x", labelrotation=0, labelsize=11)
    for lab in ax_a.get_xticklabels():
        lab.set_fontweight("bold")
        lab.set_color("0.35")
        lab.set_horizontalalignment("center")
    for t in ax_a.texts:
        t.set_fontsize(9)
    leg = ax_a.get_legend()
    if leg is not None:
        ax_a.legend(handles=leg.legend_handles,
                    labels=[t.get_text() for t in leg.get_texts()],
                    loc="upper left", ncol=2, fontsize=9, frameon=True)
    ax_b = fig.add_subplot(gs[0, 2])
    mf._draw_b(ax_b, c_small, letter="b", key=True)
    title(ax_b, "Relative Size\nZarr Vectors / smallest alternative")
    ax_b.set_ylabel("ratio of store sizes")
    ax_b.text(0.02, 1.0, "equal size", transform=ax_b.get_yaxis_transform(),
              va="top", ha="left", fontsize=8, color="0.45",
              fontweight="bold")

    # Read and write benchmarks, one key shared under the row.
    for col, (draw, name, letter) in enumerate(
            ((mf._draw_c, "Write All", "c"), (mf._draw_d, "Read All", "d"),
             (mf._draw_e, "Read One", "e"))):
        ax = fig.add_subplot(gs[1, col])
        draw(ax, c_time, letter=letter, key=False)
        title(ax, name)
    key = fig.add_subplot(gs[2, :])
    key.axis("off")
    key.legend(
        handles=[
            Patch(color=mf.GEOM_COLOR["points"],
                  label="Point cloud (single vertices)"),
            Patch(color=mf.GEOM_COLOR["streamlines"],
                  label="Streamlines (12 vertices each)"),
            Patch(color=mf.GEOM_COLOR["meshes"],
                  label="Meshes (144 vertices each)"),
            Line2D([], [], color="0.15", marker="o", linewidth=2.1,
                   label="Zarr Vectors"),
            Line2D([], [], color="0.15", linestyle="--", marker="s",
                   markerfacecolor="white", linewidth=2.1,
                   label="Alternative"),
        ],
        loc="upper center", ncol=5, frameon=False, fontsize=10.5,
        handlelength=2.4, columnspacing=1.6)

    # Impact of chunk size.
    ax_f = fig.add_subplot(gs[3, 0])
    mf._draw_f(ax_f, c_time, letter="f", key=True)
    title(ax_f, "Read Region")
    ax_g = fig.add_subplot(gs[3, 1])
    panel_g(ax_g, df)
    title(ax_g, "Read one object")

    # The alternative format.
    ax_h = fig.add_subplot(gs[3, 2])
    mf._draw_g(ax_h, c_pc, letter="h", key=True)
    title(ax_h, "Read / Write All")

    for ax, text in ((ax_a, "Data Size by Geometry Types"),
                     (fig.axes[2], "Read and write benchmarks"),
                     (ax_f, "Impact of Chunk Size"),
                     (ax_h, "Alternative Format: Precomputed")):
        box = ax.get_position()
        fig.text(box.x0 - 0.035, box.y1 + 0.045, text, **HEADING)
    for ax in (ax_a, ax_b, *fig.axes[2:5], ax_f, ax_g, ax_h):
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)
        ax.tick_params(labelsize=9)
        leg = ax.get_legend()
        if leg is not None and ax is not ax_a:
            for t in leg.get_texts():
                t.set_fontsize(8.5)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(HERE / "results" / "large"))
    ap.add_argument("--out", default=None,
                    help="output file (default <results>/paper_figure.pdf "
                         "and .png)")
    args = ap.parse_args()
    results = Path(args.results)
    df, env = load(results)
    fig = build(df, env)
    outs = ([Path(args.out)] if args.out
            else [results / "paper_figure.pdf", results / "paper_figure.png"])
    for out in outs:
        fig.savefig(out, dpi=200)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
