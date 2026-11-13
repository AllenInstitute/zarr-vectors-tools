"""Assemble the paper's benchmark figure (panels a-h) from a results CSV.

``make_figure.py`` writes each panel as its own file; this script lays
the same panels out as the paper's figure:

    Data Size by Geometry Types   a  storage per vertex, 1M vertices
                                  b  size relative to the smallest alternative
    Read and write benchmarks     c  write all   d  read all   e  read one
    Impact of Chunk Size          f  read region
                                  g  read one object vs chunk count
    Alternative Format            h  read / write all vs precomputed

Every panel but a runs to the largest size in the results (10^7 in
``results/large``); a shows the 1M-vertex store, as titled.

Panels a-f and h are drawn by ``make_figure``'s own panel functions;
only their titles are the figure's.

The timing block also records variant rows (no codec, by id) under
``zarr-vectors (...)`` labels; they are left out of c-f, which plot the
deployed configuration against the alternative.

Usage:
    python paper_figure.py [--results results/large] [--out figure.pdf]
                           [--panels DIR]
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
TIMING_BLOCKS = ("bulk", "object", "spatial_fixed", "spatial_fraction",
                 "spatial_volume")
BAR_N = 1_000_000              # panel a: bars for a store of 1M vertices
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


def contexts(df, env):
    """(bars, timing) contexts: a at 1M vertices, the rest at every size."""
    bars = df[df["n_target"].isna() | (df["n_target"] <= BAR_N)
              | (df["block"] != "storage")]
    return mf._ctx(bars, env), mf._ctx(main_series(df), env)


def draw_a(ax, c_bars, c_time, df, letter="a"):
    mf._draw_a(ax, c_bars, letter=letter, key=True)
    title(ax, "Storage Size\nstore of 1M vertices")
    ax.tick_params(axis="x", labelrotation=0, labelsize=11)
    for lab in ax.get_xticklabels():
        lab.set_fontweight("bold")
        lab.set_color("0.35")
        lab.set_horizontalalignment("center")
    for t in ax.texts:
        t.set_fontsize(9)
    leg = ax.get_legend()
    if leg is not None:
        ax.legend(handles=leg.legend_handles,
                  labels=[t.get_text() for t in leg.get_texts()],
                  loc="upper left", ncol=2, fontsize=9, frameon=True)


def draw_b(ax, c_bars, c_time, df, letter="b"):
    mf._draw_b(ax, c_time, letter=letter, key=True)
    title(ax, "Relative Size\nZarr Vectors / smallest alternative")
    ax.set_ylabel("ratio of store sizes")
    ax.text(0.02, 1.0, "equal size", transform=ax.get_yaxis_transform(),
            va="top", ha="left", fontsize=8, color="0.45",
            fontweight="bold")


def _timing_panel(draw, name):
    def f(ax, c_bars, c_time, df, letter=None):
        draw(ax, c_time, letter=letter, key=False)
        title(ax, name)
    return f


draw_c = _timing_panel(mf._draw_c, "Write All")
draw_d = _timing_panel(mf._draw_d, "Read All")
draw_e = _timing_panel(mf._draw_e, "Read One")


def draw_f(ax, c_bars, c_time, df, letter="f"):
    mf._draw_f(ax, c_time, letter=letter, key=True)
    title(ax, "Read Region")


def draw_g(ax, c_bars, c_time, df, letter="g"):
    mf._draw_h(ax, c_time, letter=letter, key=True)
    title(ax, "Read one object")


def draw_h(ax, c_bars, c_time, df, letter="h"):
    mf._draw_g(ax, c_time, letter=letter, key=True)
    title(ax, "Read / Write All")


def shared_key(ax):
    """The key shared by c, d and e."""
    ax.axis("off")
    ax.legend(
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
        loc="center", ncol=5, frameon=False, fontsize=10.5,
        handlelength=2.4, columnspacing=1.6)


def _fonts(ax, legend=True):
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)
    ax.tick_params(labelsize=9)
    leg = ax.get_legend()
    if legend and leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(8.5)


# (file stem, figure size, draw function) for the per-panel export.
PANELS = [
    ("a_storage_size", (8.6, 4.4), draw_a),
    ("b_relative_size", (4.4, 4.4), draw_b),
    ("c_write_all", (4.4, 4.2), draw_c),
    ("d_read_all", (4.4, 4.2), draw_d),
    ("e_read_one", (4.4, 4.2), draw_e),
    ("f_read_region", (4.4, 4.2), draw_f),
    ("g_read_one_vs_chunks", (4.4, 4.2), draw_g),
    ("h_vs_precomputed", (4.4, 4.2), draw_h),
]


def write_panels(df, env, out_dir: Path, fmts=("png", "pdf", "svg")):
    """One file per panel (no panel letters), plus the c-e key."""
    out_dir.mkdir(parents=True, exist_ok=True)
    c_bars, c_time = contexts(df, env)
    written = []
    for stem, size, draw in PANELS:
        fig, ax = plt.subplots(figsize=size, constrained_layout=True)
        draw(ax, c_bars, c_time, df, letter=None)
        _fonts(ax, legend=draw is not draw_a)
        for ext in fmts:
            fig.savefig(out_dir / f"{stem}.{ext}", dpi=300)
            written.append(out_dir / f"{stem}.{ext}")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(11.0, 0.6))
    shared_key(ax)
    for ext in fmts:
        fig.savefig(out_dir / f"key_c_d_e.{ext}", dpi=300,
                    bbox_inches="tight")
        written.append(out_dir / f"key_c_d_e.{ext}")
    plt.close(fig)
    return written


def build(df, env):
    c_bars, c_time = contexts(df, env)

    fig = plt.figure(figsize=(13.0, 13.4))
    gs = fig.add_gridspec(
        4, 3, height_ratios=[1.0, 1.08, 0.16, 1.08],
        hspace=0.5, wspace=0.32, left=0.06, right=0.97, top=0.92,
        bottom=0.05)

    # Data size by geometry type.
    ax_a = fig.add_subplot(gs[0, 0:2])
    draw_a(ax_a, c_bars, c_time, df)
    ax_b = fig.add_subplot(gs[0, 2])
    draw_b(ax_b, c_bars, c_time, df)

    # Read and write benchmarks, one key shared under the row.
    row = []
    for col, (draw, letter) in enumerate(
            ((draw_c, "c"), (draw_d, "d"), (draw_e, "e"))):
        ax = fig.add_subplot(gs[1, col])
        draw(ax, c_bars, c_time, df, letter=letter)
        row.append(ax)
    key = fig.add_subplot(gs[2, :])
    shared_key(key)
    key.get_legend().set_loc("upper center")

    # Impact of chunk size.
    ax_f = fig.add_subplot(gs[3, 0])
    draw_f(ax_f, c_bars, c_time, df)
    ax_g = fig.add_subplot(gs[3, 1])
    draw_g(ax_g, c_bars, c_time, df)

    # The alternative format.
    ax_h = fig.add_subplot(gs[3, 2])
    draw_h(ax_h, c_bars, c_time, df)

    for ax, text in ((ax_a, "Data Size by Geometry Types"),
                     (row[0], "Read and write benchmarks"),
                     (ax_f, "Impact of Chunk Size"),
                     (ax_h, "Alternative Format: Precomputed")):
        box = ax.get_position()
        fig.text(box.x0 - 0.035, box.y1 + 0.045, text, **HEADING)
    for ax in (ax_a, ax_b, *row, ax_f, ax_g, ax_h):
        _fonts(ax, legend=ax is not ax_a)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(HERE / "results" / "large"))
    ap.add_argument("--out", default=None,
                    help="output file (default <results>/paper_figure.pdf "
                         "and .png)")
    ap.add_argument("--panels", default=None, metavar="DIR",
                    help="also write each panel as its own file (png, pdf, "
                         "svg) into DIR")
    args = ap.parse_args()
    results = Path(args.results)
    df, env = load(results)
    if args.panels:
        for out in write_panels(df, env, Path(args.panels)):
            print(f"wrote {out}")
    fig = build(df, env)
    outs = ([Path(args.out)] if args.out
            else [results / "paper_figure.pdf", results / "paper_figure.png"])
    for out in outs:
        fig.savefig(out, dpi=200)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
