#!/usr/bin/env python
"""Turn ``results/measurements.csv`` into panels and a table.

    python make_figure.py
    python make_figure.py --keys --formats png

Reads only the CSV. Outputs, all under ``results/``:

``panels/X.png`` / ``panels/X.pdf``       one file per panel
``panels/X_key.png`` / ``X_key.pdf``     that panel's key, on its own
``supplementary_table.md`` / ``.csv``    every number behind them

The panels follow ``../paper/make_figure.py`` exactly -- one journal
column each, keys as separate files, milliseconds on the time axes, a
95 % confidence band around every line -- and borrow its helpers
rather than copying them, so a restyle there lands here too.

    A  read all cells, uncompressed      time vs vertices, log-log
    B  read all cells, zstd              time vs vertices, log-log
    C  speed-up over the CPU, uncompressed
    D  speed-up over the CPU, zstd

The colours come from the paper's validated arc and keep its meaning --
blues are the thing being shown, deep magenta is what it is compared
against -- and so do the line styles: solid and filled for the device
decode, dash-dot and filled for the host decode on the GPU, dashed and
open for the CPU.  Every pair clears the paper's all-pairs colour-blind
bar, and the line style carries identity on its own besides.

Speed-up is ``cpu time / gpu time`` at each size, on a linear axis with
a rule at 1x.  Its band is the ratio's first-order 95 % interval, the two
relative half-widths added in quadrature.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# The paper's figure module, under a name that does not shadow this one.
_spec = importlib.util.spec_from_file_location(
    "paper_figure", HERE.parent / "paper" / "make_figure.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

import matplotlib as mpl  # noqa: E402  (configured by the paper module)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

CPU = "cpu"
GPU_HOST = "gpu (host decode)"
GPU_DEV = "gpu (device decode)"

# impl -> (label, colour, linestyle, marker, filled)
STYLE = {
    GPU_DEV: ("GPU, device decode", P.GEOM_COLOR["points"], "-", "o", True),
    GPU_HOST: ("GPU, host decode", P.GEOM_COLOR["meshes"], "-.", "^", True),
    CPU: ("CPU", P.GEOM_COLOR["streamlines"], "--", "s", False),
}
TIME_ORDER = [GPU_DEV, GPU_HOST, CPU]
SPEEDUP_ORDER = [GPU_DEV, GPU_HOST]

CODEC_TITLE = {"none": "uncompressed", "zstd": "zstd"}
X_LABEL = "vertices (125 cells)"


def load(results: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(results / "measurements.csv")
    for col in ("value", "ci_hw", "n_vertices"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ci_hw"] = df["ci_hw"].fillna(0.0)
    env_path = results / "environment.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else {}
    return df, env


def _series(df, codec, impl):
    return df[(df["codec"] == codec) & (df["impl"] == impl)].sort_values(
        "n_vertices")


def _plot(ax, x, y, hw, impl):
    label, color, ls, marker, filled = STYLE[impl]
    ax.plot(x, y, color=color, linestyle=ls, marker=marker, markersize=6.0,
            linewidth=2.25, markerfacecolor=color if filled else "white",
            markeredgecolor=color, markeredgewidth=1.5, label=label, zorder=3)
    P._band(ax, np.asarray(x, float), y, hw, color, 0.16)


def _handles(order):
    out = []
    for impl in order:
        label, color, ls, marker, filled = STYLE[impl]
        out.append(Line2D([], [], color=color, linestyle=ls, marker=marker,
                          markersize=5.1, linewidth=2.1,
                          markerfacecolor=color if filled else "white",
                          markeredgecolor=color, label=label))
    return out


def _axes(ax, ylabel, title, *, ylog):
    ax.set_xscale("log")
    ax.set_yscale("log" if ylog else "linear")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    P._log_grid(ax, xlog=True, ylog=ylog)
    ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: P._si(v)))


def panel_time(ax, df, codec, key=False):
    for impl in TIME_ORDER:
        s = _series(df, codec, impl)
        _plot(ax, s["n_vertices"], s["value"] * P.MS, s["ci_hw"] * P.MS, impl)
    _axes(ax, P.T_LABEL, f"read all cells, {CODEC_TITLE[codec]}", ylog=True)
    handles = _handles(TIME_ORDER)
    if key:
        P._draw_key(ax, handles)
    return handles


def speedup(df, codec, impl) -> pd.DataFrame:
    cpu = _series(df, codec, CPU).set_index("n_vertices")
    gpu = _series(df, codec, impl).set_index("n_vertices")
    both = cpu.index.intersection(gpu.index)
    cpu, gpu = cpu.loc[both], gpu.loc[both]
    ratio = cpu["value"] / gpu["value"]
    rel = np.hypot(cpu["ci_hw"] / cpu["value"], gpu["ci_hw"] / gpu["value"])
    return pd.DataFrame({"n_vertices": both, "speedup": ratio.values,
                         "hw": (ratio * rel).values})


def panel_speedup(ax, df, codec, key=False):
    ax.axhline(1.0, color=P.INK_MUTED, linewidth=0.9, linestyle=":",
               zorder=1)
    for impl in SPEEDUP_ORDER:
        s = speedup(df, codec, impl)
        _plot(ax, s["n_vertices"], s["speedup"], s["hw"], impl)
    _axes(ax, "speed-up over CPU (×)",
          f"GPU speed-up, {CODEC_TITLE[codec]}", ylog=False)
    ax.set_ylim(bottom=0)
    handles = _handles(SPEEDUP_ORDER)
    if key:
        P._draw_key(ax, handles)
    return handles


PANELS = [
    ("A_read_uncompressed", lambda ax, df, k: panel_time(ax, df, "none", k)),
    ("B_read_zstd", lambda ax, df, k: panel_time(ax, df, "zstd", k)),
    ("C_speedup_uncompressed",
     lambda ax, df, k: panel_speedup(ax, df, "none", k)),
    ("D_speedup_zstd", lambda ax, df, k: panel_speedup(ax, df, "zstd", k)),
]


def render_panels(df, outdir, formats=("png", "pdf"), keys=False):
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, draw in PANELS:
        fig, ax = plt.subplots(figsize=P.SIZE_LINE)
        handles = draw(ax, df, keys)
        fig.tight_layout(pad=0.35)
        for fmt in formats:
            path = outdir / f"{stem}.{fmt}"
            fig.savefig(path, bbox_inches="tight")
            written.append(path)
        plt.close(fig)
        written += P._write_key(handles, outdir, f"{stem}_key", formats)
    return written


def build_figure(df, env):
    """Every panel in one lettered sheet, with one key shared below.

    Time on the top row, speed-up on the bottom; uncompressed on the
    left, zstd on the right -- so a column is one codec and a row is one
    question.  Same drawing as the separate panels, letters stamped.
    """
    fig, axes = plt.subplots(2, 2, figsize=(2 * P.SIZE_LINE[0] + 0.4,
                                            2 * P.SIZE_LINE[1] + 0.6),
                             layout="constrained")
    fig.get_layout_engine().set(h_pad=0.08, w_pad=0.12, hspace=0.06,
                                wspace=0.08)
    for ax, (stem, draw) in zip(axes.flat, PANELS):
        draw(ax, df, False)
        P._letter(ax, stem[0])
    handles = _handles(TIME_ORDER) + [
        Line2D([], [], color=P.INK_MUTED, linestyle=":", linewidth=0.9,
               label="speed-up = 1 (no gain)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4,
               frameon=False, labelcolor=P.INK, fontsize=7.5)
    runs = env.get("runs", 5)
    gpu = env.get("gpu", "GPU")
    fig.suptitle(f"read_cells, whole level: CPU vs {gpu}\n"
                 f"mean of {runs} runs, 95 % CI band",
                 fontsize=9.5, color=P.INK)
    return fig


def supplementary(df) -> pd.DataFrame:
    """One row per (codec, N): each mode's mean ± CI, and both speed-ups."""
    rows = []
    for codec in sorted(df["codec"].unique()):
        ups = {impl: speedup(df, codec, impl).set_index("n_vertices")
               for impl in SPEEDUP_ORDER}
        for n in sorted(df.loc[df["codec"] == codec, "n_vertices"].unique()):
            row = {"codec": codec, "n_vertices": int(n)}
            for impl in (CPU, GPU_HOST, GPU_DEV):
                r = _series(df, codec, impl)
                r = r[r["n_vertices"] == n].iloc[0]
                row[f"{impl} (s)"] = r["value"]
                row[f"{impl} ±95% (s)"] = r["ci_hw"]
            for impl in SPEEDUP_ORDER:
                row[f"speed-up, {STYLE[impl][0]}"] = ups[impl].loc[n, "speedup"]
            rows.append(row)
    return pd.DataFrame(rows)


def _to_markdown(table: pd.DataFrame) -> str:
    def fmt(col, v):
        if col in ("codec",):
            return str(v)
        if col == "n_vertices":
            return f"{int(v):,}"
        if col.startswith("speed-up"):
            return f"{v:.2f}×"
        return f"{v * 1e3:.2f} ms"

    head = [c.replace(" (s)", "") for c in table.columns]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join("---" for _ in head) + "|"]
    for _, r in table.iterrows():
        lines.append("| " + " | ".join(fmt(c, r[c]) for c in table.columns)
                     + " |")
    return "\n".join(lines)


def write_supplementary(table, env, results):
    table.to_csv(results / "supplementary_table.csv", index=False)
    where = ", ".join(str(env.get(k)) for k in ("gpu", "processor")
                      if env.get(k))
    body = [
        "# GPU read benchmark -- supplementary table",
        "",
        f"`read_cells` over every cell of a point cloud, "
        f"{' + '.join(env.get('arrays', []))}; mean of "
        f"{env.get('runs', '?')} runs after one warm-up, ±95 % CI "
        f"(Student's t). Measured on {where}; "
        f"zarr_vectors {env.get('zarr_vectors')} "
        f"({env.get('zarr_vectors_git')}), cupy {env.get('cupy')}, "
        f"nvCOMP {env.get('nvcomp')}.",
        "",
        _to_markdown(table),
        "",
    ]
    (results / "supplementary_table.md").write_text("\n".join(body))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--formats", default="png,pdf")
    ap.add_argument("--keys", action="store_true",
                    help="draw each key inside its panel as well")
    args = ap.parse_args()

    results = Path(args.results)
    df, env = load(results)
    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    written = render_panels(df, results / "panels", formats, keys=args.keys)
    fig = build_figure(df, env)
    for fmt in formats:
        path = results / f"figure.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    write_supplementary(supplementary(df), env, results)
    print(f"wrote {len(written)} files (panels, figure) and the "
          f"supplementary table under {results}")


if __name__ == "__main__":
    main()
