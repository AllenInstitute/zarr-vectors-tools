"""Figures for ``bench_kinds.py``: spatial read times for point clouds, meshes and networks.

    python figures_kinds.py [--results results/bench_kinds.json]

* ``zv_kinds_<latency>.png``: query time, one row per kind of data and one
  column per query (all / one vertex / one object / box); in each panel a
  row per store, a bar per reader (geometric mean over that query's sizes
  and placements of the median time);
* ``zv_kinds_bytes.png``: mean bytes read per query at 20 ms, split by array, and
  each store's size on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).parent
# Reference palette (dataviz skill), light mode; colour follows the reader everywhere.
SURFACE, INK, INK2, MUTED, GRID_C = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
READER_COLOR = {"zarr": "#2a78d6", "standard": "#6250d6", "optimised": "#eb6834"}
READER_LABEL = {"zarr": "through zarr's API only", "standard": "zarr-vectors' standard reader (default)",
                "optimised": "zarr-vectors' optimised reader (opt-in)"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "axes.axisbelow": True,
})

KIND_LABEL = {"points": "Point cloud", "mesh": "Mesh", "network": "Network"}
QUERIES = [("all", "All"), ("vertex", "One vertex"), ("object", "Objects"), ("bbox", "Box")]
STORE_LABEL = {"indexed": "sharding_indexed", "octree": "octree\n(100 kB target)", "runs": "octree runs\n(160-640 kB)"}
# Fixed order (adjacent pairs from the validated reference order); metadata and other in grey.
ARRAY_COLOR = {"vertices": "#6250d6", "vertex fragments": "#eda100", "links": "#1baf7a", "object index": "#e87ba4",
               "metadata": MUTED, "other": "#c4c3be"}


def geomean_of_medians(rows: list, field: str = "seconds") -> float:
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["label"], r["size"]), []).append(r[field])
    v = [np.median(x) for x in groups.values()]
    return float(np.exp(np.mean(np.log(np.maximum(v, 1e-9))))) if v else np.nan


def fmt(v: float) -> str:
    return f"{v:,.0f}" if v >= 10 else f"{v:,.1f}" if v >= 1 else f"{v:.2f}"


def lat_name(lat: float) -> str:
    return "local" if not lat else f"{lat * 1e3:.0f}ms"


def fig_time(res: dict, lat: float, out: Path) -> None:
    q = [r for r in res["queries"] if r["latency"] == lat]
    kinds = [k for k in KIND_LABEL if any(r["kind"] == k for r in q)]
    layouts = list(res["setup"]["layouts"])
    fig, axes = plt.subplots(len(kinds), len(QUERIES), figsize=(2.6 * len(QUERIES) + 1.4, 0.75 * len(layouts) * len(kinds) + 1.2),
                             squeeze=False)
    y = np.arange(len(layouts))[::-1].astype(float)
    readers = [m for m in READER_COLOR if any(r["reader"] == m for r in q)]
    h = 0.8 / len(readers)
    for i, kind in enumerate(kinds):
        for j, (qk, qname) in enumerate(QUERIES):
            ax = axes[i, j]
            ax.grid(axis="y", visible=False)
            if i == 0:
                ax.set_title(qname, fontsize=9)
            rows = [r for r in q if r["kind"] == kind and r["query"] == qk]
            if not rows:
                ax.text(0.5, 0.5, "not supported\nby zarr-vectors" if kind in res["setup"]["no_object_reads"] else "not run",
                        ha="center", va="center", fontsize=8, color=MUTED, transform=ax.transAxes)
                ax.set_xticks([])
            top = 0.0
            for yi, layout in zip(y, layouts):
                for k, reader in enumerate(readers):
                    mine = [r for r in rows if r["layout"] == layout and r["reader"] == reader]
                    yy = yi + h * ((len(readers) - 1) / 2 - k)
                    if not mine:
                        if rows and qk == "all" and res["setup"].get("skip_zarr_all"):
                            ax.text(0, yy, " not run (minutes through zarr)", va="center", fontsize=6.5, color=MUTED)
                        continue
                    v = geomean_of_medians(mine) * 1e3
                    ax.barh(yy, v, color=READER_COLOR[reader], height=h * 0.9)
                    ax.text(v * 1.03, yy, fmt(v), va="center", fontsize=7, color=INK2)
                    top = max(top, v)
            if top:
                ax.set_xlim(0, top * 1.4)
                ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            ax.set_ylim(-0.6, len(layouts) - 0.4)
            ax.set_yticks(y, [STORE_LABEL.get(l, l) for l in layouts] if j == 0 else [], fontsize=8)
            if j == 0:
                ax.set_ylabel(KIND_LABEL[kind], fontsize=9.5)
    setup = res["setup"]
    title = "local store" if not lat else f"{lat * 1e3:.0f} ms + {setup['bandwidth'] / 1e6:.0f} MB/s per request"
    fig.suptitle(f"Query time (ms), {title}", fontsize=10, x=0.01, ha="left")
    fig.legend(handles=[Patch(color=READER_COLOR[m], label=READER_LABEL[m]) for m in readers],
               loc="lower left", bbox_to_anchor=(0.0, -0.01), ncol=len(readers), fontsize=8, frameon=False)
    fig.tight_layout(w_pad=0.6, h_pad=0.8, rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_bytes(res: dict, lat: float, out: Path) -> None:
    q = [r for r in res["queries"] if r["latency"] == lat]
    kinds = [k for k in KIND_LABEL if any(r["kind"] == k for r in q)]
    layouts = list(res["setup"]["layouts"])
    bars = [(l, m) for l in layouts for m in READER_COLOR if any(r["reader"] == m for r in q)]
    classes = list(ARRAY_COLOR)
    cols = QUERIES + [("disk", "Store size")]
    fig, axes = plt.subplots(len(kinds), len(cols), figsize=(2.5 * len(cols) + 1.6, 0.42 * len(bars) * len(kinds) + 1.4),
                             squeeze=False)
    for i, kind in enumerate(kinds):
        for j, (qk, qname) in enumerate(cols):
            ax = axes[i, j]
            ax.grid(axis="y", visible=False)
            if i == 0:
                ax.set_title(qname + (" (MB)" if qk == "disk" else " (MB read)"), fontsize=9)
            if qk == "disk":
                labels = layouts
                parts = [res["build"][kind][l]["disk_by_array"] for l in layouts]
            else:
                labels = [f"{STORE_LABEL.get(l, l).splitlines()[0]}, {m}" for l, m in bars]
                parts = []
                for l, m in bars:
                    mine = [r for r in q if r["kind"] == kind and r["query"] == qk and r["layout"] == l and r["reader"] == m]
                    parts.append({c: float(np.mean([(r["bytes_by_array"] or {}).get(c, 0) for r in mine])) for c in classes}
                                 if mine else None)
            y = np.arange(len(labels))[::-1].astype(float)
            if all(p is None for p in parts):
                ax.text(0.5, 0.5, "not supported\nby zarr-vectors", ha="center", va="center", fontsize=8, color=MUTED,
                        transform=ax.transAxes)
                ax.set_xticks([])
            top = 0.0
            for yi, p in zip(y, parts):
                if p is None:
                    continue
                left = 0.0
                for c in classes:
                    v = (p.get(c, 0) or 0) / 1e6
                    if v > 0:
                        ax.barh(yi, v, left=left, color=ARRAY_COLOR[c], height=0.78, edgecolor="#fcfcfb", linewidth=0.6)
                        left += v
                ax.text(left * 1.03, yi, fmt(left), va="center", fontsize=7, color=INK2)
                top = max(top, left)
            if top:
                ax.set_xlim(0, top * 1.35)
                ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            ax.set_ylim(-0.6, len(labels) - 0.4)
            ax.set_yticks(y, labels if j in (0, len(cols) - 1) else [], fontsize=7.5)
            if j == len(cols) - 1:
                ax.yaxis.tick_right()
            if j == 0:
                ax.set_ylabel(KIND_LABEL[kind], fontsize=9.5)
    fig.suptitle(f"Bytes read per query at {lat * 1e3:.0f} ms latency (mean), by array; store size",
                 fontsize=10, x=0.01, ha="left")
    fig.legend(handles=[Patch(color=ARRAY_COLOR[c], label=c) for c in classes], loc="lower left",
               bbox_to_anchor=(0.0, -0.01), ncol=len(classes), fontsize=8, frameon=False)
    fig.tight_layout(w_pad=0.6, h_pad=0.8, rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results" / "bench_kinds.json")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "figures")
    args = ap.parse_args()
    res = json.loads(args.results.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    lats = sorted({r["latency"] for r in res["queries"]})
    for lat in lats:
        fig_time(res, lat, args.out / f"zv_kinds_{lat_name(lat)}.png")
    remote = [l for l in lats if l]
    if remote:
        fig_bytes(res, 0.02 if 0.02 in remote else remote[0], args.out / "zv_kinds_bytes.png")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
